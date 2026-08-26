"""Paper2Gate CLI — extract innovations from PDF papers and generate Verilog."""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from src.console import console
from rich.panel import Panel
from rich.table import Table

from src.code_generator import (
    fix_from_verification,
    generate_verilog,
    generate_with_check,
    save_error_logs,
    save_verilog,
)
from src.innovation_extractor import extract_innovations, save_specs
from src.integration_generator import generate_butterfly, save_butterfly
from src.llm_client import LLMClient
from src.pdf_parser import extract_text
from src.verification_runner import verify_module
from src.vivado_report import run_synthesis


def build_output_dir(pdf_path: str, base_dir: str = "outputs") -> Path:
    """Create a timestamped output directory for this run."""
    paper_name = Path(pdf_path).stem.replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(base_dir) / f"{paper_name}_{timestamp}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def run_pipeline(
    pdf_path: str,
    config_path: str = "config.yaml",
    output_base: str = "outputs",
    module_filter: str | None = None,
    dry_run: bool = False,
    use_review: bool = True,
    use_integrate: bool = True,
    use_verify: bool = True,
    use_synth: bool = True,
    target_interface: str | None = None,
) -> Path:
    """Run the full paper-to-Verilog pipeline.

    Args:
        target_interface: If "modmul", extract and generate a modmul module
                          with the fixed interface (clk,rst,A,B→R). The
                          intmul+P_R front-end is fixed; only the modular
                          reduction logic is generated from the paper.
                          Auto-skips butterfly integration.
    """
    project_root = Path(__file__).parent.parent

    # Resolve paths relative to project root
    if not Path(config_path).is_absolute():
        config_path = str(project_root / config_path)
    if not Path(output_base).is_absolute():
        output_base = str(project_root / output_base)

    # Stage 0: Parse PDF
    console.print(Panel.fit("[bold]Stage 0: Parsing PDF[/bold]", style="blue"))
    with console.status(f"Extracting text from {pdf_path}..."):
        paper_text = extract_text(pdf_path)
    console.print(f"  Extracted [green]{len(paper_text):,}[/green] characters")

    out_dir = build_output_dir(pdf_path, output_base)

    # Save extracted text for reference
    text_dir = out_dir / "extracted_text"
    text_dir.mkdir(parents=True, exist_ok=True)
    (text_dir / "full_text.txt").write_text(paper_text, encoding="utf-8")

    # Init LLM client
    client = LLMClient(config_path)

    # Stage 1: Extract innovations
    console.print(Panel.fit("[bold]Stage 1: Extracting Innovations[/bold]", style="blue"))
    with console.status("Analyzing paper with LLM (this may take 30-60s)..."):
        analysis = extract_innovations(paper_text, client, target_interface=target_interface)

    spec_path = save_specs(analysis, out_dir)
    console.print(f"  Found [green]{len(analysis.innovations)}[/green] innovations")
    console.print(f"  Specs saved to: {spec_path}")

    # Display innovations table
    table = Table(title="Identified Innovations")
    table.add_column("#", style="dim")
    table.add_column("Module", style="cyan")
    table.add_column("Category", style="yellow")
    table.add_column("Summary")
    for i, mod in enumerate(analysis.innovations, 1):
        table.add_row(str(i), mod.module_name, mod.category, mod.summary[:80])
    console.print(table)

    if not analysis.innovations:
        console.print("[yellow]No innovations found in the paper.[/yellow]")
        return out_dir

    # Stage 2: Generate Verilog for each module
    if dry_run:
        console.print("[yellow]--dry-run: skipping Verilog generation.[/yellow]")
        return out_dir

    console.print(Panel.fit("[bold]Stage 2: Generating Verilog[/bold]", style="blue"))
    verilog_dir = out_dir / "verilog"
    review_dir = out_dir / "review_reports"
    verilog_dir.mkdir(parents=True, exist_ok=True)
    modmul_paths: list[Path] = []  # collect paths for Stage 3

    for i, module_spec in enumerate(analysis.innovations, 1):
        if module_filter and module_filter not in module_spec.module_name:
            continue

        console.print(
            f"  [{i}/{len(analysis.innovations)}] "
            f"[cyan]{module_spec.module_name}[/cyan]..."
        )

        with console.status(f"  Working on {module_spec.module_name}..."):
            if use_review:
                result, error_logs = generate_with_check(
                    module_spec, client, target_interface=target_interface,
                )
                save_error_logs(error_logs, module_spec.module_name, review_dir)
            else:
                result = generate_verilog(
                    module_spec, client, target_interface=target_interface,
                )

        file_path = save_verilog(result, verilog_dir)
        modmul_paths.append(file_path)
        console.print(f"       Saved to: {file_path}")

        # Golden model verification + fix loop
        if use_verify and module_spec.category == "modular_arithmetic":
            verify_type = "modmul"  # modmul has A,B→R, works with standard testbench
            verify_files = [str(file_path)]
            # The module's output may carry a constant factor k relative to a
            # true a*b mod q. K-reduction designs (paper Algorithm 3) compute
            # k*a*b mod q with k=13, cancelled in the NTT by pre-scaling twiddles
            # with k^-1. The golden model MUST use the same k, or a faithful
            # implementation is wrongly failed (and a plain `%` shortcut wrongly
            # passes). correction_factor comes from Stage-1 extraction.
            k_factor = module_spec.hardware_spec.correction_factor
            if target_interface == "modmul":
                verify_params = {"DATA_WIDTH": "12", "Q": "3329"}
            else:
                verify_params = module_spec.hardware_spec.parameters
            max_verify_rounds = 3
            for v_round in range(1, max_verify_rounds + 1):
                console.print(f"       Golden model check (round {v_round}/{max_verify_rounds})...")
                try:
                    passed, failures = verify_module(
                        verify_files,
                        module_spec.module_name,
                        verify_type,
                        hardware_spec_params=verify_params,
                        num_vectors=4,
                        k_factor=k_factor,
                    )
                except ValueError as e:
                    console.print(f"       [yellow]Golden model: {e} — skipping verification[/yellow]")
                    break
                if passed:
                    break
                if v_round < max_verify_rounds:
                    console.print(f"       [yellow]Fixing functional bugs...[/yellow]")
                    try:
                        result = fix_from_verification(
                            module_spec, result.verilog_code, failures, client
                        )
                        # Re-save fixed code
                        file_path = save_verilog(result, verilog_dir)
                        console.print(f"       Re-saved to: {file_path}")
                    except Exception as e:
                        console.print(f"       [red]Fix failed (API error): {e}[/red]")
                        break
                else:
                    console.print(f"       [red]Verification still failing after {max_verify_rounds} rounds[/red]")

    # Stage 3: Integrate into complete butterfly
    bfly_path = None
    if target_interface == "modmul":
        console.print("[yellow]--modmul: skipping butterfly integration (use reference butterfly).[/yellow]")
        use_integrate = False
    elif not use_integrate:
        console.print("[yellow]--no-integrate: skipping butterfly integration.[/yellow]")
    else:
        console.print(Panel.fit("[bold]Stage 3: Integrating into Butterfly[/bold]", style="blue"))
        butterfly_dir = out_dir / "butterfly"
        with console.status("Generating butterfly top-level..."):
            butterfly = generate_butterfly(
                analysis,
                modmul_paths[0] if modmul_paths else None,
                client,
            )

        bfly_path = save_butterfly(butterfly, butterfly_dir)
        console.print(f"  Butterfly saved to: {bfly_path}")

    # Stage 3b: Vivado resource synthesis (runs for both --modmul and normal modes)
    if use_synth and modmul_paths:
        console.print(Panel.fit("[bold]Stage 3b: Resource Estimation (Vivado)[/bold]", style="blue"))
        if target_interface == "modmul":
            # --modmul mode: two synthesis runs
            # Run 1: modmul alone
            vfiles_modmul = [str(p) for p in modmul_paths]
            run_synthesis(vfiles_modmul, top="modmul", output_dir=out_dir)

            # Run 2: full butterfly (reference files + generated modmul)
            ref_dir = project_root / "reference"
            ref_files = ["butterfly.v", "div2.v", "modadd.v", "modsub.v"]
            vfiles_bfly = [str(p) for p in modmul_paths]
            for rf in ref_files:
                rf_path = ref_dir / rf
                if rf_path.exists():
                    vfiles_bfly.append(str(rf_path))
            run_synthesis(vfiles_bfly, top="butterfly", output_dir=out_dir)
        else:
            vfiles = [str(p) for p in modmul_paths] + [str(bfly_path)]
            top_module = "butterfly"
            run_synthesis(vfiles, top=top_module, output_dir=out_dir)

    console.print(f"\n[bold green]Done![/bold green] All outputs in: {out_dir}")
    return out_dir


def expand_pdf_paths(pdf_args: list[str]) -> list[str]:
    """Expand directory arguments into the PDF files they contain.

    Directories expand to their sorted *.pdf contents; explicit file
    arguments keep their given position.
    """
    expanded: list[str] = []
    for arg in pdf_args:
        p = Path(arg)
        if p.is_dir():
            pdfs = sorted(str(f) for f in p.glob("*.pdf"))
            if not pdfs:
                console.print(f"[yellow]Warning: no PDFs found in directory: {arg}[/yellow]")
            expanded.extend(pdfs)
        else:
            expanded.append(arg)
    return expanded


def main():
    parser = argparse.ArgumentParser(
        description="Paper2Gate — Generate Verilog from academic papers using LLM",
    )
    parser.add_argument(
        "pdf", nargs="+",
        help="Path to the PDF paper. Multiple papers (or directories of PDFs) "
             "are processed sequentially; one failing paper does not stop the rest.",
    )
    parser.add_argument(
        "-c", "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "-o", "--output",
        default="outputs",
        help="Output directory (default: outputs/)",
    )
    parser.add_argument(
        "-m", "--module",
        default=None,
        help="Only generate specific module (filter by name)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only extract innovations, skip Verilog generation",
    )
    parser.add_argument(
        "--no-review",
        action="store_true",
        help="Skip the self-review + auto-fix loop",
    )
    parser.add_argument(
        "--no-integrate",
        action="store_true",
        help="Skip butterfly integration (only generate individual modules)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip golden model functional verification",
    )
    parser.add_argument(
        "--no-synth",
        action="store_true",
        help="Skip Vivado resource synthesis",
    )
    parser.add_argument(
        "--modmul",
        action="store_true",
        help="Generate modmul module with fixed interface (clk,rst,A[11:0],B[11:0] -> R[11:0]). "
             "Only the paper's reduction logic is generated; intmul + P_R register are fixed. "
             "Implies --no-integrate --no-synth.",
    )

    args = parser.parse_args()

    pdfs = expand_pdf_paths(args.pdf)
    if not pdfs:
        console.print("[red]Error: no PDFs to process.[/red]")
        sys.exit(1)

    missing = [p for p in pdfs if not os.path.exists(p)]
    if missing:
        console.print("[red]Error: PDF not found:[/red] " + ", ".join(missing))
        sys.exit(1)

    results: list[tuple[str, str | None, str | None]] = []  # (paper, out_dir, error)
    for i, pdf in enumerate(pdfs, 1):
        console.print(
            Panel.fit(f"[bold]Paper {i}/{len(pdfs)}: {pdf}[/bold]", style="cyan")
        )
        try:
            out_dir = run_pipeline(
                pdf_path=pdf,
                config_path=args.config,
                output_base=args.output,
                module_filter=args.module,
                dry_run=args.dry_run,
                use_review=not args.no_review,
                use_integrate=not args.no_integrate,
                use_verify=not args.no_verify,
                use_synth=not args.no_synth,
                target_interface="modmul" if args.modmul else None,
            )
            results.append((pdf, str(out_dir), None))
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user[/yellow]")
            sys.exit(0)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            results.append((pdf, None, str(e)))

    # Batch summary
    table = Table(title="Batch Results", show_header=True)
    table.add_column("#", justify="right")
    table.add_column("Paper", overflow="fold")
    table.add_column("Status")
    table.add_column("Outputs")
    for i, (paper, out_dir, err) in enumerate(results, 1):
        if err:
            table.add_row(str(i), paper, "[red]FAILED[/red]", f"[red]{err}[/red]")
        else:
            table.add_row(str(i), paper, "[green]OK[/green]", out_dir or "")
    console.print(table)

    ok = sum(1 for _, _, e in results if e is None)
    failed = len(results) - ok
    console.print(f"[bold green]{ok}/{len(results)} papers OK[/bold green]" +
                  (f", [bold red]{failed} failed[/bold red]" if failed else ""))
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
