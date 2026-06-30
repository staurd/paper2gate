"""Paper2Gate CLI — extract innovations from PDF papers and generate Verilog."""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
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

console = Console()


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
) -> Path:
    """Run the full paper-to-Verilog pipeline."""
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
        analysis = extract_innovations(paper_text, client)

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
                result, error_logs = generate_with_check(module_spec, client)
                save_error_logs(error_logs, module_spec.module_name, review_dir)
            else:
                result = generate_verilog(module_spec, client)

        file_path = save_verilog(result, verilog_dir)
        modmul_paths.append(file_path)
        console.print(f"       Saved to: {file_path}")

        # Golden model verification + fix loop
        if use_verify and module_spec.category == "modular_arithmetic":
            max_verify_rounds = 3
            for v_round in range(1, max_verify_rounds + 1):
                console.print(f"       Golden model check (round {v_round}/{max_verify_rounds})...")
                try:
                    passed, failures = verify_module(
                        [str(file_path)],
                        module_spec.module_name,
                        "modmul",
                        hardware_spec_params=module_spec.hardware_spec.parameters,
                        num_vectors=4,
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
    if not use_integrate:
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

        # Vivado resource synthesis
        if use_synth:
            console.print(Panel.fit("[bold]Stage 3b: Resource Estimation (Vivado)[/bold]", style="blue"))
            vfiles = list(modmul_paths) + [bfly_path]
            run_synthesis(
                [str(p) for p in vfiles],
                top="butterfly",
                output_dir=out_dir,
            )

    console.print(f"\n[bold green]Done![/bold green] All outputs in: {out_dir}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(
        description="Paper2Gate — Generate Verilog from academic papers using LLM",
    )
    parser.add_argument("pdf", help="Path to the PDF paper")
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

    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        console.print(f"[red]Error: PDF not found: {args.pdf}[/red]")
        sys.exit(1)

    try:
        out_dir = run_pipeline(
            pdf_path=args.pdf,
            config_path=args.config,
            output_base=args.output,
            module_filter=args.module,
            dry_run=args.dry_run,
            use_review=not args.no_review,
            use_integrate=not args.no_integrate,
            use_verify=not args.no_verify,
            use_synth=not args.no_synth,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()
