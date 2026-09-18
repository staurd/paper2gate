"""Paper2Gate CLI: extract modular-arithmetic designs and generate Verilog."""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from rich.panel import Panel
from rich.table import Table

from src.code_generator import (
    check_verilog,
    fix_from_verification,
    generate_modmul,
    generate_with_check,
    iverilog_available,
    save_error_logs,
    save_verilog,
)
from src.config import PROJECT_ROOT, get_iverilog_config, get_output_config, get_vivado_config
from src.console import console
from src.innovation_extractor import classify_scheme, extract_innovations, save_specs, trim_references
from src.ir_models import ModuleSpec, parse_module_params
from src.llm_client import LLMClient
from src.pdf_parser import extract_text
from src.schemes import resolve_scheme
from src.verification_runner import verify_module
from src.verilog_utils import validate_modmul_interface
from src.vivado_report import (
    DEFAULT_PART,
    resolve_part,
    run_synthesis,
    vivado_available,
)


class PipelineError(RuntimeError):
    """A required pipeline stage failed."""


LLM_TIMING_OPERATIONS = (
    "classify_scheme",
    "extract_innovations",
    "generate_modmul",
    "fix_syntax",
    "fix_function",
)


def _empty_llm_timing() -> dict:
    return {
        "llm_seconds": 0.0,
        "llm_calls": 0,
        "llm_by_operation": {
            operation: {"calls": 0, "seconds": 0.0}
            for operation in LLM_TIMING_OPERATIONS
        },
    }


def _update_timing(summary: dict, pipeline_started: float, client: LLMClient | None) -> None:
    timing = _empty_llm_timing()
    if client is not None:
        try:
            client_timing = client.timing_snapshot()
        except (AttributeError, TypeError, ValueError):
            client_timing = None
        if isinstance(client_timing, dict):
            timing.update(client_timing)
    timing["pipeline_seconds"] = round(time.perf_counter() - pipeline_started, 3)
    summary["timing"] = timing


def build_output_dir(pdf_path: str, base_dir: str = "outputs") -> Path:
    """Create a timestamped output directory for this run."""
    paper_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(pdf_path).stem).strip("_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(base_dir) / f"{paper_name}_{timestamp}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def resolve_synth_part(cli_part: str | None, extracted_device: str) -> str:
    """Resolve the Vivado part using CLI, config, paper, then project default."""
    if cli_part:
        console.print(f"  Synthesis FPGA part: [cyan]{cli_part}[/cyan] (--part)")
        return cli_part

    cfg_part = str(get_vivado_config().get("part") or "").strip()
    if cfg_part:
        console.print(f"  Synthesis FPGA part: [cyan]{cfg_part}[/cyan] (config.yaml)")
        if extracted_device:
            console.print(
                f"    [dim]Paper reports {extracted_device} — config override wins[/dim]"
            )
        return cfg_part

    part, provenance = resolve_part(extracted_device)
    if provenance == "paper":
        console.print(f"  Synthesis FPGA part: [cyan]{part}[/cyan] (from paper)")
    elif provenance == "completed":
        console.print(
            f"  Synthesis FPGA part: [cyan]{part}[/cyan] "
            f"(paper reports '{extracted_device}' -> package/speed grade assumed, "
            f"override with --part)"
        )
    elif provenance == "unknown":
        console.print(
            f"  Synthesis FPGA part: [yellow]{part}[/yellow] (default — paper's "
            f"'{extracted_device}' not recognised, override with --part)"
        )
    else:
        console.print(
            f"  Synthesis FPGA part: [yellow]{part}[/yellow] "
            f"(default — paper states no device)"
        )
    return part


def _set_stage(summary: dict, name: str, status: str, detail: str = "") -> None:
    entry = {"status": status}
    if detail:
        entry["detail"] = detail
    summary["stages"][name] = entry


def _write_summary(output_dir: Path, summary: dict) -> Path:
    path = output_dir / "run_summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _resolve_output_base(output_base: str | None) -> Path:
    configured = output_base or str(get_output_config().get("base_dir") or "outputs")
    path = Path(configured)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _select_specs(
    innovations: list[ModuleSpec],
) -> list[ModuleSpec]:
    selected = [spec for spec in innovations if spec.category == "modular_arithmetic"]
    if len(selected) != 1:
        raise PipelineError(
            "The paper must yield exactly one modular_arithmetic innovation; "
            f"found {len(selected)}"
        )
    if selected[0].module_name != "modmul":
        raise PipelineError(
            "The extracted module name must be 'modmul', "
            f"got {selected[0].module_name!r}"
        )
    return selected


def run_pipeline(
    pdf_path: str,
    output_base: str | None = None,
    dry_run: bool = False,
    use_review: bool = True,
    use_verify: bool = True,
    use_synth: bool = True,
    scheme: str | None = None,
    part: str | None = None,
) -> Path:
    """Run the paper-to-Verilog pipeline and write a machine-readable summary."""
    output_dir: Path | None = None
    client: LLMClient | None = None
    pipeline_started = time.perf_counter()
    summary = {
        "paper": str(Path(pdf_path).resolve()),
        "status": "running",
        "scheme": None,
        "timing": {"pipeline_seconds": 0.0, **_empty_llm_timing()},
        "tools": {
            "iverilog": get_iverilog_config().get("binary"),
            "vivado": get_vivado_config().get("binary"),
        },
        "stages": {},
    }

    try:
        console.print(Panel.fit("[bold]Stage 0: Parsing PDF[/bold]", style="blue"))
        with console.status(f"Extracting text from {pdf_path}..."):
            paper_text = extract_text(pdf_path)
        if not paper_text.strip():
            raise PipelineError("PDF contains no extractable text; OCR is required")
        console.print(f"  Extracted [green]{len(paper_text):,}[/green] characters")
        _set_stage(summary, "parse_pdf", "passed")

        output_dir = build_output_dir(pdf_path, str(_resolve_output_base(output_base)))
        text_dir = output_dir / "extracted_text"
        text_dir.mkdir(parents=True, exist_ok=True)
        (text_dir / "full_text.txt").write_text(paper_text, encoding="utf-8")

        body_text, dropped_refs = trim_references(paper_text)
        if dropped_refs:
            console.print(
                f"  Trimmed {dropped_refs:,} chars of references "
                f"({len(paper_text):,} -> {len(body_text):,})"
            )

        client = LLMClient()
        if scheme in (None, "auto"):
            try:
                with console.status("Classifying target scheme with LLM..."):
                    scheme_name, evidence = classify_scheme(body_text, client)
                summary["scheme"] = {"name": scheme_name, "source": "llm", "evidence": evidence}
                if scheme_name in ("unknown", "mixed"):
                    raise ValueError(f"target scheme is {scheme_name}: {evidence}")
                profile = resolve_scheme(scheme_name)
            except (ValueError, RuntimeError) as exc:
                detail = f"Scheme classification failed: {exc}. Use --scheme kyber or --scheme dilithium."
                _set_stage(summary, "classify_scheme", "failed", detail)
                raise PipelineError(detail) from exc
            source = "llm"
            _set_stage(summary, "classify_scheme", "passed", evidence)
        else:
            profile = resolve_scheme(scheme)
            source, evidence = "cli", ""
            _set_stage(summary, "classify_scheme", "skipped", "--scheme override")
        summary["scheme"] = {
            "name": profile.name,
            "q": profile.q,
            "data_width": profile.data_width,
            "source": source,
            "evidence": evidence,
        }
        console.print(
            f"  Selected scheme: [cyan]{profile.name}[/cyan] ({source}, "
            f"q={profile.q}, DATA_WIDTH={profile.data_width}, "
            f"latency default={profile.default_latency})"
        )

        console.print(Panel.fit("[bold]Stage 1: Extracting Innovations[/bold]", style="blue"))
        with console.status("Analyzing paper with LLM ..."):
            analysis = extract_innovations(
                body_text,
                client,
                scheme=profile,
            )
        spec_path = save_specs(analysis, output_dir)
        _set_stage(summary, "extract", "passed", str(spec_path))
        console.print(f"  Found [green]{len(analysis.innovations)}[/green] innovations")
        console.print(f"  Specs saved to: {spec_path}")

        table = Table(title="Identified Innovations")
        table.add_column("#", style="dim")
        table.add_column("Module", style="cyan")
        table.add_column("Category", style="yellow")
        table.add_column("Summary")
        for i, mod in enumerate(analysis.innovations, 1):
            table.add_row(str(i), mod.module_name, mod.category, mod.summary[:80])
        console.print(table)

        if not analysis.innovations:
            _set_stage(summary, "generate", "skipped", "No innovations found")
            summary["status"] = "passed"
            _update_timing(summary, pipeline_started, client)
            _write_summary(output_dir, summary)
            console.print("[yellow]No innovations found in the paper.[/yellow]")
            return output_dir

        if dry_run:
            _set_stage(summary, "generate", "skipped", "--dry-run")
            _set_stage(summary, "verification", "skipped", "--dry-run")
            _set_stage(summary, "synthesis_modmul", "skipped", "--dry-run")
            summary["status"] = "passed"
            _update_timing(summary, pipeline_started, client)
            _write_summary(output_dir, summary)
            console.print("[yellow]--dry-run: skipping Verilog generation.[/yellow]")
            return output_dir

        selected_specs = _select_specs(analysis.innovations)
        synth_part = resolve_synth_part(part, analysis.fpga_device) if use_synth else None
        summary["synthesis_part"] = synth_part

        console.print(Panel.fit("[bold]Stage 2: Generating Verilog[/bold]", style="blue"))
        verilog_dir = output_dir / "verilog"
        review_dir = output_dir / "review_reports"
        generated: list[tuple[ModuleSpec, Path, object]] = []
        generation_unverified = False

        for i, module_spec in enumerate(selected_specs, 1):
            console.print(
                f"  [{i}/{len(selected_specs)}] "
                f"[cyan]{module_spec.module_name}[/cyan]..."
            )
            if use_review:
                result, error_logs = generate_with_check(
                    module_spec,
                    client,
                    scheme=profile,
                )
                save_error_logs(error_logs, module_spec.module_name, review_dir)
            else:
                result = generate_modmul(
                    module_spec,
                    client,
                    scheme=profile,
                )

            interface_errors = validate_modmul_interface(
                result.verilog_code, profile.data_width
            )
            if interface_errors:
                raise PipelineError(
                    f"Generated modmul failed interface validation: "
                    f"{'; '.join(interface_errors)}"
                )

            file_path = save_verilog(result, verilog_dir)
            generated.append((module_spec, file_path, result))
            console.print(f"       Saved to: {file_path}")

            if use_review:
                if not iverilog_available():
                    generation_unverified = True
                else:
                    syntax_ok, syntax_error = check_verilog(result.verilog_code)
                    if not syntax_ok:
                        raise PipelineError(
                            f"Icarus validation failed for {module_spec.module_name}: "
                            f"{syntax_error}"
                        )

        _set_stage(
            summary,
            "generate",
            "unverified" if generation_unverified else "passed",
            "Icarus unavailable" if generation_unverified else "",
        )

        verification_unverified = False
        verification_failed = False
        for module_spec, file_path, result in generated:
            if not use_verify:
                continue
            if not iverilog_available():
                verification_unverified = True
                continue

            k_factor = module_spec.hardware_spec.correction_factor
            raw_params = module_spec.hardware_spec.parameters
            parsed = parse_module_params(raw_params)
            has_dw = any(k.upper() in ("DATA_WIDTH", "WIDTH", "N_BITS", "DW") for k in raw_params)
            has_q = any(k.upper() in ("Q", "MODULUS", "MODULUS_Q", "MOD") for k in raw_params)
            if (has_dw and parsed["DATA_WIDTH"] != profile.data_width) or (has_q and parsed["Q"] != profile.q):
                console.print(
                    f"       [yellow]Warning: extracted width/Q "
                    f"({parsed['DATA_WIDTH']}/{parsed['Q']}) conflicts with "
                    f"{profile.name}; using scheme constants[/yellow]"
                )
            verify_params = {"DATA_WIDTH": str(profile.data_width), "Q": str(profile.q)}
            latency_cycles = module_spec.hardware_spec.latency_cycles
            latency = latency_cycles if 0 < latency_cycles <= 32 else profile.default_latency

            passed = False
            for v_round in range(1, 4):
                console.print(f"       Golden model check (round {v_round}/3)...")
                try:
                    passed, failures = verify_module(
                        [str(file_path)],
                        module_spec.module_name,
                        hardware_spec_params=verify_params,
                        num_vectors=32,
                        latency=latency,
                        k_factor=k_factor,
                    )
                except (OSError, FileNotFoundError) as exc:
                    verification_unverified = True
                    console.print(f"       [yellow]Verification unavailable: {exc}[/yellow]")
                    break
                if passed:
                    break
                if v_round < 3:
                    console.print("       [yellow]Fixing functional bugs...[/yellow]")
                    result = fix_from_verification(
                        module_spec, result.verilog_code, failures, client
                    )
                    interface_errors = validate_modmul_interface(
                        result.verilog_code, profile.data_width
                    )
                    if interface_errors:
                        raise PipelineError(
                            f"Verification fix broke modmul interface: "
                            f"{'; '.join(interface_errors)}"
                        )
                    file_path = save_verilog(result, verilog_dir)
                else:
                    verification_failed = True
            if verification_failed and not passed:
                break

        if verification_failed:
            _set_stage(summary, "verification", "failed", "Functional verification failed")
            raise PipelineError("Functional verification failed after 3 rounds")
        _set_stage(
            summary,
            "verification",
            "unverified" if verification_unverified else ("skipped" if not use_verify else "passed"),
            "Icarus unavailable" if verification_unverified else ("--no-verify" if not use_verify else ""),
        )

        if use_synth:
            modmul_path = generated[0][1]
            console.print(Panel.fit("[bold]Stage 3: Vivado Synthesis[/bold]", style="blue"))
            if not vivado_available():
                _set_stage(summary, "synthesis_modmul", "unavailable", "Vivado not found")
            else:
                modmul_stats = run_synthesis(
                    [str(modmul_path)], output_dir, synth_part or DEFAULT_PART
                )
                if modmul_stats is None:
                    _set_stage(summary, "synthesis_modmul", "failed")
                    raise PipelineError("Vivado synthesis failed for modmul")
                _set_stage(summary, "synthesis_modmul", "passed", str(output_dir / "synthesis" / "modmul"))
        else:
            _set_stage(
                summary,
                "synthesis_modmul",
                "skipped",
                "--no-synth",
            )

        statuses = [stage["status"] for stage in summary["stages"].values()]
        summary["status"] = "unverified" if "unverified" in statuses or "unavailable" in statuses else "passed"
        _update_timing(summary, pipeline_started, client)
        _write_summary(output_dir, summary)
        if summary["status"] == "passed":
            console.print(f"\n[bold green]Done![/bold green] All outputs in: {output_dir}")
        else:
            console.print(
                f"\n[bold yellow]Completed with unverified stages.[/bold yellow] "
                f"Outputs in: {output_dir}"
            )
        return output_dir
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = str(exc)
        if output_dir is not None:
            _update_timing(summary, pipeline_started, client)
            _write_summary(output_dir, summary)
        raise


def expand_pdf_paths(pdf_args: list[str]) -> list[str]:
    """Expand directory arguments into sorted PDF files."""
    expanded: list[str] = []
    for arg in pdf_args:
        path = Path(arg)
        if path.is_dir():
            pdfs = sorted(str(file) for file in path.glob("*.pdf"))
            if not pdfs:
                console.print(f"[yellow]Warning: no PDFs found in directory: {arg}[/yellow]")
            expanded.extend(pdfs)
        else:
            expanded.append(arg)
    return expanded


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper2Gate — Generate modular-arithmetic Verilog from academic papers"
    )
    parser.add_argument(
        "pdf",
        nargs="+",
        help="PDF files or directories of PDFs to process sequentially",
    )
    parser.add_argument("-o", "--output", default=None, help="Output directory (default: config.yaml output.base_dir)")
    parser.add_argument("--dry-run", action="store_true", help="Only extract the modmul specification")
    parser.add_argument("--no-review", action="store_true", help="Skip Icarus syntax review and auto-fix")
    parser.add_argument("--no-verify", action="store_true", help="Skip golden-model functional verification")
    parser.add_argument("--no-synth", action="store_true", help="Skip Vivado synthesis")
    parser.add_argument(
        "--scheme",
        choices=["auto", "kyber", "dilithium"],
        default="auto",
        help="PQC scheme profile (default: auto-detect)",
    )
    parser.add_argument(
        "--part",
        default=None,
        help=f"Vivado part (default: config.yaml, extracted device, or {DEFAULT_PART})",
    )
    args = parser.parse_args()

    pdfs = expand_pdf_paths(args.pdf)
    if not pdfs:
        console.print("[red]Error: no PDFs to process.[/red]")
        sys.exit(1)
    missing = [path for path in pdfs if not os.path.exists(path)]
    if missing:
        console.print("[red]Error: PDF not found:[/red] " + ", ".join(missing))
        sys.exit(1)

    results: list[tuple[str, str | None, str, str | None, float | None]] = []
    for i, pdf in enumerate(pdfs, 1):
        console.print(Panel.fit(f"[bold]Paper {i}/{len(pdfs)}: {pdf}[/bold]", style="cyan"))
        try:
            output_dir = run_pipeline(
                pdf_path=pdf,
                output_base=args.output,
                dry_run=args.dry_run,
                use_review=not args.no_review,
                use_verify=not args.no_verify,
                use_synth=not args.no_synth,
                scheme=args.scheme,
                part=args.part,
            )
            summary = json.loads((output_dir / "run_summary.json").read_text(encoding="utf-8"))
            llm_seconds = summary.get("timing", {}).get("llm_seconds")
            if not isinstance(llm_seconds, (int, float)):
                llm_seconds = None
            results.append((pdf, str(output_dir), summary.get("status", "passed"), None, llm_seconds))
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user[/yellow]")
            sys.exit(130)
        except Exception as exc:
            console.print(f"[red]Error: {exc}[/red]")
            results.append((pdf, None, "failed", str(exc), None))

    table = Table(title="Batch Results", show_header=True)
    table.add_column("#", justify="right")
    table.add_column("Paper", overflow="fold")
    table.add_column("Status")
    table.add_column("LLM Time")
    table.add_column("Outputs")
    for i, (paper, output_dir, status, error, llm_seconds) in enumerate(results, 1):
        label = {
            "passed": "[green]PASSED[/green]",
            "unverified": "[yellow]UNVERIFIED[/yellow]",
            "failed": "[red]FAILED[/red]",
        }.get(status, status.upper())
        llm_label = f"{llm_seconds:.3f}s" if llm_seconds is not None else "n/a"
        table.add_row(str(i), paper, label, llm_label, output_dir or error or "")
    console.print(table)

    failed = sum(status == "failed" for _, _, status, _, _ in results)
    unverified = sum(status == "unverified" for _, _, status, _, _ in results)
    passed = len(results) - failed - unverified
    console.print(
        f"[bold green]{passed} passed[/bold green], "
        f"[bold yellow]{unverified} unverified[/bold yellow], "
        f"[bold red]{failed} failed[/bold red]"
    )
    sys.exit(1 if failed else (2 if unverified else 0))


if __name__ == "__main__":
    main()
