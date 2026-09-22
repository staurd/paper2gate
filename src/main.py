"""Paper2Gate CLI: extract modular-arithmetic designs and generate Verilog."""

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
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
from src.config import (
    PROJECT_ROOT,
    get_iverilog_config,
    get_output_config,
    get_vivado_config,
    get_vision_config,
)
from src.console import console
from src.figure_analysis import analyze_figure_pages, classify_figure_captions
from src.innovation_extractor import classify_scheme, extract_innovations, save_specs, trim_references
from src.ir_models import ModuleSpec, parse_module_params
from src.llm_client import LLMClient
from src.pdf_parser import extract_figure_captions, extract_page_texts, extract_text
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
    "classify_figures",
    "analyze_figure_page",
    "generate_modmul",
    "fix_syntax",
    "fix_function",
)

RESOURCE_NAMES = ("LUT", "FF", "DSP", "BRAM")


def _empty_llm_timing() -> dict:
    return {
        "llm_seconds": 0.0,
        "llm_calls": 0,
        "llm_by_operation": {
            operation: {"calls": 0, "seconds": 0.0}
            for operation in LLM_TIMING_OPERATIONS
        },
    }


def _empty_llm_metadata() -> dict:
    return {
        "provider": None,
        "model": None,
        "extract_model": None,
        "generate_model": None,
        "by_operation": {},
        "vision": {
            "enabled": False,
            "provider": None,
            "model": None,
        },
    }


def _empty_resources() -> dict[str, int | float | None]:
    return {name: None for name in RESOURCE_NAMES}


def _optional_number(value: object) -> int | float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    return None


def _optional_text(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _normalize_resources(value: object) -> dict[str, int | float | None]:
    resources = _empty_resources()
    if not isinstance(value, dict):
        return resources
    for name in RESOURCE_NAMES:
        resources[name] = _optional_number(value.get(name))
    return resources


def _normalize_llm_metadata(value: object) -> dict:
    metadata = _empty_llm_metadata()
    if not isinstance(value, dict):
        return metadata

    for name in ("provider", "model", "extract_model", "generate_model"):
        metadata[name] = _optional_text(value.get(name))
    metadata["extract_model"] = metadata["extract_model"] or metadata["model"]
    metadata["generate_model"] = metadata["generate_model"] or metadata["model"]
    vision = value.get("vision")
    if isinstance(vision, dict):
        metadata["vision"] = {
            "enabled": bool(vision.get("enabled", False)),
            "provider": _optional_text(vision.get("provider")),
            "model": _optional_text(vision.get("model")),
        }

    by_operation = value.get("by_operation")
    if isinstance(by_operation, dict):
        normalized_operations = {}
        for operation, details in by_operation.items():
            if not isinstance(operation, str) or not isinstance(details, dict):
                continue
            normalized_operations[operation] = {
                "provider": _optional_text(details.get("provider")),
                "model": _optional_text(details.get("model")),
            }
        metadata["by_operation"] = normalized_operations
    return metadata


def _update_llm_metadata(
    summary: dict,
    client: LLMClient | None,
    vision_client: LLMClient | None = None,
    vision_config: dict | None = None,
) -> None:
    metadata = _empty_llm_metadata()
    try:
        client_metadata = client.metadata_snapshot() if client is not None else None
    except (AttributeError, TypeError, ValueError):
        client_metadata = None
    metadata.update(_normalize_llm_metadata(client_metadata))
    if vision_config is not None:
        metadata["vision"] = {
            "enabled": bool(vision_config.get("enabled", False)),
            "provider": _optional_text(vision_config.get("provider")),
            "model": _optional_text(vision_config.get("model")),
        }
    if vision_client is not None:
        try:
            vision_metadata = vision_client.metadata_snapshot()
        except (AttributeError, TypeError, ValueError):
            vision_metadata = None
        if isinstance(vision_metadata, dict):
            vision_provider = _optional_text(vision_metadata.get("provider"))
            vision_model = _optional_text(vision_metadata.get("model"))
            metadata["vision"] = {
                "enabled": True,
                "provider": vision_provider,
                "model": vision_model,
            }
            by_operation = metadata.setdefault("by_operation", {})
            by_operation["analyze_figure_page"] = {
                "provider": vision_provider,
                "model": vision_model,
            }
    else:
        metadata.setdefault("by_operation", {}).pop("analyze_figure_page", None)
    summary["llm"] = _normalize_llm_metadata(metadata)


def _update_timing(
    summary: dict,
    pipeline_started: float,
    client: LLMClient | None,
    vision_client: LLMClient | None = None,
    vision_config: dict | None = None,
) -> None:
    timing = _empty_llm_timing()
    snapshots = []
    for timing_client in (client, vision_client):
        if timing_client is None:
            continue
        try:
            client_timing = timing_client.timing_snapshot()
        except (AttributeError, TypeError, ValueError):
            client_timing = None
        if isinstance(client_timing, dict):
            snapshots.append(client_timing)
    for client_timing in snapshots:
        timing["llm_seconds"] = round(
            timing["llm_seconds"] + float(client_timing.get("llm_seconds", 0.0)), 3
        )
        timing["llm_calls"] += int(client_timing.get("llm_calls", 0))
        for operation, values in client_timing.get("llm_by_operation", {}).items():
            if not isinstance(values, dict):
                continue
            target = timing["llm_by_operation"].setdefault(
                operation, {"calls": 0, "seconds": 0.0}
            )
            target["calls"] += int(values.get("calls", 0))
            target["seconds"] = round(
                target["seconds"] + float(values.get("seconds", 0.0)), 3
            )
    timing["pipeline_seconds"] = round(time.perf_counter() - pipeline_started, 3)
    summary["timing"] = timing
    _update_llm_metadata(summary, client, vision_client, vision_config)


def _print_llm_metadata(client: LLMClient) -> None:
    provider = _optional_text(getattr(client, "provider", None)) or "unknown"
    model = _optional_text(getattr(client, "model", None))
    extract_model = _optional_text(getattr(client, "extract_model", None)) or model
    generate_model = _optional_text(getattr(client, "generate_model", None)) or model
    console.print(f"  LLM provider: [cyan]{provider}[/cyan]")
    if extract_model and extract_model == generate_model:
        console.print(f"  LLM model: [cyan]{extract_model}[/cyan]")
    else:
        if extract_model:
            console.print(f"  LLM extract model: [cyan]{extract_model}[/cyan]")
        if generate_model:
            console.print(f"  LLM generate model: [cyan]{generate_model}[/cyan]")


def _print_vision_metadata(config: dict) -> None:
    if not config.get("enabled", False):
        console.print("  Vision analysis: [dim]disabled[/dim]")
        return
    provider = _optional_text(config.get("provider")) or "unknown"
    model = _optional_text(config.get("model")) or "unknown"
    console.print("  Vision analysis: [cyan]enabled[/cyan]")
    console.print(f"  Vision provider: [cyan]{provider}[/cyan]")
    console.print(f"  Vision model: [cyan]{model}[/cyan]")


def build_output_dir(pdf_path: str, base_dir: str = "outputs") -> Path:
    """Create a timestamped output directory for this run."""
    paper_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(pdf_path).stem).strip("_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    out = base_path / f"{paper_name}_{timestamp}"
    suffix = 1
    while out.exists():
        out = base_path / f"{paper_name}_{timestamp}_{suffix}"
        suffix += 1
    out.mkdir()
    return out


def build_batch_output_dir(base_dir: Path) -> Path:
    """Create a unique timestamped parent directory for a multi-paper run."""
    base_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = base_dir / f"batch_{stamp}"
    suffix = 1
    while candidate.exists():
        candidate = base_dir / f"batch_{stamp}_{suffix}"
        suffix += 1
    candidate.mkdir()
    return candidate


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


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _write_summary(output_dir: Path, summary: dict) -> Path:
    path = output_dir / "run_summary.json"
    return _write_json(path, summary)


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
    vision_client: LLMClient | None = None
    vision_config = get_vision_config()
    pipeline_started = time.perf_counter()
    summary = {
        "paper": str(Path(pdf_path).resolve()),
        "status": "running",
        "scheme": None,
        "llm": _empty_llm_metadata(),
        "timing": {"pipeline_seconds": 0.0, **_empty_llm_timing()},
        "resources": _empty_resources(),
        "tools": {
            "iverilog": get_iverilog_config().get("binary"),
            "vivado": get_vivado_config().get("binary"),
        },
        "stages": {},
    }
    summary["llm"]["vision"] = {
        "enabled": bool(vision_config.get("enabled", False)),
        "provider": _optional_text(vision_config.get("provider")),
        "model": _optional_text(vision_config.get("model")),
    }

    try:
        console.print(Panel.fit("[bold]Stage 0: Parsing PDF[/bold]", style="blue"))
        with console.status(f"Extracting text from {pdf_path}..."):
            paper_text = extract_text(pdf_path)
            # Keep page storage aligned with the parser's physical page order.
            # The fallback also keeps mocked text-only pipeline tests usable.
            page_texts = (
                extract_page_texts(pdf_path)
                if Path(pdf_path).exists()
                else [paper_text]
            )
        if not paper_text.strip():
            raise PipelineError("PDF contains no extractable text; OCR is required")
        console.print(f"  Extracted [green]{len(paper_text):,}[/green] characters")
        _set_stage(summary, "parse_pdf", "passed")

        output_dir = build_output_dir(pdf_path, str(_resolve_output_base(output_base)))
        text_dir = output_dir / "extracted_text"
        text_dir.mkdir(parents=True, exist_ok=True)
        (text_dir / "full_text.txt").write_text(paper_text, encoding="utf-8")
        pages_dir = text_dir / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)
        for page_number, page_text in enumerate(page_texts, 1):
            (pages_dir / f"page_{page_number:04d}.txt").write_text(
                page_text,
                encoding="utf-8",
            )
        console.print(
            f"  Stored [green]{len(page_texts)}[/green] page text files under: {pages_dir}"
        )

        body_text, dropped_refs = trim_references(paper_text)
        if dropped_refs:
            console.print(
                f"  Trimmed {dropped_refs:,} chars of references "
                f"({len(paper_text):,} -> {len(body_text):,})"
            )

        client = LLMClient()
        _print_llm_metadata(client)
        _print_vision_metadata(vision_config)
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

        visual_evidence: list[dict] = []
        if not vision_config.get("enabled", False):
            _set_stage(summary, "figure_selection", "skipped", "vision disabled")
            _set_stage(summary, "figure_analysis", "skipped", "vision disabled")
        else:
            console.print("  Extracting figure captions with PDF page mapping...")
            candidates_path = output_dir / "figure_candidates.json"
            evidence_path = output_dir / "vision_evidence.json"
            figure_extraction_error = None
            try:
                figure_refs = extract_figure_captions(pdf_path)
            except Exception as exc:
                figure_refs = []
                figure_extraction_error = str(exc)
                console.print(f"  [yellow]Figure caption extraction failed: {exc}[/yellow]")
            console.print(f"  Figure captions found: [green]{len(figure_refs)}[/green]")
            if not figure_refs:
                if figure_extraction_error:
                    _write_json(candidates_path, {"figures": [], "error": figure_extraction_error})
                    _write_json(
                        evidence_path,
                        {"evidence": [], "errors": [{"stage": "caption_extraction", "error": figure_extraction_error}]},
                    )
                    _set_stage(summary, "figure_selection", "unverified", figure_extraction_error)
                else:
                    _write_json(candidates_path, [])
                    _write_json(evidence_path, {"evidence": [], "errors": []})
                    _set_stage(summary, "figure_selection", "skipped", "no figure captions found")
                _set_stage(summary, "figure_analysis", "skipped", "no relevant figure captions")
            else:
                selection_failed = False
                try:
                    candidates = classify_figure_captions(
                        figure_refs,
                        client,
                        progress=lambda message: console.print(f"  {message}"),
                    )
                    _write_json(candidates_path, candidates)
                    relevant = [item for item in candidates if item.get("relevant")]
                    if relevant:
                        locations = ", ".join(
                            f"{item['figure_id']} (PDF page {item.get('pdf_page', item.get('page'))})"
                            for item in relevant
                        )
                        console.print(f"  Relevant figures: [cyan]{locations}[/cyan]")
                    else:
                        console.print("  Relevant figures: [dim]none[/dim]")
                    _set_stage(
                        summary,
                        "figure_selection",
                        "passed",
                        f"{sum(bool(item.get('relevant')) for item in candidates)} relevant figure(s)",
                    )
                except Exception as exc:
                    candidates = []
                    selection_failed = True
                    console.print(f"  [yellow]Figure caption selection failed: {exc}[/yellow]")
                    _write_json(candidates_path, {"figures": figure_refs, "error": str(exc)})
                    _write_json(
                        evidence_path,
                        {"evidence": [], "errors": [{"stage": "selection", "error": str(exc)}]},
                    )
                    _set_stage(summary, "figure_selection", "unverified", str(exc))
                    _set_stage(summary, "figure_analysis", "skipped", "figure selection failed")

                relevant_candidates = [item for item in candidates if item.get("relevant")]
                if selection_failed:
                    pass
                elif not relevant_candidates:
                    _write_json(evidence_path, {"evidence": [], "errors": []})
                    _set_stage(summary, "figure_analysis", "skipped", "no relevant figure captions")
                else:
                    try:
                        vision_client = LLMClient(config=vision_config, vision=True)
                        visual_evidence, vision_errors = analyze_figure_pages(
                            pdf_path,
                            figure_refs,
                            relevant_candidates,
                            output_dir,
                            vision_client,
                            progress=lambda message: console.print(f"  {message}"),
                        )
                        _write_json(
                            evidence_path,
                            {"evidence": visual_evidence, "errors": vision_errors},
                        )
                        console.print(
                            f"  Vision evidence collected: [green]{len(visual_evidence)}[/green] page(s)"
                        )
                        if vision_errors:
                            console.print(
                                f"  [yellow]Vision analysis errors: {len(vision_errors)} page(s); "
                                "continuing with text evidence[/yellow]"
                            )
                            _set_stage(
                                summary,
                                "figure_analysis",
                                "unverified",
                                f"{len(vision_errors)} figure page(s) failed",
                            )
                        else:
                            _set_stage(
                                summary,
                                "figure_analysis",
                                "passed",
                                f"{len(visual_evidence)} page(s) analyzed",
                            )
                    except Exception as exc:
                        _write_json(
                            evidence_path,
                            {"evidence": [], "errors": [{"stage": "analysis", "error": str(exc)}]},
                        )
                        console.print(f"  [yellow]Vision analysis unavailable: {exc}[/yellow]")
                        _set_stage(summary, "figure_analysis", "unverified", str(exc))

        console.print(Panel.fit("[bold]Stage 1: Extracting Innovations[/bold]", style="blue"))
        with console.status("Analyzing paper with LLM ..."):
            analysis = extract_innovations(
                body_text,
                client,
                scheme=profile,
                visual_evidence=visual_evidence,
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
            statuses = [stage["status"] for stage in summary["stages"].values()]
            summary["status"] = "unverified" if "unverified" in statuses else "passed"
            _update_timing(summary, pipeline_started, client, vision_client, vision_config)
            _write_summary(output_dir, summary)
            console.print("[yellow]No innovations found in the paper.[/yellow]")
            return output_dir

        if dry_run:
            _set_stage(summary, "generate", "skipped", "--dry-run")
            _set_stage(summary, "verification", "skipped", "--dry-run")
            _set_stage(summary, "synthesis_modmul", "skipped", "--dry-run")
            statuses = [stage["status"] for stage in summary["stages"].values()]
            summary["status"] = "unverified" if "unverified" in statuses else "passed"
            _update_timing(summary, pipeline_started, client, vision_client, vision_config)
            _write_summary(output_dir, summary)
            console.print("[yellow]--dry-run: skipping Verilog generation.[/yellow]")
            return output_dir

        selected_specs = _select_specs(analysis.innovations)
        target_part = resolve_synth_part(part, analysis.fpga_device)
        synth_part = target_part if use_synth else None
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
                    target_part=target_part,
                )
                save_error_logs(error_logs, module_spec.module_name, review_dir)
            else:
                result = generate_modmul(
                    module_spec,
                    client,
                    scheme=profile,
                    target_part=target_part,
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
                summary["resources"] = _normalize_resources(modmul_stats)
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
        _update_timing(summary, pipeline_started, client, vision_client, vision_config)
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
            _update_timing(summary, pipeline_started, client, vision_client, vision_config)
            _write_summary(output_dir, summary)
            # Let the batch caller retain a failed paper's output directory.
            try:
                setattr(exc, "output_dir", output_dir)
            except AttributeError:
                pass
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


@dataclass
class _BatchResult:
    paper: str
    status: str
    part: str | None = None
    llm_seconds: int | float | None = None
    total_seconds: int | float | None = None
    llm_calls: int | None = None
    llm: dict = field(default_factory=_empty_llm_metadata)
    resources: dict[str, int | float | None] = field(default_factory=_empty_resources)
    output_dir: Path | None = None
    error: str | None = None


def _batch_result_from_summary(
    pdf: str,
    summary: dict,
    output_dir: Path | None = None,
) -> _BatchResult:
    timing = summary.get("timing")
    if not isinstance(timing, dict):
        timing = {}
    status = summary.get("status", "passed")
    if not isinstance(status, str):
        status = "passed"
    paper = summary.get("paper")
    if not isinstance(paper, str) or not paper.strip():
        paper = str(Path(pdf).resolve())
    llm_calls = timing.get("llm_calls")
    if not isinstance(llm_calls, int) or isinstance(llm_calls, bool):
        llm_calls = None
    return _BatchResult(
        paper=paper,
        status=status,
        part=_optional_text(summary.get("synthesis_part")),
        llm_seconds=_optional_number(timing.get("llm_seconds")),
        total_seconds=_optional_number(timing.get("pipeline_seconds")),
        llm_calls=llm_calls,
        llm=_normalize_llm_metadata(summary.get("llm")),
        resources=_normalize_resources(summary.get("resources")),
        output_dir=output_dir,
        error=_optional_text(summary.get("error")),
    )


def _format_seconds(seconds: int | float | None) -> str:
    return f"{seconds:.3f}s" if seconds is not None else "n/a"


def _build_batch_results_table(results: list[_BatchResult]) -> Table:
    table = Table(title="Batch Results", show_header=True)
    table.add_column("#", justify="right")
    table.add_column("Paper", overflow="fold")
    table.add_column("Status")
    table.add_column("Part")
    table.add_column("LLM Time", justify="right")
    table.add_column("Total Time", justify="right")
    for name in RESOURCE_NAMES:
        table.add_column(name, justify="right")

    for i, result in enumerate(results, 1):
        label = {
            "passed": "[green]PASSED[/green]",
            "unverified": "[yellow]UNVERIFIED[/yellow]",
            "unavailable": "[yellow]UNAVAILABLE[/yellow]",
            "failed": "[red]FAILED[/red]",
        }.get(result.status, result.status.upper())
        resource_labels = [
            str(result.resources[name])
            if result.resources.get(name) is not None
            else "n/a"
            for name in RESOURCE_NAMES
        ]
        table.add_row(
            str(i),
            result.paper,
            label,
            result.part or "n/a",
            _format_seconds(result.llm_seconds),
            _format_seconds(result.total_seconds),
            *resource_labels,
        )
    return table


def _sum_batch_metric(results: list[_BatchResult], name: str) -> int | float:
    return round(
        sum(
            value
            for result in results
            if (value := getattr(result, name)) is not None
        ),
        3,
    )


def _batch_result_to_dict(result: _BatchResult, batch_dir: Path) -> dict:
    item = {
        "paper": result.paper,
        "status": result.status,
        "output_dir": None,
        "run_summary": None,
        "synthesis_part": result.part,
        "timing": {
            "llm_seconds": result.llm_seconds,
            "pipeline_seconds": result.total_seconds,
            "llm_calls": result.llm_calls,
        },
        "llm": result.llm,
        "resources": result.resources,
    }
    if result.output_dir is not None:
        item["output_dir"] = Path(
            os.path.relpath(result.output_dir, batch_dir)
        ).as_posix()
        item["run_summary"] = Path(
            os.path.relpath(result.output_dir / "run_summary.json", batch_dir)
        ).as_posix()
    if result.error:
        item["error"] = result.error
    return item


def _write_batch_summary(
    batch_dir: Path,
    started_at: str,
    finished_at: str,
    wall_seconds: float,
    results: list[_BatchResult],
) -> Path:
    status_counts = {"passed": 0, "unverified": 0, "failed": 0}
    for result in results:
        if result.status == "failed":
            status_counts["failed"] += 1
        elif result.status in ("unverified", "unavailable"):
            status_counts["unverified"] += 1
        else:
            status_counts["passed"] += 1

    if status_counts["failed"]:
        status = "failed"
    elif status_counts["unverified"]:
        status = "unverified"
    else:
        status = "passed"

    summary = {
        "batch_id": batch_dir.name,
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_seconds": round(wall_seconds, 3),
        "paper_count": len(results),
        "status_counts": status_counts,
        "totals": {
            "pipeline_seconds": _sum_batch_metric(results, "total_seconds"),
            "llm_seconds": _sum_batch_metric(results, "llm_seconds"),
            "llm_calls": int(_sum_batch_metric(results, "llm_calls")),
        },
        "papers": [_batch_result_to_dict(result, batch_dir) for result in results],
    }
    return _write_json(batch_dir / "batch_summary.json", summary)


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

    batch_dir: Path | None = None
    batch_started_at = ""
    batch_started = 0.0
    if len(pdfs) > 1:
        output_root = _resolve_output_base(args.output)
        batch_dir = build_batch_output_dir(output_root)
        batch_started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        batch_started = time.perf_counter()
        console.print(f"  Batch outputs: [cyan]{batch_dir}[/cyan]")

    results: list[_BatchResult] = []
    for i, pdf in enumerate(pdfs, 1):
        console.print(Panel.fit(f"[bold]Paper {i}/{len(pdfs)}: {pdf}[/bold]", style="cyan"))
        try:
            output_dir = run_pipeline(
                pdf_path=pdf,
                output_base=str(batch_dir) if batch_dir is not None else args.output,
                dry_run=args.dry_run,
                use_review=not args.no_review,
                use_verify=not args.no_verify,
                use_synth=not args.no_synth,
                scheme=args.scheme,
                part=args.part,
            )
            summary = json.loads((output_dir / "run_summary.json").read_text(encoding="utf-8"))
            results.append(_batch_result_from_summary(pdf, summary, output_dir))
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user[/yellow]")
            sys.exit(130)
        except Exception as exc:
            console.print(f"[red]Error: {exc}[/red]")
            failed_output_dir = getattr(exc, "output_dir", None)
            if not isinstance(failed_output_dir, Path):
                failed_output_dir = None
            failed_summary = {}
            if failed_output_dir is not None:
                summary_path = failed_output_dir / "run_summary.json"
                if summary_path.exists():
                    try:
                        failed_summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        failed_summary = {}
            results.append(
                _batch_result_from_summary(
                    pdf,
                    failed_summary or {"status": "failed", "error": str(exc)},
                    failed_output_dir,
                )
            )

    if len(pdfs) > 1:
        console.print(_build_batch_results_table(results))
        assert batch_dir is not None
        finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
        batch_summary_path = _write_batch_summary(
            batch_dir,
            batch_started_at,
            finished_at,
            time.perf_counter() - batch_started,
            results,
        )
        console.print(f"  Batch summary: [cyan]{batch_summary_path}[/cyan]")

    failed = sum(result.status == "failed" for result in results)
    unverified = sum(result.status in ("unverified", "unavailable") for result in results)
    passed = len(results) - failed - unverified
    console.print(
        f"[bold green]{passed} passed[/bold green], "
        f"[bold yellow]{unverified} unverified[/bold yellow], "
        f"[bold red]{failed} failed[/bold red]"
    )
    sys.exit(1 if failed else (2 if unverified else 0))


if __name__ == "__main__":
    main()
