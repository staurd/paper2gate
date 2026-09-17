"""Stage 1: Extract innovations from paper text and produce hardware specs."""

import json
import re
from pathlib import Path

from src.config import get_llm_config
from src.console import console
from src.ir_models import HardwareSpec, ModuleSpec, PaperAnalysis, PortSpec
from src.llm_client import LLMClient
from src.prompt_manager import load_prompt
from src.schemes import KYBER, SchemeProfile, render_vars

# Generous enough that no ordinary paper is cut. What matters is not this
# number but what precedes it: references are trimmed first (below), and the
# facts Stage 1 asks for — FPGA device, latency, DSP usage, correction factor —
# live in the experimental-setup section near the END of the paper, so a tight
# budget silently starves the model of exactly what it was asked for.
DEFAULT_MAX_CHARS = 100000

_REFERENCES_RE = re.compile(r"^\s*(?:\d+\.?\s*)?(references|bibliography)\s*$", re.M | re.I)


def _resolve_max_chars(explicit: int | None) -> int:
    """Character budget for the paper text sent to the extractor."""
    if explicit is not None:
        return explicit
    try:
        return int(get_llm_config().get("max_paper_chars", DEFAULT_MAX_CHARS))
    except (TypeError, ValueError):
        return DEFAULT_MAX_CHARS


def _trim_references(paper_text: str) -> tuple[str, int]:
    """Drop the trailing references/bibliography section.

    References run 12-21% of a paper's extracted text and carry no hardware
    specs, so they are pure cost in the extraction prompt — trimming them buys
    that budget back for the experimental section, where the device, latency
    and area figures actually are.

    Only trims when the heading sits in the last 35% of the text AND has a
    substantial tail after it, so a mid-document "References" (survey papers
    cite per-section) or a stray match in the body cannot eat real content.

    Returns (text, dropped_chars); dropped_chars == 0 means nothing was cut.
    """
    matches = list(_REFERENCES_RE.finditer(paper_text))
    if not matches:
        return paper_text, 0
    last = matches[-1]
    dropped = len(paper_text) - last.start()
    if last.start() < len(paper_text) * 0.65 or dropped < 1000:
        return paper_text, 0
    return paper_text[:last.start()], dropped


def extract_innovations(
    paper_text: str,
    client: LLMClient,
    max_chars: int | None = None,
    max_retries: int = 2,
    target_interface: str | None = None,
    scheme: SchemeProfile | None = None,
) -> PaperAnalysis:
    """
    Extract innovative hardware modules from paper text.

    Trims the references section, then truncates to max_chars (default:
    DEFAULT_MAX_CHARS, overridable via config.yaml llm.max_paper_chars) to fit
    the LLM context window — and says so in the prompt when it does truncate.
    Retries on JSON parse failure with a stricter prompt.

    Args:
        target_interface: If "modred", add target interface constraint
                          so extracted ports match the reference modred.v.
        scheme: PQC scheme profile. Renders the fixed-interface constants
                (operand widths, modulus) into the prompt. Defaults to KYBER
                for backward compatibility with direct callers.
    """
    profile = scheme or KYBER
    limit = _resolve_max_chars(max_chars)

    body, dropped_refs = _trim_references(paper_text)
    if dropped_refs:
        console.print(
            f"  Trimmed {dropped_refs:,} chars of references "
            f"({len(paper_text):,} -> {len(body):,})"
        )

    truncated = body[:limit]
    truncation_note = ""
    if len(body) > limit:
        truncation_note = (
            f"NOTE: The text above is an EXCERPT — the paper continues for "
            f"another {len(body) - limit:,} characters (experimental setup, "
            f"results, comparison tables) which are NOT shown. For anything "
            f"not visible here — the FPGA device, cycle counts, area figures — "
            f"report it as unknown (empty string / 0) instead of guessing."
        )
        console.print(
            f"  [yellow]Paper text truncated to {limit:,} of {len(body):,} "
            f"chars — later sections (often the experimental setup) are not "
            f"sent to the model. Raise llm.max_paper_chars in config.yaml "
            f"to include them.[/yellow]"
        )

    user_prompt = load_prompt(
        "extract_innovation.jinja",
        paper_text=truncated,
        truncation_note=truncation_note,
        target_interface=target_interface,
        **render_vars(profile),
    )

    system_prompt = (
        "You are an expert hardware design analyst. "
        "You extract novel hardware innovations from academic papers "
        "and produce detailed hardware module specifications. "
        "Respond ONLY with valid JSON, no other text."
    )

    last_error = None
    for attempt in range(1 + max_retries):
        retry_hint = ""
        if attempt > 0:
            retry_hint = (
                f"\n\nYour previous response was not valid JSON. Error: {last_error}. "
                "Make sure the response is a single, complete JSON object. "
                "All strings must be properly escaped. No trailing commas."
            )

        raw = client.generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt + retry_hint,
            stage="extract",
        )

        try:
            data = json.loads(raw)
            return _parse_analysis(data)
        except json.JSONDecodeError as e:
            last_error = str(e)
            continue

    raise ValueError(f"Failed to parse LLM JSON response after {max_retries + 1} attempts: {last_error}")


def _parse_analysis(data: dict) -> PaperAnalysis:
    """Parse raw dict into validated Pydantic models."""
    innovations = []
    for item in data.get("innovations", []):
        hw = item.get("hardware_spec", {})

        ports: dict[str, list[PortSpec]] = {}
        for direction in ("input", "output"):
            port_list = hw.get("ports", {}).get(direction, [])
            ports[direction] = [
                PortSpec(
                    name=p["name"],
                    width=str(p.get("width", "1")),
                    direction=direction,
                    desc=p.get("desc", ""),
                )
                for p in port_list
            ]

        try:
            correction_factor = int(hw.get("correction_factor", 1))
        except (TypeError, ValueError):
            correction_factor = 1

        try:
            latency_cycles = int(hw.get("latency_cycles", 0))
        except (TypeError, ValueError):
            latency_cycles = 0

        hardware_spec = HardwareSpec(
            parameters=hw.get("parameters", {}),
            ports=ports,
            behavior=hw.get("behavior", ""),
            timing=hw.get("timing", ""),
            constraints=hw.get("constraints", ""),
            correction_factor=correction_factor,
            latency_cycles=latency_cycles,
        )

        innovations.append(ModuleSpec(
            module_name=item["module_name"],
            category=item.get("category", "general"),
            summary=item.get("summary", ""),
            hardware_spec=hardware_spec,
        ))

    return PaperAnalysis(
        paper_title=data.get("paper_title", "Unknown"),
        innovations=innovations,
        # str() guards against a non-string (e.g. a list) from the LLM.
        fpga_device=str(data.get("fpga_device") or "").strip(),
    )


def save_specs(analysis: PaperAnalysis, output_dir: Path) -> Path:
    """Save hardware specs as JSON to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "hardware_specs.json"
    out_path.write_text(
        analysis.model_dump_json(indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out_path
