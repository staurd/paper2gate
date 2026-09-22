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
SCHEME_CONTEXT_CHARS = 2000

_REFERENCES_RE = re.compile(r"^\s*(?:\d+\.?\s*)?(references|bibliography)\s*$", re.M | re.I)


def _resolve_max_chars(explicit: int | None) -> int:
    """Character budget for the paper text sent to the extractor."""
    if explicit is not None:
        return explicit
    try:
        return int(get_llm_config().get("max_paper_chars", DEFAULT_MAX_CHARS))
    except (TypeError, ValueError):
        return DEFAULT_MAX_CHARS


def trim_references(paper_text: str) -> tuple[str, int]:
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


def classify_scheme(paper_text: str, client: LLMClient) -> tuple[str, str]:
    """Classify the paper's target modular arithmetic from its opening text."""
    raw = client.generate_structured(
        system_prompt=(
            "Identify the target PQC scheme from the supplied paper excerpt. "
            "Respond ONLY with valid JSON."
        ),
        user_prompt=load_prompt("classify_scheme.jinja", paper_text=paper_text[:SCHEME_CONTEXT_CHARS]),
        stage="extract",
        operation="classify_scheme",
    )
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid scheme classification JSON: {exc}") from exc
    if not isinstance(result, dict):
        raise ValueError("Scheme classification must be a JSON object")
    scheme = result.get("scheme")
    evidence = result.get("evidence")
    if scheme not in ("kyber", "dilithium", "unknown", "mixed"):
        raise ValueError(f"Invalid classified scheme: {scheme!r}")
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError("Scheme classification needs a non-empty evidence string")
    return scheme, evidence.strip()


def extract_innovations(
    paper_text: str,
    client: LLMClient,
    max_chars: int | None = None,
    max_retries: int = 2,
    scheme: SchemeProfile | None = None,
    visual_evidence: list[dict] | None = None,
) -> PaperAnalysis:
    """
    Extract innovative hardware modules from paper text.

    Trims the references section, then truncates to max_chars (default:
    DEFAULT_MAX_CHARS, overridable via config.yaml llm.max_paper_chars) to fit
    the LLM context window — and says so in the prompt when it does truncate.
    Retries on JSON parse failure with a stricter prompt.

    Args:
        scheme: PQC scheme profile. Renders the fixed-interface constants
                (operand widths, modulus) into the prompt. Defaults to KYBER
                for backward compatibility with direct callers.
    """
    profile = scheme or KYBER
    limit = _resolve_max_chars(max_chars)

    body, dropped_refs = trim_references(paper_text)
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
        visual_evidence=(
            json.dumps(visual_evidence, ensure_ascii=False, indent=2)
            if visual_evidence
            else ""
        ),
        **render_vars(profile),
    )

    system_prompt = (
        "You are an expert hardware design analyst. "
        "You extract novel hardware innovations from academic papers "
        "and produce detailed hardware module specifications. "
        "Extract only designs proposed or explicitly modified by the paper's authors; "
        "treat cited prior work, baselines, comparisons, and reproduced designs as context, "
        "not as the paper's innovation. "
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
            operation="extract_innovations",
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
    if not isinstance(data, dict):
        raise ValueError("LLM response must be a JSON object")
    raw_innovations = data.get("innovations", [])
    if not isinstance(raw_innovations, list):
        raise ValueError("'innovations' must be a JSON array")

    innovations = []
    for index, item in enumerate(raw_innovations):
        if not isinstance(item, dict):
            raise ValueError(f"innovation[{index}] must be a JSON object")
        if not isinstance(item.get("module_name"), str) or not item["module_name"].strip():
            raise ValueError(f"innovation[{index}] is missing a module_name")
        hw = item.get("hardware_spec", {})
        if not isinstance(hw, dict):
            raise ValueError(f"innovation[{index}].hardware_spec must be an object")

        ports: dict[str, list[PortSpec]] = {}
        raw_ports = hw.get("ports", {})
        if raw_ports is None:
            raw_ports = {}
        if not isinstance(raw_ports, dict):
            raise ValueError(f"innovation[{index}].hardware_spec.ports must be an object")
        for direction in ("input", "output"):
            port_list = raw_ports.get(direction, [])
            if not isinstance(port_list, list):
                raise ValueError(
                    f"innovation[{index}].hardware_spec.ports.{direction} must be an array"
                )
            normalized_ports = []
            for port_index, port in enumerate(port_list):
                if not isinstance(port, dict) or not isinstance(port.get("name"), str):
                    raise ValueError(
                        f"innovation[{index}] {direction} port[{port_index}] needs a name"
                    )
                normalized_ports.append(port)
            ports[direction] = [
                PortSpec(
                    name=p["name"],
                    width=str(p.get("width", "1")),
                    direction=direction,
                    desc=p.get("desc", ""),
                )
                for p in normalized_ports
            ]

        correction_factor = _parse_correction_factor(
            hw,
            summary=item.get("summary", ""),
        )

        try:
            latency_cycles = int(hw.get("latency_cycles", 0))
        except (TypeError, ValueError):
            latency_cycles = 0

        raw_parameters = hw.get("parameters", {})
        if raw_parameters is None:
            raw_parameters = {}
        if not isinstance(raw_parameters, dict):
            raise ValueError(f"innovation[{index}].hardware_spec.parameters must be an object")

        hardware_spec = HardwareSpec(
            parameters={str(k): str(v) for k, v in raw_parameters.items()},
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


def _parse_correction_factor(hardware_spec: dict, summary: str = "") -> int:
    """Parse a signed factor from the module's *output* semantics.

    Reduction internals often contain negative constant multiplications (for
    example ``-13 * Cl``) even when the final result is ``+13*A*B mod q``.
    Only flip a positive factor when a negative expression is explicitly tied
    to the output/result, so those internal steps cannot corrupt verification.
    """
    try:
        factor = int(hardware_spec.get("correction_factor", 1))
    except (TypeError, ValueError):
        return 1
    if factor <= 0:
        return factor

    # Models sometimes return the magnitude (13) even when the surrounding
    # algorithm text explicitly says ``C' = -k*C`` or ``R = -13*A*B``.
    # Normalize common PDF punctuation before checking the output expression.
    output_desc = " ".join(
        str(port.get("desc", ""))
        for port in (hardware_spec.get("ports", {}) or {}).get("output", [])
        if isinstance(port, dict)
    )
    evidence = " ".join(
        [summary, output_desc]
        + [str(hardware_spec.get(field, "")) for field in ("behavior", "timing", "constraints")]
    )
    evidence = (
        evidence.replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("′", "'")
    )
    factor_token = rf"(?:{re.escape(str(factor))}(?!\d)|[kK])"
    product_tail = (
        r"(?:\s*(?:\*|x|×|·)\s*)?\(?\s*"
        r"(?:A|B|C|P_R|product|a|b|c)\b"
        r"(?:\s*(?:\*|x|×|·)\s*(?:A|B|C|P_R|product|a|b|c)\b)?"
    )

    # Direct assignment is the least ambiguous form: C' = -k*C, R = -13*A*B.
    direct_output = re.search(
        rf"(?:\bR\b|\bC\s*'?)\s*(?:=|is)\s*-\s*{factor_token}"
        rf"{product_tail}",
        evidence,
        re.IGNORECASE,
    )
    # Also accept prose such as "the output represents -13*(A*B) mod q";
    # keep the match within one sentence to avoid unrelated internal steps.
    prose_output = re.search(
        rf"\b(?:output|result(?:ing)?|reduced\s+(?:product|result|value)|module\s+output)\b"
        rf"[^.!?;\n]{{0,100}}?-\s*{factor_token}{product_tail}",
        evidence,
        re.IGNORECASE,
    )
    return -factor if direct_output or prose_output else factor


def save_specs(analysis: PaperAnalysis, output_dir: Path) -> Path:
    """Save hardware specs as JSON to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "hardware_specs.json"
    out_path.write_text(
        analysis.model_dump_json(indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out_path
