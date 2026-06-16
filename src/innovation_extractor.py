"""Stage 1: Extract innovations from paper text and produce hardware specs."""

import json
from pathlib import Path

from src.ir_models import HardwareSpec, ModuleSpec, PaperAnalysis, PortSpec
from src.llm_client import LLMClient

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "extract_innovation.txt"


def _load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def extract_innovations(
    paper_text: str,
    client: LLMClient,
    max_chars: int = 50000,
    max_retries: int = 2,
) -> PaperAnalysis:
    """
    Extract innovative hardware modules from paper text.

    Truncates paper text to max_chars to fit LLM context window.
    Retries on JSON parse failure with a stricter prompt.
    """
    prompt = _load_prompt()
    truncated = paper_text[:max_chars]
    user_prompt = prompt.replace("{{paper_text}}", truncated)

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
                    width=p.get("width", "1"),
                    direction=direction,
                    desc=p.get("desc", ""),
                )
                for p in port_list
            ]

        hardware_spec = HardwareSpec(
            parameters=hw.get("parameters", {}),
            ports=ports,
            behavior=hw.get("behavior", ""),
            timing=hw.get("timing", ""),
            constraints=hw.get("constraints", ""),
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
