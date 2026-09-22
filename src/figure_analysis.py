"""Two-stage figure selection and multimodal analysis for hardware papers."""

import json
import re
from pathlib import Path
from typing import Callable

from src.llm_client import LLMClient
from src.pdf_parser import render_page
from src.prompt_manager import load_prompt


MAX_SELECTION_CHARS = 20000


def _figure_chunks(figures: list[dict]) -> list[list[dict]]:
    chunks: list[list[dict]] = []
    current: list[dict] = []
    current_size = 0
    for figure in figures:
        serialized_size = len(json.dumps(figure, ensure_ascii=False)) + 2
        if current and current_size + serialized_size > MAX_SELECTION_CHARS:
            chunks.append(current)
            current = []
            current_size = 0
        current.append(figure)
        current_size += serialized_size
    if current:
        chunks.append(current)
    return chunks


def _parse_json_response(raw: str) -> object:
    return json.loads(LLMClient._extract_json(raw))


def classify_figure_captions(
    figures: list[dict],
    client: LLMClient,
    progress: Callable[[str], None] | None = None,
) -> list[dict]:
    """Use the text model to classify captions before rendering any pages."""
    if not figures:
        return []

    by_ref = {
        _figure_key(figure): {
            **figure,
            "relevant": False,
            "category": "unknown",
            "confidence": 0.0,
            "reason": "Not selected by the figure classifier",
        }
        for figure in figures
    }
    chunks = _figure_chunks(figures)
    for chunk_index, chunk in enumerate(chunks, 1):
        if progress:
            progress(f"Classifying figure captions with text model (batch {chunk_index}/{len(chunks)})...")
        raw = client.generate_structured(
            system_prompt=(
                "You are a hardware-paper figure triage analyst. "
                "Select only figures strongly related to modular multiplication "
                "or modular reduction datapaths. Exclude NTT, memory, permutation, "
                "and unrelated control diagrams. Return a JSON array only."
            ),
            user_prompt=load_prompt(
                "classify_figures.jinja",
                figures_json=json.dumps(chunk, ensure_ascii=False, indent=2),
            ),
            stage="extract",
            operation="classify_figures",
            max_retries=2,
        )
        parsed = _parse_json_response(raw)
        if isinstance(parsed, dict):
            parsed = parsed.get("figures", parsed.get("candidates", []))
        if not isinstance(parsed, list):
            raise ValueError("Figure classifier response must be a JSON array")
        for item in parsed:
            if not isinstance(item, dict):
                continue
            figure_id = item.get("figure_id")
            if not isinstance(figure_id, str):
                continue
            item_page = item.get("pdf_page", item.get("page"))
            matching_keys = [
                key for key, figure in by_ref.items()
                if figure.get("figure_id") == figure_id
                and (item_page is None or _figure_page(figure) == _safe_page(item_page))
            ]
            if item_page is None and len(matching_keys) > 1:
                continue
            for key in matching_keys:
                result = by_ref[key]
                result["relevant"] = bool(item.get("relevant", False))
                result["category"] = str(item.get("category") or "unknown")
                confidence = item.get("confidence", 0.0)
                result["confidence"] = confidence if isinstance(confidence, (int, float)) else 0.0
                result["reason"] = str(item.get("reason") or "")
    return list(by_ref.values())


def analyze_figure_pages(
    pdf_path: str,
    figures: list[dict],
    candidates: list[dict],
    output_dir: Path,
    vision_client: LLMClient,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Render selected pages once each and collect vision evidence/errors."""
    by_page: dict[int, list[dict]] = {}
    for candidate in candidates:
        if not candidate.get("relevant"):
            continue
        reference = _find_reference(figures, candidate)
        if reference is None:
            continue
        by_page.setdefault(_figure_page(reference), []).append(reference)

    evidence: list[dict] = []
    errors: list[dict] = []
    for page, page_figures in sorted(by_page.items()):
        figure_names = [figure["figure_id"] for figure in page_figures]
        safe_name = "_".join(re.sub(r"[^A-Za-z0-9_.-]+", "_", name) for name in figure_names)
        image_path = output_dir / "figures" / f"{safe_name}_page_{page}.png"
        try:
            if progress:
                progress(
                    f"Rendering PDF page {page} for {', '.join(figure_names)}..."
                )
            render_page(pdf_path, page, image_path)
            if progress:
                progress(f"Analyzing PDF page {page} with vision model...")
            user_text = load_prompt(
                "analyze_figure.jinja",
                page=page,
                figures_json=json.dumps(page_figures, ensure_ascii=False, indent=2),
            )
            raw = vision_client.generate_multimodal_structured(
                system_prompt=(
                    "You are an expert RTL and modular-arithmetic hardware analyst. "
                    "Analyze only modular multiplication or modular reduction shown "
                    "in the page image. Ignore NTT, memory, permutation, and unrelated "
                    "control structures. Return JSON only and never infer details that "
                    "are not visible or stated in the supplied caption/context."
                ),
                user_text=user_text,
                image_path=image_path,
                operation="analyze_figure_page",
            )
            parsed = _parse_json_response(raw)
            if not isinstance(parsed, dict):
                raise ValueError("Figure vision response must be a JSON object")
            parsed.setdefault("page", page)
            parsed.setdefault("pdf_page", page)
            parsed.setdefault("figures", figure_names)
            evidence.append(parsed)
        except Exception as exc:
            errors.append(
                {
                    "page": page,
                    "figures": figure_names,
                    "error": str(exc),
                }
            )
    return evidence, errors


def _figure_page(figure: dict) -> int:
    return _safe_page(figure.get("pdf_page", figure.get("page")))


def _safe_page(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _figure_key(figure: dict) -> tuple[str, int]:
    return str(figure.get("figure_id", "")), _figure_page(figure)


def _find_reference(figures: list[dict], candidate: dict) -> dict | None:
    figure_id = candidate.get("figure_id")
    candidate_page = candidate.get("pdf_page", candidate.get("page"))
    matches = [
        figure for figure in figures
        if figure.get("figure_id") == figure_id
        and (candidate_page is None or _figure_page(figure) == _safe_page(candidate_page))
    ]
    return matches[0] if len(matches) == 1 else None
