"""PDF text extraction, figure-caption discovery, and page rendering."""

import re
from pathlib import Path

import fitz  # pymupdf


# Prefer punctuation after the figure number.  Unpunctuated captions are
# accepted only when their text looks like a caption, avoiding prose
# references ("Figure 2 would...") and table rows ("Figure 4 49 32 1").
_FIGURE_START_RE = re.compile(
    r"^\s*((?:figure|fig\.)\s*\d+[a-z]?)\s*([:.-])?\s*(.*)$",
    re.IGNORECASE,
)


def extract_page_texts(pdf_path: str) -> list[str]:
    """Extract one cleaned text string per PDF page, preserving page order."""
    with fitz.open(pdf_path) as doc:
        return [page.get_text("text").strip() for page in doc]


def extract_text(pdf_path: str) -> str:
    """
    Extract all text from a PDF file, preserving section structure.
    Returns clean text with sections separated by double newlines.
    """
    pages = extract_page_texts(pdf_path)
    pages = [text for text in pages if text]
    return "\n\n".join(pages)


def extract_figure_captions(pdf_path: str) -> list[dict]:
    """Return figure captions with their 1-based page numbers and context."""
    figures: list[dict] = []
    with fitz.open(pdf_path) as doc:
        for page_number, page in enumerate(doc, 1):
            page_text = page.get_text("text").strip()
            if not page_text:
                continue
            blocks = sorted(page.get_text("blocks"), key=lambda block: (block[1], block[0]))
            for block in blocks:
                block_text = str(block[4]).strip()
                lines = block_text.splitlines()
                starts = [
                    index for index, line in enumerate(lines)
                    if _FIGURE_START_RE.match(line)
                ]
                for start_index, line_index in enumerate(starts):
                    match = _FIGURE_START_RE.match(lines[line_index])
                    if match is None:
                        continue
                    next_line = starts[start_index + 1] if start_index + 1 < len(starts) else len(lines)
                    figure_label = re.sub(r"\s+", "", match.group(1).lower())
                    figure_number = re.sub(r"^(?:figure|fig\.)", "", figure_label)
                    figure_id = f"fig_{figure_number}"
                    caption = " ".join(
                        [match.group(3).strip(), *(line.strip() for line in lines[line_index + 1:next_line])]
                    ).strip()
                    if not _looks_like_caption(caption, bool(match.group(2))):
                        continue
                    figures.append(
                        {
                            "figure_id": figure_id,
                            "pdf_page": page_number,
                            "page": page_number,
                            "caption": caption,
                            "context": page_text[:4000],
                            "source": "caption",
                        }
                    )
    # Keep one source record for a figure on a page.  The page number is part
    # of the key because a malformed PDF can repeat a figure label elsewhere.
    unique: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for figure in figures:
        key = (figure["figure_id"], int(figure["pdf_page"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(figure)
    return unique


def _looks_like_caption(caption: str, has_delimiter: bool) -> bool:
    """Reject labels whose following content is only table-like numbers."""
    if not caption:
        return False
    words = re.findall(r"[A-Za-z][A-Za-z0-9]*", caption)
    if not words:
        return False
    if has_delimiter:
        return True
    # Without punctuation, a caption normally starts with a title-like word;
    # multiple numeric fields indicate a table row rather than a caption.
    return caption[0].isupper() and len(re.findall(r"\d+", caption)) <= 1


def render_page(pdf_path: str, page_number: int, output_path: Path, scale: float = 2.0) -> Path:
    """Render a 1-based PDF page as a PNG for a multimodal request."""
    if page_number < 1:
        raise ValueError(f"PDF page numbers are 1-based, got {page_number}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with fitz.open(pdf_path) as doc:
        if page_number > len(doc):
            raise ValueError(f"PDF has {len(doc)} pages, cannot render page {page_number}")
        pixmap = doc[page_number - 1].get_pixmap(
            matrix=fitz.Matrix(scale, scale),
            alpha=False,
        )
        pixmap.save(str(output_path))
    return output_path
