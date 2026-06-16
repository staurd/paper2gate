"""PDF text extraction using PyMuPDF."""

import fitz  # pymupdf


def extract_text(pdf_path: str) -> str:
    """
    Extract all text from a PDF file, preserving section structure.
    Returns clean text with sections separated by double newlines.
    """
    with fitz.open(pdf_path) as doc:
        pages: list[str] = []
        for page in doc:
            text = page.get_text("text")
            if text.strip():
                pages.append(text.strip())
    return "\n\n".join(pages)


def extract_text_with_font_info(pdf_path: str) -> list[dict]:
    """
    Extract text blocks with font size information.
    Useful for identifying section headers (larger/bold fonts).

    NOTE: Currently unused — reserved for future section-aware paper
    parsing where section headers need to be distinguished from body text.
    """
    with fitz.open(pdf_path) as doc:
        blocks_output: list[dict] = []
        for page_num, page in enumerate(doc, 1):
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if block["type"] != 0:  # skip images
                    continue
                for line in block["lines"]:
                    spans = line["spans"]
                    if not spans:
                        continue
                    text = "".join(s["text"] for s in spans)
                    font_sizes = [s["size"] for s in spans]
                    avg_size = sum(font_sizes) / len(font_sizes)
                    blocks_output.append({
                        "page": page_num,
                        "text": text.strip(),
                        "font_size": round(avg_size, 1),
                    })
    return blocks_output
