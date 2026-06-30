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
