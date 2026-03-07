"""
utils/pdf_parser.py — Extract clean text from PDF using PyMuPDF
"""
from __future__ import annotations


def extract_pdf_text(pdf_path: str, max_chars: int = 60_000) -> str:
    """
    Extract and clean text from a PDF.
    Truncates at max_chars to stay within LLM context.
    Returns a single string of clean text.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF required: pip install pymupdf")

    doc = fitz.open(pdf_path)
    pages = []

    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        # Clean up common PDF artefacts
        text = text.replace("\x0c", "\n")   # form feeds
        text = "\n".join(
            line.strip() for line in text.splitlines()
            if line.strip() and len(line.strip()) > 2
        )
        pages.append(f"[Page {page_num + 1}]\n{text}")

    full_text = "\n\n".join(pages)

    if len(full_text) > max_chars:
        full_text = full_text[:max_chars] + "\n\n[... truncated for context window ...]"

    doc.close()
    return full_text
