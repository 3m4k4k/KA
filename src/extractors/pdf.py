"""
PDF extraction.

Strategy per page:
    1. Try the PDF's embedded text layer (cheap, no API calls).
    2. Run it through text_quality.assess().
    3. If it's usable -> done, extraction_method = pdf_text.
    4. If not -> render the page to a PNG and mark it PENDING_VISION.
       No OCR/vision call happens here. This module's job stops at
       "here is a clean image of the page, saved to disk, ready for
       whichever OCR/vision step you plug in next." Keeping that
       boundary means you can swap Tesseract for a vision model, or a
       cheap model for an expensive one, without touching this file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pymupdf as fitz  # PyMuPDF -- `fitz` is the deprecated import alias

sys.path.append(str(Path(__file__).resolve().parent.parent))
from schemas import ExtractionMethod, PageExtraction  # noqa: E402
import text_quality  # noqa: E402


def extract_pdf(
    file_path: Path,
    *,
    render_dir: Path,
    render_dpi: int = 200,
) -> list[PageExtraction]:
    pages: list[PageExtraction] = []
    doc = fitz.open(file_path)

    try:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text")
            quality = text_quality.assess(text)

            if not quality.is_garbled:
                pages.append(
                    PageExtraction(
                        page=i,
                        text=text,
                        extraction_method=ExtractionMethod.PDF_TEXT,
                        confidence=quality.confidence,
                        is_garbled=False,
                        char_count=quality.char_count,
                    )
                )
                continue

            # Text layer unusable -- render the page as an image and
            # hand off to vision/OCR (not performed here).
            render_dir.mkdir(parents=True, exist_ok=True)
            image_path = render_dir / f"{file_path.stem}_p{i:04d}.png"
            zoom = render_dpi / 72
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            pix.save(str(image_path))

            pages.append(
                PageExtraction(
                    page=i,
                    text=text if text else None,
                    extraction_method=ExtractionMethod.PENDING_VISION,
                    confidence=0.0,
                    is_garbled=True,
                    garbled_reason=quality.reason,
                    char_count=quality.char_count,
                    rendered_image_path=str(image_path),
                )
            )
    finally:
        doc.close()

    return pages
