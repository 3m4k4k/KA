"""
Image extraction (Telegram screenshots, scanned single pages, etc).

If pytesseract + a tesseract binary are available, use them as a free
first pass. Tesseract is often good enough for clean screenshots of
digital text and bad at handwriting, low-res photos, or messy scans --
text_quality.assess() decides which bucket a given result falls into,
same as the PDF path. Anything it doesn't trust gets marked
PENDING_VISION with the original image path so a stronger vision model
can be pointed at it later, instead of silently accepting bad OCR.

If pytesseract isn't installed/configured, everything just goes
straight to PENDING_VISION -- that's a valid, safe default, not an
error.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from schemas import ExtractionMethod, PageExtraction  # noqa: E402
import text_quality  # noqa: E402

try:
    import pytesseract
    from PIL import Image

    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False


def extract_image(file_path: Path) -> list[PageExtraction]:
    if not _OCR_AVAILABLE:
        return [
            PageExtraction(
                page=1,
                text=None,
                extraction_method=ExtractionMethod.PENDING_VISION,
                confidence=0.0,
                is_garbled=True,
                garbled_reason="ocr_not_configured",
                char_count=0,
                rendered_image_path=str(file_path),
            )
        ]

    try:
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)
    except Exception as exc:  # noqa: BLE001 -- OCR failures are routing
        # signals, not crashes; any failure here should fall through to
        # a human/vision review rather than aborting the whole file.
        return [
            PageExtraction(
                page=1,
                text=None,
                extraction_method=ExtractionMethod.PENDING_VISION,
                confidence=0.0,
                is_garbled=True,
                garbled_reason=f"ocr_error:{exc}",
                char_count=0,
                rendered_image_path=str(file_path),
            )
        ]

    quality = text_quality.assess(text)

    if quality.is_garbled:
        return [
            PageExtraction(
                page=1,
                text=text if text else None,
                extraction_method=ExtractionMethod.PENDING_VISION,
                confidence=0.0,
                is_garbled=True,
                garbled_reason=quality.reason,
                char_count=quality.char_count,
                rendered_image_path=str(file_path),
            )
        ]

    return [
        PageExtraction(
            page=1,
            text=text,
            extraction_method=ExtractionMethod.IMAGE_OCR,
            confidence=quality.confidence,
            is_garbled=False,
            char_count=quality.char_count,
        )
    ]
