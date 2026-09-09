"""
Core data structures for the ingestion / reconstruction layer.

Nothing in this file makes decisions about question content, taxonomy,
or answers. It only describes: what file did this come from, how was
text obtained from it, how much do we trust that extraction, and where
can a human go look at the original if something looks wrong.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

PIPELINE_VERSION = "0.1.0"


class ExtractionMethod(str, Enum):
    PDF_TEXT = "pdf_text"           # direct text layer in the PDF
    DOCX_TEXT = "docx_text"         # python-docx paragraph/table extraction
    IMAGE_OCR = "image_ocr"         # pytesseract OCR on an image/screenshot
    PDF_PAGE_OCR = "pdf_page_ocr"   # page rendered to image, then OCR'd
    VISION_MODEL = "vision_model"   # transcribed by a Claude vision call
    PENDING_VISION = "pending_vision"  # extraction unusable, image saved for a
                                        # human or a vision model to handle later
    FAILED = "failed"               # could not extract anything at all


class FileType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    IMAGE = "image"
    UNSUPPORTED = "unsupported"


class PageExtraction(BaseModel):
    """One page (or one logical chunk, for formats without real pages)."""

    page: int
    text: Optional[str] = None
    extraction_method: ExtractionMethod
    confidence: float = Field(ge=0.0, le=1.0)

    # Quality-check output. See text_quality.py. Kept separate from
    # `confidence` so a low-confidence-but-flagged page is inspectable
    # rather than just a single opaque number.
    is_garbled: bool = False
    garbled_reason: Optional[str] = None
    char_count: int = 0

    # If the page had to be rendered as an image (bad text layer, or the
    # page IS an image), the render is kept on disk so a human or a
    # vision model can look at exactly what the pipeline saw.
    rendered_image_path: Optional[str] = None

    # Set only when extraction_method == VISION_MODEL. Provenance: which
    # model produced this transcription, so it's traceable and so a
    # later re-run with a better/cheaper model is an explicit choice,
    # not a silent overwrite.
    vision_model_used: Optional[str] = None


class SourceDocument(BaseModel):
    """The reconstructed representation of one input file.

    This is the ONLY thing later pipeline stages (question segmentation,
    parsing, classification, verification) are allowed to read from.
    They never re-touch the original file directly, which keeps the
    provenance chain intact and makes re-runs deterministic.
    """

    document_id: str
    pipeline_version: str = PIPELINE_VERSION

    # Provenance
    original_filename: str
    original_path: str
    file_type: FileType
    file_hash: str  # sha256 of the raw file bytes

    # Assigned by the human via folder structure, not by the pipeline.
    # subtopic is intentionally left as a null placeholder for now --
    # nothing in this layer or later layers should try to fill it in
    # automatically until that's explicitly built and reviewed.
    subject: Optional[str] = None
    topic: Optional[str] = None
    subtopic: Optional[str] = None

    pages: list[PageExtraction] = Field(default_factory=list)

    ingested_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # Rolled up from pages, so downstream triage doesn't need to
    # re-scan every page to decide whether a document needs attention.
    needs_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)

    def compute_rollup(self) -> None:
        reasons: list[str] = []
        if not self.pages:
            reasons.append("no_pages_extracted")
        for p in self.pages:
            if p.extraction_method in (
                ExtractionMethod.PENDING_VISION,
                ExtractionMethod.FAILED,
            ):
                reasons.append(f"page_{p.page}_needs_vision_or_failed")
            elif p.is_garbled:
                reasons.append(f"page_{p.page}_garbled:{p.garbled_reason}")
            elif p.confidence < 0.6:
                reasons.append(f"page_{p.page}_low_confidence:{p.confidence:.2f}")
        self.review_reasons = reasons
        self.needs_review = len(reasons) > 0


class ManifestEntry(BaseModel):
    """One row in the cache/manifest.json dedup ledger."""

    file_hash: str
    original_path: str
    document_id: str
    pipeline_version: str
    processed_at: str
