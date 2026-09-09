"""
DOCX extraction.

Word documents have no reliable page concept at the file-format level
(pagination depends on the renderer, fonts, margins). Rather than
fake page numbers, this treats the whole document as a single logical
unit (page=1) and preserves paragraph and table structure in the text
so later stitching/segmentation still has something to work with.

If you later find real page breaks matter for a given source set,
python-docx can detect explicit <w:br w:type="page"/> elements -- that
would split logical_page into multiple PageExtraction entries. Not
done here to avoid guessing at a granularity you haven't asked for.
"""

from __future__ import annotations

import sys
from pathlib import Path

import docx

sys.path.append(str(Path(__file__).resolve().parent.parent))
from schemas import ExtractionMethod, PageExtraction  # noqa: E402
import text_quality  # noqa: E402


def extract_docx(file_path: Path) -> list[PageExtraction]:
    document = docx.Document(str(file_path))

    parts: list[str] = []
    for para in document.paragraphs:
        if para.text.strip():
            parts.append(para.text)

    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))

    full_text = "\n".join(parts)
    quality = text_quality.assess(full_text)

    return [
        PageExtraction(
            page=1,
            text=full_text if full_text.strip() else None,
            extraction_method=ExtractionMethod.DOCX_TEXT,
            confidence=quality.confidence if not quality.is_garbled else 0.0,
            is_garbled=quality.is_garbled,
            garbled_reason=quality.reason,
            char_count=quality.char_count,
        )
    ]
