"""
Turns ONE input file into ONE SourceDocument.

This is the only place that decides "what kind of file is this" and
"which extractor handles it". Everything else in the ingestion layer
operates on SourceDocument / PageExtraction, never on raw files again.
"""

from __future__ import annotations

from pathlib import Path

from hashing import Manifest, hash_file
from schemas import (
    PIPELINE_VERSION,
    FileType,
    ManifestEntry,
    PageExtraction,
    SourceDocument,
)
from extractors.pdf import extract_pdf
from extractors.docx import extract_docx
from extractors.image import extract_image

_EXT_TO_TYPE = {
    ".pdf": FileType.PDF,
    ".docx": FileType.DOCX,
    ".png": FileType.IMAGE,
    ".jpg": FileType.IMAGE,
    ".jpeg": FileType.IMAGE,
    ".webp": FileType.IMAGE,
}


class UnsupportedFileType(Exception):
    pass


def detect_file_type(path: Path) -> FileType:
    return _EXT_TO_TYPE.get(path.suffix.lower(), FileType.UNSUPPORTED)


def reconstruct_document(
    file_path: Path,
    *,
    subject: str | None,
    topic: str | None,
    manifest: Manifest,
    render_dir: Path,
    force: bool = False,
) -> tuple[SourceDocument | None, str]:
    """Returns (document_or_none, status).

    status is one of: "processed", "skipped_duplicate", "unsupported".
    A None document with status "skipped_duplicate" means the file's
    content hash was already processed by this pipeline_version --
    nothing was re-extracted, no cost was spent.
    """
    file_hash = hash_file(file_path)

    existing = manifest.already_processed(file_hash)
    # Same bytes AND same pipeline version -> genuinely nothing to do.
    # If the pipeline version has changed since this file was last
    # processed, it gets reprocessed even though the content hash
    # matches -- that's the "reprocess old records when the pipeline
    # changes" requirement.
    if existing and existing.pipeline_version == PIPELINE_VERSION and not force:
        return None, "skipped_duplicate"

    file_type = detect_file_type(file_path)
    if file_type == FileType.UNSUPPORTED:
        return None, "unsupported"

    if file_type == FileType.PDF:
        pages: list[PageExtraction] = extract_pdf(file_path, render_dir=render_dir)
    elif file_type == FileType.DOCX:
        pages = extract_docx(file_path)
    elif file_type == FileType.IMAGE:
        pages = extract_image(file_path)
    else:  # pragma: no cover -- guarded above
        return None, "unsupported"

    document_id = f"DOC-{file_hash[:12]}"

    doc = SourceDocument(
        document_id=document_id,
        original_filename=file_path.name,
        original_path=str(file_path),
        file_type=file_type,
        file_hash=file_hash,
        subject=subject,
        topic=topic,
        subtopic=None,  # placeholder -- filled in by a human, later
        pages=pages,
    )
    doc.compute_rollup()

    manifest.record(
        ManifestEntry(
            file_hash=file_hash,
            original_path=str(file_path),
            document_id=document_id,
            pipeline_version=doc.pipeline_version,
            processed_at=doc.ingested_at,
        )
    )

    return doc, "processed"
