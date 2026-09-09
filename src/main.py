"""
Ingestion / reconstruction CLI.

    python main.py --input ../input --output ../output

Expected input layout (subject/topic decided by YOU, not the pipeline):

    input/
      Anatomy/
        Upper Limb/
          exam_2023.pdf
          telegram_screenshot_04.jpg
        Thorax/
          old_scan.pdf
      Physiology/
        renal_questions.docx

- Two folder levels under `input/`  -> subject = level 1, topic = level 2
- One folder level under `input/`   -> subject = level 1, topic = None
- File directly inside `input/`     -> subject = None, topic = None

`subtopic` is never inferred here -- it's always null in the output,
a placeholder for a later manual/review step.

Output: one JSON file per source document in `output/`, named
`<document_id>.json`. Files that fail extraction entirely are copied
into `failed/` alongside a `.error.txt` note, and the batch continues
rather than aborting.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hashing import Manifest  # noqa: E402
from reconstruct import reconstruct_document  # noqa: E402

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".png", ".jpg", ".jpeg", ".webp"}


def derive_subject_topic(file_path: Path, input_root: Path) -> tuple[str | None, str | None]:
    rel_parts = file_path.relative_to(input_root).parts[:-1]  # drop filename
    subject = rel_parts[0] if len(rel_parts) >= 1 else None
    topic = rel_parts[1] if len(rel_parts) >= 2 else None
    return subject, topic


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest raw files into reconstructed documents.")
    parser.add_argument("--input", default="../input", type=Path)
    parser.add_argument("--output", default="../output", type=Path)
    parser.add_argument("--failed", default="../failed", type=Path)
    parser.add_argument("--cache", default="../cache/manifest.json", type=Path)
    parser.add_argument(
        "--render-dir",
        default=None,
        type=Path,
        help="Where page renders for pending-vision pages get saved. "
        "Defaults to <output>/_renders.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess even files already recorded as processed in the manifest.",
    )
    args = parser.parse_args()

    input_root: Path = args.input.resolve()
    output_root: Path = args.output.resolve()
    failed_root: Path = args.failed.resolve()
    render_dir: Path = (args.render_dir or (output_root / "_renders")).resolve()

    output_root.mkdir(parents=True, exist_ok=True)
    failed_root.mkdir(parents=True, exist_ok=True)

    manifest = Manifest(args.cache.resolve())

    all_files = [
        p for p in input_root.rglob("*")
        if p.is_file() and not p.name.startswith(".")
    ]

    stats = {"processed": 0, "skipped_duplicate": 0, "unsupported": 0, "failed": 0, "needs_review": 0}

    for file_path in sorted(all_files):
        if file_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            stats["unsupported"] += 1
            print(f"[skip] unsupported file type: {file_path}")
            continue

        subject, topic = derive_subject_topic(file_path, input_root)

        try:
            doc, status = reconstruct_document(
                file_path,
                subject=subject,
                topic=topic,
                manifest=manifest,
                render_dir=render_dir,
                force=args.force,
            )
        except Exception as exc:  # noqa: BLE001 -- one bad file must never
            # kill a run processing thousands of files.
            stats["failed"] += 1
            failed_copy = failed_root / file_path.name
            try:
                shutil.copy2(file_path, failed_copy)
            except Exception:
                pass
            error_note = failed_root / f"{file_path.name}.error.txt"
            error_note.write_text(
                f"Failed during reconstruction.\n\n{traceback.format_exc()}"
            )
            print(f"[FAILED] {file_path} -> see {error_note}")
            continue

        if status == "skipped_duplicate":
            stats["skipped_duplicate"] += 1
            print(f"[dup]  {file_path} (already processed, content unchanged)")
            continue

        if status == "unsupported" or doc is None:
            stats["unsupported"] += 1
            print(f"[skip] {file_path} (unsupported)")
            continue

        out_path = output_root / f"{doc.document_id}.json"
        out_path.write_text(json.dumps(doc.model_dump(), indent=2, default=str))

        stats["processed"] += 1
        if doc.needs_review:
            stats["needs_review"] += 1
            print(f"[OK, FLAGGED] {file_path.name} -> {out_path.name}  reasons={doc.review_reasons}")
        else:
            print(f"[OK] {file_path.name} -> {out_path.name}")

    print("\n--- Summary ---")
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
