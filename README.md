# Question Bank Pipeline — Ingestion / Reconstruction Layer (v0.1)

This is **only** the first stage of the full pipeline:

```
ANY FILE
   ↓
reconstructed document
   ↓
clean/usable representation
   ↓
with page/layout/provenance preserved
```

There is **no LLM question parser, no taxonomy classification, no
answer verification** in this version, on purpose. The goal right now
is to get raw files into a trustworthy, inspectable, structured form
before any AI touches question content. Test this against your real
files first — the JSON output is exactly what the future question
parser will read.

## What it does

1. Walks your `input/` folder.
2. Reads **subject** and **topic** from the folder structure (you
   decide these — the pipeline never guesses). `subtopic` is always
   `null`: a placeholder for a manual/review step you haven't built
   yet.
3. Hashes each file's content. If a file with that exact hash has
   already been processed by the current pipeline version, it's
   skipped — no re-extraction, no re-spending on OCR/vision later.
   If you bump `PIPELINE_VERSION` in `src/schemas.py`, previously
   processed files with the old version get reprocessed automatically.
4. Extracts text:
   - **PDF** — tries the embedded text layer first.
   - **DOCX** — paragraphs + tables via `python-docx`.
   - **Images** (screenshots, scans) — Tesseract OCR if installed,
     otherwise routed straight to review.
5. Runs every extraction result through a quality check
   (`src/text_quality.py`) that looks for more than just "is it
   empty" — it also catches replacement characters, broken
   encodings, repeated-character spam, and suspiciously low
   real-word ratios (common with broken PDF fonts).
6. Anything that fails the quality check gets:
   - marked `pending_vision` (not silently kept or discarded),
   - a rendered PNG of the page saved to `output/_renders/`,
   - a specific machine-readable reason (`garbled_reason`), not just
     a bare confidence number.
7. Writes one JSON file per document to `output/`, named
   `<document_id>.json`, where `document_id` is derived from the
   content hash (`DOC-<hash prefix>`) — stable across re-runs.
8. Files that error out completely (corrupt file, unreadable format)
   are copied to `failed/` with an `.error.txt` explaining why. **One
   bad file never stops the batch.**

## Folder structure it expects

```
input/
  Anatomy/
    Upper Limb/
      exam_2023.pdf
      telegram_screenshot_04.jpg
    Thorax/
      old_scan.pdf
  Physiology/
    renal_questions.docx
```

- Two folder levels under `input/` → `subject` = level 1, `topic` = level 2
- One folder level → `subject` = level 1, `topic` = `null`
- File directly in `input/` → both `null`

## Running it

```bash
pip install -r requirements.txt --break-system-packages   # or use a venv
cd src
python main.py --input ../input --output ../output --failed ../failed --cache ../cache/manifest.json
```

Useful flags:
- `--force` — reprocess everything, ignoring the dedup manifest.
- `--render-dir` — where flagged-page images get saved (defaults to `output/_renders`).

## Example output (`output/DOC-abc123456789.json`)

```json
{
  "document_id": "DOC-abc123456789",
  "pipeline_version": "0.1.0",
  "original_filename": "anatomy_exam.pdf",
  "original_path": "/path/to/input/Anatomy/Upper Limb/anatomy_exam.pdf",
  "file_type": "pdf",
  "file_hash": "abc123456789...",
  "subject": "Anatomy",
  "topic": "Upper Limb",
  "subtopic": null,
  "pages": [
    {
      "page": 1,
      "text": "...",
      "extraction_method": "pdf_text",
      "confidence": 0.87,
      "is_garbled": false,
      "garbled_reason": null,
      "char_count": 1204,
      "rendered_image_path": null
    }
  ],
  "needs_review": false,
  "review_reasons": []
}
```

A flagged page looks the same but with `extraction_method:
"pending_vision"`, `confidence: 0.0`, a `garbled_reason`, and a
`rendered_image_path` pointing at the saved page image.

## Things deliberately left undone (next layers, not this one)

- **Multi-page question stitching.** This layer preserves pages
  separately with correct ordering; joining a question that starts on
  page 4 and finishes on page 5 belongs to the question-segmentation
  stage, once there's actual question text to reason about.
- **OCR/vision for flagged pages.** `extractors/image.py` and
  `extractors/pdf.py` stop at "here's a clean image, saved to disk."
  Wire in Tesseract (already stubbed) or a vision model call
  wherever `pending_vision` shows up — kept as a separate step on
  purpose so you can swap providers without touching extraction logic.
- **Taxonomy / classification** — intentionally removed from this
  pipeline for now, per your call. `subject`/`topic` come from folders
  you control; `subtopic` stays `null` everywhere until you decide to
  add that layer.
- **Answer parsing, verification, duplicate detection, explanations**
  — all later stages, all reading from this layer's JSON output, none
  of them built yet.

## A note on the quality heuristics

`text_quality.assess()` is a **routing signal**, not a calibrated
probability. It decides "trust this / don't trust this," and the
`confidence` number it returns should not be treated as "% likely to
be correct" anywhere downstream. Once you've run this against a
batch of your real files, it's worth spot-checking a sample of both
the passed and flagged pages — if it's over- or under-flagging on
your specific source material (e.g. heavily symbol/formula-based
pages triggering false positives), the thresholds in that one file
are the only thing that needs tuning.
