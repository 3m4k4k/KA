"""
Vision fill-in stage.

Reads the JSON documents produced by the ingestion/reconstruction
layer, finds every page marked `pending_vision`, and sends the saved
page render to a vision-capable Claude model for transcription.

This is deliberately a SEPARATE stage from reconstruction --
reconstruction never makes an AI call; this is the first place in the
pipeline that does. Keeping the boundary here means:

  - re-running ingestion costs nothing, ever
  - this stage is independently resumable/idempotent: a page already
    filled in is skipped on the next run unless --force is passed, so
    a batch of thousands of documents can be safely stopped and
    restarted without re-paying for pages already done
  - swapping models or providers later means touching this one file

The model is instructed to transcribe ONLY -- not answer, not correct,
not invent missing text -- same "never silently fix the source"
principle as the rest of this pipeline. Anything genuinely illegible
gets marked [ILLEGIBLE] inline rather than guessed at, and the
rendered image stays attached either way so a human can always check.

Requires:
    pip install anthropic --break-system-packages
    export ANTHROPIC_API_KEY=sk-ant-...
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from schemas import ExtractionMethod, SourceDocument  # noqa: E402
import text_quality  # noqa: E402

try:
    import anthropic
except ImportError:
    anthropic = None

TRANSCRIBE_PROMPT = (
    "Transcribe all visible text on this exam page exactly as it appears. "
    "Preserve line breaks, question numbering, and option labels (A/B/C/D, "
    "1/2/3, etc.) exactly as shown -- do not renumber or relabel anything. "
    "Do not answer, solve, correct, or explain anything. Do not add any "
    "commentary, headers, or notes of your own. If part of the page is "
    "genuinely illegible, write [ILLEGIBLE] in that spot instead of "
    "guessing. Output only the transcription, nothing else."
)


def _media_type(path: Path) -> str:
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(path.suffix.lower(), "image/png")


def transcribe_image(client, image_path: Path, model: str) -> str | None:
    data = base64.standard_b64encode(image_path.read_bytes()).decode("utf-8")
    try:
        response = client.messages.create(
            model=model,
            max_tokens=2000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": _media_type(image_path),
                                "data": data,
                            },
                        },
                        {"type": "text", "text": TRANSCRIBE_PROMPT},
                    ],
                }
            ],
        )
    except Exception as exc:  # noqa: BLE001 -- an API failure on one page
        # must never kill a batch run over thousands of documents; it
        # just leaves that page pending_vision for the next attempt.
        print(f"  [vision error] {image_path.name}: {exc}")
        return None

    text_blocks = [
        b.text for b in response.content if getattr(b, "type", None) == "text"
    ]
    joined = "\n".join(text_blocks).strip()
    return joined or None


def fill_document(
    client, doc_path: Path, model: str, force: bool, sleep_s: float
) -> bool:
    doc = SourceDocument(**json.loads(doc_path.read_text()))
    changed = False

    for page in doc.pages:
        should_process = page.extraction_method == ExtractionMethod.PENDING_VISION or (
            force and page.rendered_image_path is not None
        )
        if not should_process:
            continue
        if page.rendered_image_path is None:
            continue

        image_path = Path(page.rendered_image_path)
        if not image_path.exists():
            print(
                f"  [missing render] {doc.original_filename} page {page.page}: {image_path}"
            )
            continue

        text = transcribe_image(client, image_path, model)
        changed = True  # an attempt was made, even if it failed

        if text is None:
            continue  # left as pending_vision, nothing to record

        quality = text_quality.assess(text, min_chars=5)
        page.text = text
        page.extraction_method = ExtractionMethod.VISION_MODEL
        page.vision_model_used = model

        if quality.is_garbled:
            page.is_garbled = True
            page.garbled_reason = quality.reason
            page.confidence = 0.3
        else:
            repeat_reason = text_quality.detect_repeated_block(text)
            if repeat_reason:
                page.is_garbled = True
                page.garbled_reason = f"repeated_content:{repeat_reason}"
                page.confidence = 0.3
            else:
                page.is_garbled = False
                page.garbled_reason = None
                page.confidence = quality.confidence

        page.char_count = quality.char_count
        # rendered_image_path is left in place -- it's provenance, not
        # a marker of failure, and a human should always be able to
        # check the transcription against the original image.

        if sleep_s:
            time.sleep(sleep_s)

    if changed:
        doc.compute_rollup()
        doc_path.write_text(json.dumps(doc.model_dump(), indent=2, default=str))

    return changed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fill in pending_vision pages using a Claude vision model."
    )
    parser.add_argument("--output", default="../output", type=Path)
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-transcribe pages even if already filled in by a previous run.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between API calls (basic rate-limit safety).",
    )
    args = parser.parse_args()

    if anthropic is None:
        print(
            "The 'anthropic' package isn't installed.\n"
            "Run: pip install anthropic --break-system-packages"
        )
        return

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

    output_root: Path = args.output.resolve()
    doc_paths = sorted(output_root.glob("DOC-*.json"))

    if not doc_paths:
        print(f"No documents found in {output_root}. Run main.py first.")
        return

    stats = {"docs_touched": 0, "docs_skipped": 0}
    for doc_path in doc_paths:
        changed = fill_document(client, doc_path, args.model, args.force, args.sleep)
        if changed:
            stats["docs_touched"] += 1
            print(f"[vision-filled] {doc_path.name}")
        else:
            stats["docs_skipped"] += 1

    print("\n--- Vision fill summary ---")
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
