"""
Vision fill-in stage using a local Ollama vision model.

Same job as vision_fill_anthropic.py -- transcribe pending_vision
pages using their saved page renders -- but calls a model running
entirely on your own machine via Ollama. No API key, no per-page
cost, nothing leaves your computer.

Trade-off, honestly stated: local models are currently a notch behind
frontier hosted vision models on messy handwriting or very cluttered
layouts. For clean-ish scanned exam text it should do well; spot-check
a sample against the rendered images before trusting a large batch.
`glm-ocr` specifically is a small (~0.9B parameter), narrowly-trained
OCR model rather than a general instruction-following chat model --
it may not respect the "[ILLEGIBLE]" / "don't add commentary" parts of
the prompt as reliably as a larger model would, since that kind of
instruction-following isn't really what it was trained for. Check its
actual output style on a page or two before assuming those conventions
are being followed.

Setup:
    1. Install Ollama: https://ollama.com/download
    2. Pull a vision-capable model tuned for document/OCR work. Pick
       based on your RAM -- these are all Ollama-library models:
           ollama pull glm-ocr             # ~2.5GB RAM, purpose-built for
                                            # document/table OCR -- runs
                                            # comfortably on any 8GB+ Mac,
                                            # good default
           ollama pull minicpm-v4.5        # ~6GB RAM, stronger general
                                            # document understanding, but
                                            # tight on an 8GB machine if
                                            # you're multitasking
           ollama pull llama3.2-vision:11b # ~8GB RAM, needs 16GB+ to
                                            # run comfortably alongside
                                            # other apps
    3. Leave the Ollama app running (it serves on localhost:11434).
    4. python ollama_vision_fill.py --output ../output --model glm-ocr

Same principles as the rest of this pipeline: transcribe only, never
answer/correct/invent, [ILLEGIBLE] instead of guessing, and this stage
is idempotent -- pages already filled in are skipped on a re-run
unless you pass --force.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from schemas import ExtractionMethod, SourceDocument  # noqa: E402
import text_quality  # noqa: E402

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"

# glm-ocr is a small (1.1B), narrowly-trained OCR model, not a general
# instruction-following chat model -- its own documented usage is the
# bare trigger phrase "Text Recognition:", nothing more elaborate.
# Real-world testing showed the long, instruction-heavy prompt below
# pushing it out of its trained distribution: it would transcribe part
# of a page, then loop back and re-read the page header, then
# re-transcribe the same questions again -- a restart artifact, not
# real duplicated content in the source. Using its native minimal
# prompt is the primary fix; general chat-tuned models (minicpm-v,
# llama3.2-vision) get the fuller instruction set since they're
# actually built to follow it.
GLM_OCR_PROMPT = "Text Recognition:"

CHAT_MODEL_PROMPT = (
    "Transcribe all visible text on this exam page exactly as it appears. "
    "Preserve line breaks, question numbering, and option labels (A/B/C/D, "
    "1/2/3, etc.) exactly as shown -- do not renumber or relabel anything. "
    "Do not answer, solve, correct, or explain anything. Do not add any "
    "commentary, headers, or notes of your own. If part of the page is "
    "genuinely illegible, write [ILLEGIBLE] in that spot instead of "
    "guessing. Output only the transcription, nothing else."
)


def prompt_for_model(model: str) -> str:
    return GLM_OCR_PROMPT if "glm-ocr" in model.lower() else CHAT_MODEL_PROMPT


def transcribe_image(
    image_path: Path, model: str, num_ctx: int, timeout: int = 180
) -> str | None:
    data = base64.standard_b64encode(image_path.read_bytes()).decode("utf-8")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt_for_model(model),
                "images": [data],
            }
        ],
        "stream": False,
        "options": {
            # The image itself can tokenize to well over Ollama's small
            # default context window (commonly 2048-4096) before any
            # text is even added. Ask for more room explicitly rather
            # than silently truncating on larger/denser page scans.
            "num_ctx": num_ctx,
            # Low temperature + a real repeat penalty as a second line
            # of defense against the looping/restart behavior seen in
            # testing, on top of the prompt fix above -- OCR should be
            # close to deterministic, not creative.
            "temperature": 0.1,
            "repeat_penalty": 1.3,
        },
    }
    try:
        resp = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=timeout)
    except requests.RequestException as exc:  # noqa: BLE001 -- one page
        # failing (server hiccup, timeout on a huge image) must never
        # kill a batch run over thousands of pages.
        print(f"  [ollama error] {image_path.name}: {exc}")
        return None

    if resp.status_code != 200:
        # The interesting diagnostic is in the response BODY, not the
        # status code -- print it instead of a generic "400 Bad
        # Request" that tells us nothing actionable.
        print(
            f"  [ollama error] {image_path.name}: HTTP {resp.status_code} -- {resp.text[:500]}"
        )
        return None

    body = resp.json()
    text = (body.get("message") or {}).get("content", "").strip()
    return text or None


def fill_document(
    doc_path: Path, model: str, force: bool, sleep_s: float, num_ctx: int
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

        text = transcribe_image(image_path, model, num_ctx)
        changed = True  # an attempt was made either way

        if text is None:
            continue  # left as pending_vision

        quality = text_quality.assess(text, min_chars=5)
        page.text = text
        page.extraction_method = ExtractionMethod.VISION_MODEL
        page.vision_model_used = f"ollama:{model}"

        if quality.is_garbled:
            page.is_garbled = True
            page.garbled_reason = quality.reason
            page.confidence = 0.3
        else:
            repeat_reason = text_quality.detect_repeated_block(text)
            if repeat_reason:
                # Model looped and re-emitted content it already
                # produced -- flag it, don't try to deduplicate
                # automatically. A human should see the original
                # doubled text and the rendered page, not a "cleaned"
                # version that hides what actually happened.
                page.is_garbled = True
                page.garbled_reason = f"repeated_content:{repeat_reason}"
                page.confidence = 0.3
            else:
                page.is_garbled = False
                page.garbled_reason = None
                page.confidence = quality.confidence

        page.char_count = quality.char_count
        # rendered_image_path stays -- provenance, so you can always
        # check a transcription against the original page.

        if sleep_s:
            time.sleep(sleep_s)

    if changed:
        doc.compute_rollup()
        doc_path.write_text(json.dumps(doc.model_dump(), indent=2, default=str))

    return changed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fill in pending_vision pages using a local Ollama vision model."
    )
    parser.add_argument("--output", default="../output", type=Path)
    parser.add_argument("--model", default="glm-ocr:latest")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-transcribe pages even if already filled in by a previous run.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between calls -- rarely needed locally, but useful "
        "if you're running something else on the GPU at the same time.",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=8192,
        help="Context window requested per call. The rendered page image alone "
        "can exceed Ollama's small default (often 2048-4096) before any text "
        "is added -- 8192 covers a typical single page with headroom.",
    )
    args = parser.parse_args()

    try:
        requests.get(OLLAMA_TAGS_URL, timeout=5)
    except requests.RequestException:
        print(
            "Can't reach Ollama at http://localhost:11434 -- is it installed and running?\n"
            "Install: https://ollama.com/download\n"
            f"Then in a terminal: ollama pull {args.model}"
        )
        return

    output_root: Path = args.output.resolve()
    doc_paths = sorted(output_root.glob("DOC-*.json"))

    if not doc_paths:
        print(f"No documents found in {output_root}. Run main.py first.")
        return

    stats = {"docs_touched": 0, "docs_skipped": 0}
    for doc_path in doc_paths:
        changed = fill_document(doc_path, args.model, args.force, args.sleep, args.num_ctx)
        if changed:
            stats["docs_touched"] += 1
            print(f"[vision-filled] {doc_path.name}")
        else:
            stats["docs_skipped"] += 1

    print("\n--- Vision fill summary (Ollama / local) ---")
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
