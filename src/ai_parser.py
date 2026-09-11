"""
AI Parser stage -- Layer 3.

Reads QuestionDocument JSON (produced by segmentation.py) and adds a
quality/confidence judgment on top of each already-structured
question. This stage does NOT re-parse anything -- question_text,
options, matching_pairs, and source_answer are segmentation's
canonical output and are never touched here. All it does is ask a
local model "does this record look complete and correctly parsed, how
confident should we be, what (if anything) looks off" and record the
answer in new ai_* fields on QuestionRecord (see question_schemas.py).
Same non-destructive, additive-only discipline as every other stage.

Why a judgment pass and not a second parse: segmentation.py already
does the structural work (regex/line-based), tested against real
files. This stage exists to catch things regex can't judge -- does
this *read* as a coherent, complete question, not just "did it match
a pattern".

Routing (computed fresh every run, never stored as its own field):
  - a question with ANY existing segmentation-time anomaly always
    routes to needs_review, no matter what the AI says. AI confidence
    can never override a structural flag already raised -- same
    principle as everywhere else in this pipeline.
  - a question that was never successfully reviewed (ai_confidence is
    None -- not yet run, or every attempt failed/was unparsable) also
    routes to needs_review. Silence from the model is never treated
    as confidence.
  - otherwise: ai_confidence >= --threshold -> auto_import,
    ai_confidence < --threshold -> needs_review.

Nothing is ever dropped in either bucket -- both are just views over
the same enriched per-document JSON files written to --output.

Local model via Ollama, no API key, no per-question cost. Default is
phi4-mini (3.8B, ~2.5-3GB RAM) rather than the glm-ocr used for vision
fill -- this is a genuine judgment task, not narrow OCR transcription,
so a small reasoning-tuned model matters more here than a
purpose-built OCR model would. --model is swappable if it doesn't
hold up on your real questions, same pattern as ollama_vision_fill.py.

Setup:
    ollama pull phi4-mini
    python ai_parser.py --input ../questions --output ../reviewed

Idempotent like the rest of the pipeline: a question is only marked
reviewed (ai_reviewed_at set) on a successful, parseable response. A
failed or garbled response leaves it untouched, so it's automatically
retried on the next run without needing --force.

NOT YET TESTED AGAINST REAL OLLAMA OUTPUT -- built and self-tested
with a mocked model response only (no Ollama available in the build
sandbox). Same situation ollama_vision_fill.py was in originally: run
this against your real ../questions output and spot-check a sample of
both buckets' confidence/concerns against the source questions before
trusting a large run.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from question_schemas import QuestionDocument, QuestionRecord  # noqa: E402

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"

SYSTEM_PROMPT = (
    "You are reviewing ALREADY-PARSED exam question records for quality -- "
    "you are not parsing raw text and you are not answering the question. "
    "You will be given a question's text, its options or matching pairs "
    "(if any), and the answer key value already extracted for it. Judge "
    "only whether this record looks complete, coherent, and correctly "
    "structured -- do not try to solve the question, and do not suggest a "
    "different answer.\n\n"
    "Respond with ONLY a JSON object, no other text, no markdown fences, "
    "in exactly this shape:\n"
    '{"confidence": <float 0.0-1.0>, "concerns": [<short strings>], '
    '"notes": <short string, or empty string>}\n\n'
    "confidence reflects how likely this record is a clean, complete, "
    "correctly-structured question -- NOT whether the medical content is "
    "correct. Lower it for: truncated or nonsensical question text, "
    "options that look cut off, duplicated, or missing, an answer key "
    "value that doesn't correspond to any option label shown, or anything "
    "else structurally off. concerns should be short machine-readable-ish "
    "phrases (e.g. \"answer_not_in_options\", \"truncated_stem\"), not full "
    "sentences -- leave the list empty if there are none. notes is for "
    "one short human-readable sentence if something needs explaining, "
    "otherwise leave it as an empty string."
)


def _question_prompt(q: QuestionRecord) -> str:
    lines = [f"section_type: {q.section_type.value}", f"question_text: {q.question_text}"]
    if q.options:
        lines.append("options:")
        for opt in q.options:
            lines.append(f"  {opt.label}) {opt.text}")
    if q.matching_pairs:
        lines.append("matching_pairs:")
        for mp in q.matching_pairs:
            lines.append(
                f"  {mp.left_label}) {mp.left_text}  <->  {mp.right_label}) {mp.right_text}"
            )
    lines.append(f"source_answer: {q.source_answer!r}")
    return "\n".join(lines)


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.S)


def parse_model_response(raw: str) -> dict | None:
    """Pulls a {confidence, concerns, notes} dict out of a raw model
    response. Returns None on anything unparsable -- callers must treat
    that the same as a failed call, never as "confidence 0"."""
    match = _JSON_BLOCK_RE.search(raw)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if "confidence" not in data:
        return None
    try:
        confidence = float(data["confidence"])
    except (TypeError, ValueError):
        return None
    confidence = max(0.0, min(1.0, confidence))

    concerns = data.get("concerns") or []
    if not isinstance(concerns, list):
        concerns = [str(concerns)]
    concerns = [str(c) for c in concerns]

    notes = str(data.get("notes") or "")

    return {"confidence": confidence, "concerns": concerns, "notes": notes}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def review_question(
    q: QuestionRecord, model: str, num_ctx: int, timeout: int = 120
) -> dict | None:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _question_prompt(q)},
        ],
        "stream": False,
        "options": {
            # Text-only prompts here (no image), so this needs far
            # less headroom than the vision-fill stage's 8192 -- but
            # still overridable per-call if a document has an unusually
            # long question/matching block.
            "num_ctx": num_ctx,
            # Low but not zero -- this is a judgment task, not
            # deterministic transcription, so some temperature is fine;
            # kept low so confidence scores don't swing wildly re-run
            # to re-run.
            "temperature": 0.2,
        },
    }
    try:
        resp = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=timeout)
    except requests.RequestException as exc:  # noqa: BLE001 -- one question
        # failing must never kill a batch run over thousands of them.
        print(f"  [ollama error] {q.question_id}: {exc}")
        return None

    if resp.status_code != 200:
        print(
            f"  [ollama error] {q.question_id}: HTTP {resp.status_code} -- {resp.text[:300]}"
        )
        return None

    body = resp.json()
    content = (body.get("message") or {}).get("content", "")
    result = parse_model_response(content)
    if result is None:
        preview = content[:200].replace("\n", " ")
        print(f"  [unparsable response] {q.question_id}: {preview!r}")
    return result


def review_document(
    doc: QuestionDocument, model: str, force: bool, sleep_s: float, num_ctx: int
) -> bool:
    changed = False
    for q in doc.questions:
        if q.ai_reviewed_at is not None and not force:
            continue

        result = review_question(q, model, num_ctx)
        if result is None:
            # Left exactly as it was -- not marked reviewed, so a
            # future run (even without --force) retries it. Silence
            # from the model must never be recorded as confidence.
            continue

        q.ai_confidence = result["confidence"]
        q.ai_concerns = result["concerns"]
        q.ai_notes = result["notes"] or None
        q.ai_model_used = f"ollama:{model}"
        q.ai_reviewed_at = _now_iso()
        changed = True

        if sleep_s:
            time.sleep(sleep_s)

    return changed


def route(q: QuestionRecord, threshold: float) -> str:
    """Returns "auto_import" or "needs_review". A structural anomaly
    from segmentation always wins over AI confidence; an unreviewed or
    unparsable question is never treated as confident."""
    if q.anomalies:
        return "needs_review"
    if q.ai_confidence is None:
        return "needs_review"
    if q.ai_confidence < threshold:
        return "needs_review"
    return "auto_import"


def build_buckets(
    docs: list[QuestionDocument], threshold: float
) -> tuple[list[dict], list[dict]]:
    """Builds the two flat bucket views across ALL documents, freshly
    computed every run (never incrementally appended, so they can never
    drift out of sync with the per-document files)."""
    auto_import: list[dict] = []
    needs_review: list[dict] = []

    for doc in docs:
        for q in doc.questions:
            entry = q.model_dump()
            entry["document_id"] = doc.document_id
            entry["original_filename"] = doc.original_filename
            entry["subject"] = doc.subject
            entry["topic"] = doc.topic
            bucket = auto_import if route(q, threshold) == "auto_import" else needs_review
            bucket.append(entry)

    return auto_import, needs_review


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add an AI confidence/quality pass to segmented questions and "
        "route them into auto_import / needs_review buckets."
    )
    parser.add_argument("--input", default="../questions", type=Path)
    parser.add_argument("--output", default="../reviewed", type=Path)
    parser.add_argument("--model", default="phi4-mini")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Minimum ai_confidence (0.0-1.0) to route an otherwise-clean question "
        "to auto_import instead of needs_review.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-review questions even if already reviewed by a previous run.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between calls -- rarely needed locally.",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=4096,
        help="Context window requested per call. Text-only prompts here need far "
        "less than the vision-fill stage's 8192, but raise this if a document has "
        "unusually long questions or large matching blocks.",
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

    input_root: Path = args.input.resolve()
    output_root: Path = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    doc_paths = sorted(input_root.glob("DOC-*.json"))
    if not doc_paths:
        print(f"No question documents found in {input_root}. Run segmentation.py first.")
        return

    stats = {"docs_touched": 0, "docs_skipped": 0, "questions_reviewed": 0}
    reviewed_docs: list[QuestionDocument] = []

    for doc_path in doc_paths:
        doc = QuestionDocument(**json.loads(doc_path.read_text()))
        before_reviewed = sum(1 for q in doc.questions if q.ai_reviewed_at is not None)

        changed = review_document(doc, args.model, args.force, args.sleep, args.num_ctx)

        after_reviewed = sum(1 for q in doc.questions if q.ai_reviewed_at is not None)
        stats["questions_reviewed"] += after_reviewed - before_reviewed

        out_path = output_root / doc_path.name
        out_path.write_text(json.dumps(doc.model_dump(), indent=2, default=str))
        reviewed_docs.append(doc)

        if changed:
            stats["docs_touched"] += 1
            print(f"[reviewed] {doc_path.name}")
        else:
            stats["docs_skipped"] += 1
            print(f"[unchanged] {doc_path.name} (already reviewed, use --force to redo)")

    auto_import, needs_review = build_buckets(reviewed_docs, args.threshold)
    (output_root / "_auto_import.json").write_text(
        json.dumps(auto_import, indent=2, default=str)
    )
    (output_root / "_needs_review.json").write_text(
        json.dumps(needs_review, indent=2, default=str)
    )

    print("\n--- AI Parser summary ---")
    for k, v in stats.items():
        print(f"{k}: {v}")
    print(f"auto_import: {len(auto_import)} questions")
    print(f"needs_review: {len(needs_review)} questions")


if __name__ == "__main__":
    main()
