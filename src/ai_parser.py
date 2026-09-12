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
  - a question with a non-empty ai_concerns list always routes to
    needs_review, regardless of ai_confidence. See the fourth item in
    the fix history below for why confidence alone isn't sufficient.
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

Ran once against both real documents (174 questions, phi4-mini). Real-
output inspection surfaced two problems, both traced to SYSTEM_PROMPT,
not to phi4-mini being unusably bad:

  1. The original prompt's "does source_answer correspond to an option
     label" check was written MCQ-first and applied uniformly to every
     section_type -- every fill_in_blank (no options, by design) and
     matching (matching_pairs + a compound-key source_answer, not flat
     options) question was getting wrongly zeroed to confidence 0.0.
     Fixed by branching the validity criteria by section_type in the
     prompt below.
  2. On at least one mcq, the model scored confidence 0.0 and
     explained itself with an anatomical correctness argument, despite
     the prompt explicitly saying confidence is NOT about whether the
     medical content is correct. That's the model overriding an
     explicit instruction boundary, not a wording gap -- the prompt
     below repeats and emphasizes the "not a medical opinion" boundary
     several times as a mitigation, but this should be re-checked
     against real output rather than assumed fixed. If phi4-mini keeps
     doing this after the re-run, the next move is trying a different
     --model, not further prompt tweaking.

Both failure modes routed to needs_review, not auto_import (the safe
direction -- see routing rule above), but that means until this is
re-verified, needs_review can't yet be trusted as "these specifically
need human attention for the stated reason" -- some entries in it may
be there because of these prompt bugs rather than a real issue with
the question.

RE-RUN AGAINST REAL OUTPUT (both documents, 174 questions) confirmed
both fixes: fill_in_blank/matching are no longer wrongly zeroed
(auto_import went 148->160, needs_review 26->14, and every remaining
needs_review entry traces to a real, pre-existing segmentation
anomaly), and no medical-correctness language turned up anywhere in
ai_notes/ai_concerns across either document, including on mcq-47
(the record that originally triggered Bug 2).

That same real-output inspection surfaced a THIRD, narrower issue:
DOC-2c9ba7d2333e-mcq-50 has an obviously duplicated question_text
("X: X:") and 3 repeated option labels -- a vision-transcription-loop
artifact, the same failure mode text_quality.detect_repeated_block()
exists to catch at the page level -- but phi4-mini scored it
ai_confidence=1.0 with no concerns. It was still correctly routed to
needs_review because the segmentation-time anomaly
(unexpected_option_count:7) wins regardless of AI confidence, so
nothing slipped into auto_import -- but it means the AI pass alone is
not a reliable independent check for this failure mode. Fix: added
detect_structural_duplication(), a deterministic (non-LLM) check that
runs on every question after the model's own review and can only push
ai_confidence down / add concerns, never suppress or override them.
This is a backstop for duplication specifically, not a replacement for
the LLM's broader structural judgment.

That same real-output inspection also surfaced a FOURTH issue, this
time in route() rather than SYSTEM_PROMPT: DOC-2c9ba7d2333e-mcq-35 has
ai_confidence=1.0 but its own ai_concerns list contains "source_answer
does not match any option label" -- a false claim, since source_answer
'b' does match option label 'b' in that record. phi4-mini's confidence
score and its own concerns list can be internally inconsistent, and
route() was only checking anomalies and the confidence threshold, so
this record would have gone to auto_import despite the model itself
flagging something. Fixed by having route() also force needs_review
whenever ai_concerns is non-empty, regardless of confidence. Checked
across all 160 auto_import questions from the previous run: mcq-35 was
the only one with a non-empty ai_concerns list, so this was an
isolated inconsistency, not a systemic pattern of low-stakes concerns
riding along with justified high confidence -- gating unconditionally
does not meaningfully shrink auto_import. If a future document set
shows the opposite pattern, narrow this to specific concern categories
instead of reverting to confidence-only routing.
"""

from __future__ import annotations

import argparse
import difflib
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
    "You are reviewing ALREADY-PARSED exam question records for STRUCTURAL "
    "quality only -- you are not parsing raw text and you are not answering "
    "the question. You will be given a question's section_type, its text, "
    "its options or matching pairs (if any), and the answer key value "
    "already extracted for it.\n\n"
    "HARD BOUNDARY, repeated because it is the single most important rule "
    "here: you are NOT a medical or anatomy expert for this task, and you "
    "must not act like one. Do not evaluate whether the answer is "
    "medically/anatomically/factually correct. Do not reason about what the "
    "right answer would be. Do not lower confidence, add a concern, or "
    "write a note because you believe the stated answer is medically wrong "
    "or because you personally would have picked a different option. A "
    "record with a medically debatable or even medically wrong answer, but "
    "with clean structure, gets HIGH confidence and NO concerns -- medical "
    "correctness is entirely out of scope for you and belongs to a "
    "separate verification stage you are not part of. If you notice "
    "yourself forming an opinion about which answer is medically correct, "
    "discard that thought -- it is not relevant to this task.\n\n"
    "What you DO judge is structural completeness, and the criteria depend "
    "on section_type -- do not apply MCQ-style checks to a type they don't "
    "fit:\n\n"
    "- section_type == mcq: the question should have options, and "
    "source_answer should correspond to one of the option labels shown. "
    "Lower confidence if source_answer doesn't match any option label, "
    "options look cut off/duplicated/missing, or the stem is truncated or "
    "nonsensical.\n"
    "- section_type == fill_in_blank: there are NO options, by design -- "
    "never flag missing/no options for this type, that is expected and "
    "correct. source_answer is free-text (the blank's answer), not a "
    "label -- do not check it against any option list. Judge only whether "
    "question_text reads as a complete, coherent sentence with a sensible "
    "blank, and whether source_answer looks like a plausible short "
    "answer/fill-in value (not empty, not truncated, not obviously the "
    "wrong data type e.g. a whole sentence dumped where a word is "
    "expected).\n"
    "- section_type == matching: there are matching_pairs, not flat "
    "options -- never check source_answer against option labels for this "
    "type. source_answer here is a compound key like \"1-B,2-C,3-A\" "
    "referencing left_label-right_label pairs. Judge whether every "
    "left_label referenced in source_answer has a corresponding entry in "
    "matching_pairs, whether the compound key's format looks well-formed "
    "and complete (not missing entries, not malformed), and whether the "
    "pairs themselves look complete/uncut.\n\n"
    "Respond with ONLY a JSON object, no other text, no markdown fences, "
    "in exactly this shape:\n"
    '{"confidence": <float 0.0-1.0>, "concerns": [<short strings>], '
    '"notes": <short string, or empty string>}\n\n'
    "confidence reflects ONLY how likely this record is a clean, complete, "
    "correctly-structured record for its section_type, using the "
    "type-specific criteria above -- never medical/factual correctness. "
    "concerns should be short machine-readable-ish phrases appropriate to "
    "the section_type (e.g. \"answer_not_in_options\" for mcq, "
    "\"answer_key_missing_left_label\" for matching, \"truncated_stem\"), "
    "not full sentences -- leave the list empty if there are none. notes "
    "is for one short human-readable sentence if something structural "
    "needs explaining, otherwise leave it as an empty string. Never put a "
    "medical-correctness opinion in notes."
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


def _self_repeat_ratio(text: str, min_len: int = 20) -> float:
    """Heuristic: does `text` say the same thing twice back-to-back?

    This is the question-level counterpart to
    text_quality.detect_repeated_block(), which catches a vision model
    looping and re-emitting a whole *page* it already transcribed.
    That same artifact can also land inside a single question_text
    field once segmentation has chopped a page into questions (e.g.
    "The segments of the spinal cord are: The segments of the spinal
    cord are:"). detect_repeated_block() operates on blank-line-
    separated blocks and doesn't apply to a single short string, so
    this is a simpler, purpose-built check: split the text at its
    midpoint and compare the two halves. A near-exact match on a
    string of meaningful length means the whole thing repeated, not
    that it coincidentally started the same way -- real question
    stems tested well under 0.4 (see module tests), the real
    duplicate case scored 0.99.
    """
    stripped = text.strip()
    if len(stripped) < min_len * 2:
        return 0.0
    mid = len(stripped) // 2
    a, b = stripped[:mid], stripped[mid:]
    return difflib.SequenceMatcher(None, a, b).ratio()


_DUPLICATION_RATIO_THRESHOLD = 0.85


def detect_structural_duplication(q: QuestionRecord) -> list[str]:
    """Deterministic, model-independent backstop for a specific known
    failure mode: a vision-transcription loop (see
    ollama_vision_fill.py) re-emitting content it already produced.
    Confirmed against real output that phi4-mini's judgment pass does
    NOT reliably catch this on its own even when structurally obvious
    (DOC-2c9ba7d2333e-mcq-50: duplicated question_text plus 3 repeated
    option labels, scored ai_confidence=1.0 with no concerns) -- so
    this check runs independently of what the LLM says, and its
    findings are never suppressed by a high ai_confidence. This is a
    backstop specifically for duplication, not a replacement for the
    LLM's broader judgment; it doesn't try to catch everything the
    prompt asks the model to judge.
    """
    found: list[str] = []

    if q.question_text and _self_repeat_ratio(q.question_text) >= _DUPLICATION_RATIO_THRESHOLD:
        found.append("duplicated_question_text")

    if q.options:
        labels = [opt.label for opt in q.options]
        if len(set(labels)) < len(labels):
            found.append("duplicate_option_labels")

    if q.matching_pairs:
        left_labels = [mp.left_label for mp in q.matching_pairs]
        right_labels = [mp.right_label for mp in q.matching_pairs]
        if len(set(left_labels)) < len(left_labels):
            found.append("duplicate_matching_left_labels")
        if len(set(right_labels)) < len(right_labels):
            found.append("duplicate_matching_right_labels")

    return found


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

        # Deterministic backstop, independent of the model's own
        # judgment -- see detect_structural_duplication(). Merged in
        # rather than trusted alone: the model's concerns/notes are
        # kept, this only ever adds to them and can only push
        # confidence down, never up.
        dup_concerns = detect_structural_duplication(q)
        if dup_concerns:
            for c in dup_concerns:
                if c not in result["concerns"]:
                    result["concerns"].append(c)
            result["confidence"] = min(result["confidence"], 0.2)
            if not result["notes"]:
                result["notes"] = (
                    "Flagged by deterministic duplication check "
                    "(independent of AI judgment)."
                )

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
    unparsable question is never treated as confident.

    A non-empty `ai_concerns` also forces needs_review, regardless of
    `ai_confidence`. Real-output inspection (DOC-2c9ba7d2333e-mcq-35)
    found phi4-mini raising a concern ("source_answer does not match
    any option label") while still scoring ai_confidence=1.0 -- the
    confidence score and the model's own concerns list can disagree
    with each other. Trusting confidence alone in that case would have
    silently put an internally-inconsistent record into auto_import.
    Checked across all 174 real questions: this was a single isolated
    case, not a systemic pattern of low-stakes concerns riding along
    with high confidence, so gating unconditionally on "any concern"
    doesn't meaningfully dilute the auto_import bucket. If a future
    batch shows the opposite -- lots of routine/informational concerns
    alongside justified high confidence -- narrow this to specific
    concern categories instead of loosening it back to confidence-only.
    """
    if q.anomalies:
        return "needs_review"
    if q.ai_confidence is None:
        return "needs_review"
    if q.ai_confidence < threshold:
        return "needs_review"
    if q.ai_concerns:
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
