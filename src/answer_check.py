"""
Answer-correctness + explanation stage -- Layer 5.

For every MCQ in the reviewed QuestionDocuments, asks a hosted LLM
(via groq_router.py, which handles Groq's free-tier limits and model
fallback) two things in one call:

  1. Is the answer key factually right?  (second opinion, not truth)
  2. Why is the right option right, and why is each other option wrong?

Results land in the additive check_* fields on QuestionRecord. Nothing
the pipeline or a human already wrote is touched: source_answer,
verified_answer, question text, options and every ai_* field stay
exactly as they were. Same flag-don't-repair discipline as every other
stage -- a disagreement becomes a flag for the review UI, never an
edit to the key.

Design points
-------------
  - Operates IN PLACE on --input (default ../reviewed), the same
    canonical files review_server.py writes to, so human corrections
    (verified_answer, verified_question_text, verified_options) are
    what gets checked: the checked answer is verified_answer if set,
    else source_answer.
  - Idempotent and resumable. A question is skipped when it already has
    a check whose input fingerprint (text + options + answer) still
    matches; if a reviewer later edits any of those, it's re-checked
    automatically. --force redoes everything.
  - check_agrees is DERIVED here by comparing the model's suggested
    option to the key. The model's own "agrees" flag is only used when
    it names no usable option, and a contradiction between the two is
    recorded as a concern (the same lesson as ai_parser's mcq-35).
  - A response that can't be parsed/validated leaves the question
    unchecked (retried next run) -- silence is never recorded as a pass.
  - Stops cleanly when every model is out of daily budget: progress is
    saved, the resume time is printed, re-run later.
  - Scope for now: MCQ with a usable key. fill_in_blank and matching are
    skipped (counted in the summary), not guessed at.

IMPORTANT: don't run this while review_server.py is running. Both
rewrite the whole document file; whichever writes last wins and would
silently drop the other's changes. Stop the server, run this, restart.

Requires:
    export GROQ_API_KEY=...

Usage:
    python answer_check.py --dry-run                 # what would be checked, no API calls
    python answer_check.py --input ../reviewed
    python answer_check.py --only 20,47,56           # specific question numbers
    python answer_check.py --limit 50                # at most 50 questions this run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from groq_router import (  # noqa: E402
    AllModelsExhausted,
    CallResult,
    FatalAPIError,
    GroqRouter,
    RouterError,
    extract_json,
)
from question_schemas import Option, QuestionDocument, QuestionRecord, QuestionType  # noqa: E402

SYSTEM_PROMPT = (
    "You are checking ONE first-year medical school exam question "
    "(anatomy, histology, physiology, biochemistry). Judge using your own "
    "textbook knowledge. The question text was extracted by OCR, so it may "
    "contain typos, run-together words or stray non-English characters -- "
    "read through those where the meaning is clear.\n\n"
    "Respond with ONLY a JSON object, no markdown fences, in exactly this "
    "shape:\n"
    '{"agrees": <true/false>, "suggested_answer": <option label, or null if '
    'no option is correct>, "explanation": <1-2 sentences on why the correct '
    'option is right>, "why_others_wrong": {<label>: <short reason>, ...}, '
    '"concerns": [<short snake_case strings>]}\n\n'
    "Decide the correct option independently FIRST, then compare with the "
    "given answer: 'agrees' is true only if the given answer is that option. "
    "'why_others_wrong' covers every option except the one you picked. "
    "Use these concern tags when they apply: garbled_option, stem_truncated, "
    "ambiguous_question, no_correct_option, multiple_correct_options, "
    "key_likely_wrong. If the question is flawed, say so with a concern "
    "instead of forcing an answer. Keep every string short."
)


# ---------------------------------------------------------------------------
# Record helpers
# ---------------------------------------------------------------------------


def effective_fields(q: QuestionRecord) -> tuple[str, list[Option], str | None]:
    """Human-corrected fields win over the originals (originals are
    never edited -- see review_server.py)."""
    text = q.verified_question_text or q.question_text
    options = q.verified_options if q.verified_options is not None else q.options
    answer = q.verified_answer or q.source_answer
    return text, options, answer


def input_hash(text: str, options: list[Option], answer: str | None) -> str:
    payload = json.dumps([text, [[o.label, o.text] for o in options], answer], ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def skip_reason(q: QuestionRecord) -> str | None:
    """Why this question can't be checked at all (None = eligible)."""
    if q.section_type != QuestionType.MCQ:
        return f"type_{q.section_type.value}"
    text, options, answer = effective_fields(q)
    if not answer:
        return "no_answer_key"
    if len(options) < 2:
        return "fewer_than_2_options"
    if answer.strip().lower() not in {o.label.strip().lower() for o in options}:
        # A structural problem (e.g. a dropped option) -- a human should
        # fix the options first; checking would just produce noise.
        return "answer_not_in_options"
    return None


def build_prompt(text: str, options: list[Option], answer: str) -> str:
    opts = "\n".join(f"  {o.label}) {o.text}" for o in options)
    return f"Question: {text}\nOptions:\n{opts}\nGiven answer: {answer}"


def validate(parsed: dict | None, options: list[Option], given: str) -> dict | None:
    """Turn a raw model reply into a normalized verdict, or None if it
    isn't usable. Deterministic checks sit on top of the model's claims."""
    if not isinstance(parsed, dict):
        return None
    labels = {o.label.strip().lower(): o.label for o in options}
    given_l = given.strip().lower()
    concerns: list[str] = []

    raw_sug = parsed.get("suggested_answer")
    sug_label = None
    if raw_sug not in (None, "", "null"):
        sug_label = labels.get(str(raw_sug).strip().lower())
        if sug_label is None:
            concerns.append("suggested_answer_not_an_option")

    model_agrees = parsed.get("agrees") if isinstance(parsed.get("agrees"), bool) else None

    if sug_label is not None:
        agrees = sug_label.strip().lower() == given_l
        if model_agrees is not None and model_agrees != agrees:
            concerns.append("agrees_flag_inconsistent")
    elif model_agrees is not None:
        agrees = model_agrees
        concerns.append("no_valid_suggested_option")
    else:
        return None

    explanation = str(parsed.get("explanation") or "").strip()
    if not explanation:
        concerns.append("missing_explanation")

    why: dict[str, str] = {}
    raw_why = parsed.get("why_others_wrong")
    if isinstance(raw_why, dict):
        for k, v in raw_why.items():
            label = labels.get(str(k).strip().lower())
            if label is not None and label != sug_label:
                why[label] = str(v).strip()

    for c in parsed.get("concerns") or []:
        c = str(c).strip()
        if c and c not in concerns:
            concerns.append(c)

    return {
        "agrees": agrees,
        "suggested": sug_label,
        "explanation": explanation or None,
        "why": why or None,
        "concerns": concerns,
    }


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# One question
# ---------------------------------------------------------------------------


def check_question(
    router: GroqRouter,
    q: QuestionRecord,
    max_completion: int,
    retry_completion: int,
    est_completion: int,
) -> tuple[dict, CallResult] | None:
    text, options, answer = effective_fields(q)
    prompt = build_prompt(text, options, answer)

    # Second attempt exists mainly for reasoning models that spend the
    # whole first budget thinking (finish_reason == "length") and return
    # nothing usable; it costs more tokens, so it's only tried then / on
    # a parse failure.
    for budget in (max_completion, retry_completion):
        res = router.chat(
            SYSTEM_PROMPT,
            prompt,
            max_completion_tokens=budget,
            est_completion_tokens=est_completion,
        )
        verdict = validate(extract_json(res.text), options, answer)
        if verdict is not None:
            return verdict, res
        print(f"    [unusable reply from {res.model}, finish={res.finish_reason}, budget={budget}]")
    return None


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------


def run(
    router: GroqRouter | None,
    input_root: Path,
    *,
    force: bool = False,
    only: set[str] | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    max_completion: int = 1000,
    retry_completion: int = 1800,
    est_completion: int = 600,
) -> dict:
    doc_paths = sorted(input_root.glob("DOC-*.json"))
    stats = {
        "checked": 0, "agreed": 0, "disagreed": 0, "with_concerns": 0,
        "already_done": 0, "unusable_reply": 0, "to_check": 0,
    }
    skipped: dict[str, int] = {}
    by_model: dict[str, int] = {}
    budget_hit = False

    for doc_path in doc_paths:
        doc = QuestionDocument(**json.loads(doc_path.read_text()))
        dirty = False
        limit_reached = False
        try:
            for q in doc.questions:
                if only and q.question_number not in only:
                    continue
                reason = skip_reason(q)
                if reason:
                    skipped[reason] = skipped.get(reason, 0) + 1
                    continue

                text, options, answer = effective_fields(q)
                h = input_hash(text, options, answer)
                if q.check_reviewed_at and q.check_input_hash == h and not force:
                    stats["already_done"] += 1
                    continue

                if dry_run:
                    stats["to_check"] += 1
                    continue
                if limit is not None and stats["checked"] >= limit:
                    limit_reached = True
                    break

                out = check_question(router, q, max_completion, retry_completion, est_completion)
                if out is None:
                    stats["unusable_reply"] += 1
                    continue
                v, res = out

                q.check_given_answer = answer
                q.check_agrees = v["agrees"]
                q.check_suggested_answer = v["suggested"]
                q.check_explanation = v["explanation"]
                q.check_why_others_wrong = v["why"]
                q.check_concerns = v["concerns"]
                q.check_model_used = f"groq:{res.model}"
                q.check_input_hash = h
                q.check_reviewed_at = now_iso()
                dirty = True

                stats["checked"] += 1
                stats["agreed" if v["agrees"] else "disagreed"] += 1
                if v["concerns"]:
                    stats["with_concerns"] += 1
                by_model[res.model] = by_model.get(res.model, 0) + 1
                flag = "" if v["agrees"] and not v["concerns"] else "   <-- needs a look"
                print(
                    f"[{doc.document_id[4:]} q{q.question_number}] key={answer} "
                    f"model={v['suggested']} {'agrees' if v['agrees'] else 'DISAGREES'} "
                    f"({res.model}){' ' + str(v['concerns']) if v['concerns'] else ''}{flag}"
                )

                # Save as we go: a crash / Ctrl+C / budget stop loses at
                # most the question in flight.
                if stats["checked"] % 5 == 0:
                    doc_path.write_text(json.dumps(doc.model_dump(), indent=2, default=str))
                    dirty = False
        except AllModelsExhausted as exc:
            budget_hit = True
            if exc.resume_in_s == float("inf"):
                print("\n[stopped] a request is too big for every model's limits -- not a wait, a config problem.")
            else:
                when = datetime.now() + timedelta(seconds=exc.resume_in_s)
                print(
                    f"\n[stopped] all models are out of budget. Earliest one frees in "
                    f"{exc.resume_in_s / 60:.0f} min (about {when:%H:%M}). Progress is saved; re-run then."
                )
        finally:
            if dirty:
                doc_path.write_text(json.dumps(doc.model_dump(), indent=2, default=str))
        if budget_hit or limit_reached:
            break

    stats["skipped"] = skipped
    stats["by_model"] = by_model
    stats["stopped_on_budget"] = budget_hit
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check answer keys and generate explanations with Groq (via groq_router)."
    )
    parser.add_argument("--input", default="../reviewed", type=Path)
    parser.add_argument("--force", action="store_true", help="Re-check even if already checked.")
    parser.add_argument("--only", help="Comma-separated question numbers, e.g. 20,47,56")
    parser.add_argument("--limit", type=int, help="Max questions to check this run.")
    parser.add_argument("--dry-run", action="store_true", help="Count what would be checked; no API calls.")
    parser.add_argument("--max-completion", type=int, default=1000,
                        help="max_completion_tokens per call. Groq charges this against the "
                        "per-minute token limit up front, so keep it near what's needed.")
    parser.add_argument("--retry-completion", type=int, default=1800)
    parser.add_argument("--max-wait", type=float, default=120.0,
                        help="Longest the router will sleep for a limit to free up before stopping.")
    args = parser.parse_args()

    input_root = args.input.resolve()
    if not input_root.exists():
        print(f"Input directory not found: {input_root}")
        return

    router = None
    if not args.dry_run:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("Set GROQ_API_KEY first.")
            return
        router = GroqRouter(api_key, max_wait_s=args.max_wait)

    only = {n.strip() for n in args.only.split(",")} if args.only else None
    try:
        stats = run(
            router, input_root,
            force=args.force, only=only, limit=args.limit, dry_run=args.dry_run,
            max_completion=args.max_completion, retry_completion=args.retry_completion,
        )
    except FatalAPIError as exc:
        print(f"\n[fatal] {exc}")
        return
    except RouterError as exc:
        print(f"\n[router error] {exc}")
        return

    print("\n--- Answer check summary ---")
    if args.dry_run:
        print(f"would check: {stats['to_check']}")
        print(f"already checked (up to date): {stats['already_done']}")
    else:
        for k in ("checked", "agreed", "disagreed", "with_concerns", "already_done", "unusable_reply"):
            print(f"{k}: {stats[k]}")
        for m, n in stats["by_model"].items():
            print(f"  answered by {m}: {n}")
    if stats["skipped"]:
        print("skipped (can't check): " + ", ".join(f"{k}={v}" for k, v in sorted(stats["skipped"].items())))
    if router is not None:
        print("\nbudget now:")
        print("\n".join(router.status_lines()))


if __name__ == "__main__":
    main()
