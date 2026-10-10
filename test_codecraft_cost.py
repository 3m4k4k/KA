"""
Throwaway probe: tests CodeCraft API (codecraftapi.com) as a possible
correctness-check backend, in place of Gemini.

This is deliberately NOT a pipeline stage -- same as
test_grounding_cost.py before it, this doesn't write to any
QuestionRecord and isn't meant to be kept around. Its job is finding
out what CodeCraft's API actually does in practice, since its public
docs are the only thing we have to go on and we've now twice watched
a provider's docs not match live behavior (Gemini's model
availability, and its grounding free tier).

IMPORTANT GAP, read before running: CodeCraft's own Chat Completions
reference (https://codecraftapi.com/docs/chat-completions) does NOT
document any request parameter to enable/disable web search, and does
NOT document a response field listing what was searched -- unlike
Gemini's groundingMetadata.webSearchQueries. The `web_search`
capability flag on a model (from GET /v1/models) appears to mean the
model searches automatically when it decides to, with no toggle. This
script therefore does NOT try to force or measure search count the
way the Gemini probe did -- it just calls a web_search-capable model
normally and dumps the FULL raw response so you can look for
citations/evidence yourself and decide if it's usable.

Requires:
    pip install openai --break-system-packages
    export CODECRAFT_API_KEY=...   (from codecraftapi.com/dashboard)

Usage:
    # First, find out which models actually have web_search:
    python test_codecraft_cost.py --list-models

    # Then probe with one of those model IDs:
    python test_codecraft_cost.py --input ../reviewed/DOC-56eb58bdf252.questions.json \\
        --model <model-id-with-web_search> --count 5
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from openai import OpenAI

BASE_URL = "https://codecraftapi.com/v1"

# Same "ground it in evidence, not in how sure it feels" discipline as
# every other AI-judgment stage in this pipeline.
PROMPT_TEMPLATE = (
    "You are fact-checking ONE exam question against real, verifiable "
    "sources -- not your own training memory. If you have web search "
    "available, use it to check whether the given answer is actually "
    "correct.\n\n"
    "Question: {question_text}\n"
    "Options:\n{options}\n"
    "Given answer: {answer}\n\n"
    "Respond with ONLY a JSON object, no markdown fences, in exactly this "
    "shape:\n"
    '{{"agrees": <true/false>, "confidence": <float 0.0-1.0>, '
    '"suggested_answer": <option label or null>, "evidence_notes": '
    '<short string>, "used_search": <true/false -- did you actually search, '
    "or answer from memory>}}\n\n"
    "confidence should reflect how well evidence you found actually "
    "supports your judgment -- not how sure you feel. Be honest about "
    "used_search; do not claim you searched if you didn't."
)


# Separate, simpler prompt for --test-question: no MCQ options to
# format, and deliberately does NOT hint that search is optional --
# the whole point of this mode is forcing a clean read on whether
# used_search ever flips to true at all, on a question picked
# specifically because confident recall shouldn't be possible.
TEST_QUESTION_PROMPT = (
    "Answer this question. If you are not already certain of the answer "
    "from memory, use web search to check it.\n\n"
    "Question: {question}\n\n"
    "Respond with ONLY a JSON object, no markdown fences, in exactly this "
    "shape:\n"
    '{{"answer": <short string>, "confidence": <float 0.0-1.0>, '
    '"used_search": <true/false -- be honest, do not claim you searched '
    "if you didn't>}}"
)


def probe_test_question(client: OpenAI, model: str, question: str) -> dict:
    prompt = TEST_QUESTION_PROMPT.format(question=question)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    choice = response.choices[0]
    return {
        "question": question,
        "raw_text": choice.message.content,
        "full_response": response.model_dump(),
    }


def format_options(options: list[dict]) -> str:
    return "\n".join(f"  {o['label']}) {o['text']}" for o in options)


def list_web_search_models(client: OpenAI) -> None:
    models = client.models.list()
    found = [
        m for m in models.data
        if "web_search" in (getattr(m, "capabilities", None) or [])
    ]
    if not found:
        print(
            "No models reporting a 'web_search' capability were returned. "
            "Either none are available on your plan, or the capability is "
            "exposed differently than documented -- print the raw response "
            "and check manually:"
        )
        for m in models.data:
            print(f"  {m.id}: {getattr(m, 'capabilities', None)}")
        return

    print("Models with web_search capability:")
    for m in found:
        pricing = getattr(m, "pricing", None)
        print(f"  {m.id}  (context: {getattr(m, 'context_window', '?')}, pricing: {pricing})")


def probe_question(client: OpenAI, model: str, q: dict) -> dict:
    prompt = PROMPT_TEMPLATE.format(
        question_text=q["question_text"],
        options=format_options(q["options"]),
        answer=q.get("verified_answer") or q.get("source_answer"),
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )

    choice = response.choices[0]
    # Dump the full response dict -- since CodeCraft's docs don't
    # document a citations/search-evidence field, we don't know what
    # to look for in advance. Capture everything, inspect after.
    raw_response = response.model_dump()

    return {
        "question_id": q["question_id"],
        "raw_text": choice.message.content,
        "full_response": raw_response,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe CodeCraft API as a correctness-check backend. "
        "Throwaway diagnostic, not a pipeline stage."
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--model", help="Model ID with web_search capability.")
    parser.add_argument("--sleep", type=float, default=2.0)
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Just list models with a web_search capability, then exit.",
    )
    parser.add_argument(
        "--test-question",
        help="Ask one arbitrary question directly (bypasses --input/--count) "
        "-- use this with something the model can't know from training data "
        "(recent, obscure) to test whether used_search ever actually fires.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("CODECRAFT_API_KEY")
    if not api_key:
        print("Set CODECRAFT_API_KEY first (from codecraftapi.com/dashboard).")
        return

    client = OpenAI(api_key=api_key, base_url=BASE_URL)

    if args.list_models:
        list_web_search_models(client)
        return

    if args.test_question:
        if not args.model:
            print("--model is required with --test-question.")
            return
        result = probe_test_question(client, args.model, args.test_question)
        print(result["raw_text"])
        print(
            "\nCheck used_search above. If it's still false on a question "
            "the model genuinely can't know from training data, the "
            "web_search capability likely isn't actually firing through "
            "this endpoint -- if it's true, the capability is real and it "
            "was just choosing not to use it on the easier questions."
        )
        return

    if not args.input or not args.model:
        print("--input and --model are required unless using --list-models.")
        return

    doc = json.loads(args.input.read_text())
    questions = [
        q for q in doc["questions"]
        if q.get("verified_answer") or q.get("source_answer")
    ][: args.count]

    if not questions:
        print(f"No answerable questions found in {args.input}.")
        return

    results = []
    for i, q in enumerate(questions, start=1):
        print(f"[{i}/{len(questions)}] probing {q['question_id']}...")
        try:
            result = probe_question(client, args.model, q)
        except Exception as exc:  # noqa: BLE001 -- one bad probe shouldn't
            # kill the rest; we want data from every question we can get.
            print(f"  [error] {exc}")
            continue

        results.append(result)
        print(f"  -> {result['raw_text']}")

        if args.sleep:
            time.sleep(args.sleep)

    print("\n--- CodeCraft probe summary ---")
    print(f"model: {args.model}")
    print(f"questions probed: {len(results)}")
    print(
        "\nLook through the saved full_response objects for anything "
        "resembling citations, sources, or search evidence -- field names "
        "aren't documented, so this has to be eyeballed. If nothing shows "
        "up, 'used_search' in the model's own JSON answer is the only "
        "signal we have, and it's self-reported (the model could be wrong "
        "about its own behavior), so treat it as a hint, not proof."
    )

    out_path = args.input.parent / "_codecraft_probe_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()