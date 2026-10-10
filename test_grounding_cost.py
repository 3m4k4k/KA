"""
Throwaway probe: measures how many Google Search grounding queries
Gemini actually burns per correctness-check call.

This is deliberately NOT a pipeline stage -- it doesn't write to any
QuestionRecord, doesn't touch reviewed/ files, and isn't meant to be
kept around. Its only job is answering one question before the real
correctness-check stage gets designed: does a single grounded call
cost 1 search or several per question, and how much does that vary.

IMPORTANT: run this against whichever model you actually intend to
use in production. Google's billing rule differs by model generation
-- Gemini 2.5 and older bill per PROMPT (one grounded call = 1,
regardless of how many searches it runs internally), Gemini 3.x bill
per SEARCH QUERY (a call that runs 3 searches = 3). Testing on the
wrong generation will give you numbers that don't transfer.

Requires:
    pip install google-genai --break-system-packages
    export GEMINI_API_KEY=...   (from Google AI Studio)

Usage:
    python test_grounding_cost.py --input ../reviewed/DOC-56eb58bdf252.questions.json --count 5
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from google import genai
from google.genai import types

# Deliberately asks the model to ground its judgment in what it finds,
# not in how sure it "feels" -- same discipline as the rest of this
# pipeline's AI stages (never self-reported confidence as ground truth).
PROMPT_TEMPLATE = (
    "You are fact-checking ONE exam question against real, verifiable "
    "sources -- not your own training memory. Use Google Search to check "
    "whether the given answer is actually correct.\n\n"
    "Question: {question_text}\n"
    "Options:\n{options}\n"
    "Given answer: {answer}\n\n"
    "Respond with ONLY a JSON object, no markdown fences, in exactly this "
    "shape:\n"
    '{{"agrees": <true/false>, "confidence": <float 0.0-1.0>, '
    '"suggested_answer": <option label or null>, "evidence_notes": '
    '<short string>}}\n\n'
    "confidence should reflect how well the search results you found "
    "actually support your judgment -- not how sure you feel."
)


def format_options(options: list[dict]) -> str:
    return "\n".join(f"  {o['label']}) {o['text']}" for o in options)


def probe_question(
    client: genai.Client, model: str, q: dict, use_grounding: bool = True
) -> dict:
    prompt = PROMPT_TEMPLATE.format(
        question_text=q["question_text"],
        options=format_options(q["options"]),
        answer=q.get("verified_answer") or q.get("source_answer"),
    )

    # --no-grounding is a diagnostic-only mode: it strips the
    # google_search tool entirely so a plain, ungrounded call can be
    # used to isolate whether a failure is specific to grounding (a
    # currently-documented free-tier issue on some new projects) or
    # is a base account/model access problem instead. It intentionally
    # skips the "don't trust self-reported confidence" prompt logic
    # for this same reason -- this mode exists to test connectivity,
    # not to produce a usable correctness judgment.
    config = types.GenerateContentConfig()
    if use_grounding:
        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        config = types.GenerateContentConfig(tools=[grounding_tool])

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )

    queries: list[str] = []
    try:
        metadata = response.candidates[0].grounding_metadata
        queries = list(metadata.web_search_queries or [])
    except (AttributeError, IndexError, TypeError):
        pass  # no grounding metadata came back -- record 0, don't crash

    return {
        "question_id": q["question_id"],
        "search_query_count": len(queries),
        "search_queries": queries,
        "raw_text": response.text,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe real Gemini grounding search cost per question. "
        "Throwaway diagnostic, not a pipeline stage."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument(
        "--model",
        default="gemini-flash-latest",
        help="Use the exact model you plan to run in production -- billing "
        "counts differently on Gemini 2.5 vs 3.x, see module docstring.",
    )
    parser.add_argument("--sleep", type=float, default=4.0)
    parser.add_argument(
        "--no-grounding",
        action="store_true",
        help="Diagnostic only: strip the google_search tool and make a plain "
        "call instead, to check whether a failure is specific to grounding "
        "or is a base account/model access problem.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY first (get one from Google AI Studio).")
        return

    client = genai.Client(api_key=api_key)

    doc = json.loads(args.input.read_text())
    questions = [
        q for q in doc["questions"]
        if q.get("verified_answer") or q.get("source_answer")
    ][: args.count]

    if not questions:
        print(f"No answerable questions found in {args.input}.")
        return

    results = []
    total_queries = 0
    for i, q in enumerate(questions, start=1):
        print(f"[{i}/{len(questions)}] probing {q['question_id']}...")
        try:
            result = probe_question(
                client, args.model, q, use_grounding=not args.no_grounding
            )
        except Exception as exc:  # noqa: BLE001 -- one bad probe shouldn't
            # kill the rest; we want data from every question we can get.
            print(f"  [error] {exc}")
            continue

        total_queries += result["search_query_count"]
        results.append(result)
        n = result["search_query_count"]
        print(f"  -> {n} search quer{'y' if n == 1 else 'ies'}: {result['search_queries']}")

        if args.sleep:
            time.sleep(args.sleep)

    print("\n--- Grounding cost probe summary ---")
    print(f"model: {args.model}")
    print(f"questions probed: {len(results)}")
    print(f"total search queries counted: {total_queries}")
    if results:
        print(f"avg searches/question: {total_queries / len(results):.2f}")
    print(
        "\nNow open the AI Studio quota dashboard and compare its "
        "'Search Grounding' usage delta against total_queries above. "
        "If they match, this script's counting method is trustworthy for "
        "sizing the real correctness-check stage. If they don't match, "
        "trust the dashboard, not this script."
    )

    out_path = args.input.parent / "_grounding_probe_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()