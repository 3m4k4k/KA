"""
Throwaway probe: measures structural model validation and response behavior
via an OpenAI-compatible API client interface.

This is deliberately NOT a pipeline stage -- it doesn't write to any
QuestionRecord, doesn't touch reviewed/ files, and isn't meant to be
kept around. Its only job is testing single question validation and
parsing behavior before the real correctness-check stage gets designed.

Requires:
    pip install openai --break-system-packages
    export API_KEY=...   (or your chosen environment variable)

Usage:
    python test_agentkthx_probe.py --input ../reviewed/DOC-56eb58bdf252.questions.json --count 5
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from openai import OpenAI

# Deliberately asks the model to ground its judgment in rigorous academic
# standards -- same discipline as the rest of this pipeline's AI stages 
# (never self-reported confidence as ground truth).
PROMPT_TEMPLATE = (
    "You are fact-checking ONE exam question against rigorous academic "
    "standards -- not just your own training memory. Evaluate whether the "
    "given answer is actually correct.\n\n"
    "Question: {question_text}\n"
    "Options:\n{options}\n"
    "Given answer: {answer}\n\n"
    "Respond with ONLY a JSON object, no markdown fences, in exactly this "
    "shape:\n"
    '{{"agrees": <true/false>, "confidence": <float 0.0-1.0>, '
    '"suggested_answer": <option label or null>, "evidence_notes": '
    '<short string>}}\n\n'
    "confidence should reflect how well the evidence or principles "
    "support your judgment -- not how sure you feel."
)


def format_options(options: list[dict]) -> str:
    return "\n".join(f"  {o['label']}) {o['text']}" for o in options)


def probe_question(
    client: OpenAI, model: str, q: dict, dry_run: bool = False
) -> dict:
    prompt = PROMPT_TEMPLATE.format(
        question_text=q["question_text"],
        options=format_options(q["options"]),
        answer=q.get("verified_answer") or q.get("source_answer"),
    )

    # --dry-run is a diagnostic-only mode: it tests connection and parsing
    # behavior without full evaluation logic if needed.
    if dry_run:
        return {
            "question_id": q["question_id"],
            "parsed_response": {"dry_run": True},
            "raw_text": "DRY_RUN_MODE",
        }

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a precise academic proctor."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
    )

    raw_text = response.choices[0].message.content
    parsed = {}
    try:
        # Clean potential markdown fences defensively
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("\n", 1)[0]
        parsed = json.loads(cleaned.strip())
    except Exception:
        pass  # keep raw_text available even if parsing fails

    return {
        "question_id": q["question_id"],
        "parsed_response": parsed,
        "raw_text": raw_text,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe OpenAI-compatible model validation per question. "
        "Throwaway diagnostic, not a pipeline stage."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument(
        "--model",
        default="gemini-3.8-flash",
        help="Model identifier to use with the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--base-url",
        default="https://generativelanguage.googleapis.com/v1beta/openai/",
        help="API base URL for the OpenAI-compatible provider.",
    )
    parser.add_argument(
        "--api-key-env",
        default="GEMINI_API_KEY",
        help="Name of the environment variable holding your API key.",
    )
    parser.add_argument("--sleep", type=float, default=4.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Diagnostic only: skip actual API generation calls to test flow.",
    )
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(f"Set environment variable {args.api_key_env} first.")
        return

    client = OpenAI(base_url=args.base_url, api_key=api_key)

    doc = json.loads(args.input.read_text())
    questions = [
        q for q in doc["questions"]
        if q.get("verified_answer") or q.get("source_answer")
    ][: args.count]

    if not questions:
        print(f"No answerable questions found in {args.input}.")
        return

    results = []
    success_count = 0
    for i, q in enumerate(questions, start=1):
        print(f"[{i}/{len(questions)}] probing {q['question_id']}...")
        try:
            result = probe_question(client, args.model, q, dry_run=args.dry_run)
        except Exception as exc:  # noqa: BLE001 -- one bad probe shouldn't kill the rest
            print(f"  [error] {exc}")
            continue

        results.append(result)
        agrees = result["parsed_response"].get("agrees")
        if agrees is not None:
            success_count += 1
        print(f"  -> agrees: {agrees}, confidence: {result['parsed_response'].get('confidence')}")

        if args.sleep:
            time.sleep(args.sleep)

    print("\n--- Model probe summary ---")
    print(f"model: {args.model}")
    print(f"questions probed: {len(results)}")
    print(f"successful parses: {success_count}")
    print(
        "\nFull results written to the output file below. "
        "Inspect the parsed response shapes to confirm schema compliance."
    )

    out_path = args.input.parent / "_agentkthx_probe_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"Full results written to {out_path}")


if __name__ == "__main__":
    main()