"""
Throwaway probe: local Ollama model + Ollama's hosted web_search API.

NOT a pipeline stage -- writes nothing to any QuestionRecord. Its job
is answering one question before the real correctness-check stage
gets designed: does a small LOCAL model actually call the search tool
when it should, and does the search API work on the free tier?

Unlike the Gemini and CodeCraft probes, search usage here is MEASURED,
not self-reported: this script runs the tool-calling loop itself, so
every web_search / web_fetch call the model makes passes through our
code and gets logged. We never have to trust the model's own claim
about whether it searched.

How it works: the model runs locally (your machine). When it decides
to search, this script calls Ollama's hosted search endpoint
(ollama.com/api/web_search), feeds the results back, and lets the
model continue -- up to --max-rounds rounds.

Requires:
    pip install --upgrade ollama
    export OLLAMA_API_KEY=...      # free key: ollama.com/settings/keys
    ollama pull qwen3:4b           # or any model tagged "tools" on ollama.com

Usage:
    # 1. Cheapest pass/fail: does the model search when it MUST?
    python3 test_ollama_search.py --model qwen3:4b \\
        --test-question "What was announced at the most recent major tech company earnings call this week?"

    # 2. Then real questions:
    python3 test_ollama_search.py --model qwen3:4b \\
        --input reviewed/DOC-56eb58bdf252.questions.json --count 5
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from ollama import chat, web_fetch, web_search

TOOLS = {"web_search": web_search, "web_fetch": web_fetch}
MAX_TOOL_CHARS = 6000  # keep tool output from swamping the context window

MCQ_PROMPT = (
    "You are fact-checking ONE exam question. Do NOT rely on your own "
    "memory: use the web_search tool to find evidence, then judge whether "
    "the given answer is correct.\n\n"
    "Question: {question_text}\nOptions:\n{options}\n"
    "Given answer: {answer}\n\n"
    "When done, respond with ONLY a JSON object, no markdown fences:\n"
    '{{"agrees": <true/false>, "confidence": <0.0-1.0, based on how well '
    'the evidence you found supports your judgment, not how sure you '
    'feel>, "suggested_answer": <option label or null>, '
    '"evidence_notes": <short string>, "sources": [<urls you relied on>]}}'
)

FREEFORM_PROMPT = (
    "Answer this question. If you are not certain from memory, use the "
    "web_search tool.\n\nQuestion: {question}\n\n"
    'Respond with ONLY a JSON object: {{"answer": <short string>, '
    '"confidence": <0.0-1.0>, "sources": [<urls>]}}'
)


def run_agent(model: str, prompt: str, max_rounds: int, num_ctx: int) -> dict:
    messages = [{"role": "user", "content": prompt}]
    searches: list[str] = []
    fetches: list[str] = []
    errors: list[str] = []
    final_text = None
    status = "max_rounds_exceeded"

    for _ in range(max_rounds):
        response = chat(
            model=model,
            messages=messages,
            tools=[web_search, web_fetch],
            options={"num_ctx": num_ctx},
        )
        messages.append(response.message)

        if not response.message.tool_calls:
            final_text = response.message.content
            status = "answered"
            break

        for call in response.message.tool_calls:
            name = call.function.name
            args = call.function.arguments
            if name == "web_search":
                searches.append(str(args.get("query")))
            elif name == "web_fetch":
                fetches.append(str(args.get("url")))
            fn = TOOLS.get(name)
            try:
                result = str(fn(**args)) if fn else f"unknown tool {name}"
            except Exception as exc:  # noqa: BLE001 -- surface quota/auth
                # errors verbatim; the exact wording is the data we want.
                result = f"TOOL ERROR: {exc}"
                errors.append(f"{name}: {exc}")
                print(f"  [tool error] {name}: {exc}")
            messages.append(
                {"role": "tool", "tool_name": name, "content": result[:MAX_TOOL_CHARS]}
            )

    return {
        "status": status,
        "search_calls": len(searches),
        "search_queries": searches,
        "fetch_calls": len(fetches),
        "fetched_urls": fetches,
        "tool_errors": errors,
        "final_text": final_text,
    }


def format_options(options: list[dict]) -> str:
    return "\n".join(f"  {o['label']}) {o['text']}" for o in options)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe Ollama local model + web_search.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--test-question", help="Ask one free-form question instead.")
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--sleep", type=float, default=2.0)
    args = parser.parse_args()

    if not os.environ.get("OLLAMA_API_KEY"):
        print("Set OLLAMA_API_KEY first (free key: ollama.com/settings/keys).")
        return

    if args.test_question:
        result = run_agent(
            args.model,
            FREEFORM_PROMPT.format(question=args.test_question),
            args.max_rounds,
            args.num_ctx,
        )
        print(json.dumps(result, indent=2))
        print(f"\nsearch_calls = {result['search_calls']}  (measured by this script, not self-reported)")
        return

    if not args.input:
        print("Need --input (or --test-question).")
        return

    doc = json.loads(args.input.read_text())
    questions = [
        q for q in doc["questions"] if q.get("verified_answer") or q.get("source_answer")
    ][: args.count]

    results = []
    total_searches = 0
    for i, q in enumerate(questions, start=1):
        print(f"[{i}/{len(questions)}] {q['question_id']}...")
        prompt = MCQ_PROMPT.format(
            question_text=q["question_text"],
            options=format_options(q["options"]),
            answer=q.get("verified_answer") or q.get("source_answer"),
        )
        try:
            result = run_agent(args.model, prompt, args.max_rounds, args.num_ctx)
        except Exception as exc:  # noqa: BLE001 -- keep going, want data
            print(f"  [error] {exc}")
            continue
        result["question_id"] = q["question_id"]
        results.append(result)
        total_searches += result["search_calls"]
        print(f"  status={result['status']}  searches={result['search_calls']}  {result['search_queries']}")
        print(f"  -> {result['final_text']}")
        if args.sleep:
            time.sleep(args.sleep)

    print("\n--- Ollama search probe summary ---")
    print(f"model: {args.model}")
    print(f"questions probed: {len(results)}")
    print(f"total web_search calls (measured): {total_searches}")
    if results:
        print(f"avg searches/question: {total_searches / len(results):.2f}")
        print(f"questions with any tool error: {sum(1 for r in results if r['tool_errors'])}")

    out_path = args.input.parent / "_ollama_search_probe_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()