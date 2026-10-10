"""
Throwaway probe for OpenRouter: measures how many web-search server tool 
calls models actually trigger per correctness-check call.

This is deliberately NOT a pipeline stage -- it doesn't write to any
QuestionRecord, doesn't touch reviewed/ files, and isn't meant to be
kept around. Its only job is answering one question before the real
correctness-check stage gets designed: how many search queries does an
OpenRouter model execute per question when given the web search server tool.

Requires:
    pip install requests --break-system-packages
    export OPENROUTER_API_KEY=...   (from OpenRouter dashboard)

Usage:
    python test_openrouter_grounding_cost.py --input ../reviewed/DOC-56eb58bdf252.questions.json --count 5
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
import requests

PROMPT_TEMPLATE = (
    "You are fact-checking ONE exam question against real, verifiable "
    "sources -- not your own training memory. Use web search to check "
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
    api_key: str, model: str, q: dict, use_grounding: bool = True
) -> dict:
    prompt = PROMPT_TEMPLATE.format(
        question_text=q["question_text"],
        options=format_options(q["options"]),
        answer=q.get("verified_answer") or q.get("source_answer"),
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # Optional headers for OpenRouter rankings/site attribution
        "HTTP-Referer": "https://github.com/local-probe",
        "X-Title": "Correctness Check Probe",
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }

    # Attach OpenRouter's universal web search tool if enabled
    if use_grounding:
        payload["tools"] = [{"type": "openrouter:web_search"}]

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()

    # Extract response content and look for tool/search metadata
    choice = data.get("choices", [{}])[0]
    message = choice.get("message", {})
    raw_text = message.get("content", "")
    
    # OpenRouter returns executed tools or search metadata in message structures
    # or annotations depending on the provider backend. Let's aggregate safely.
    queries: list[str] = []
    
    # Check if tool calls were logged in the response
    tool_calls = message.get("tool_calls", [])
    for call in tool_calls:
        if call.get("type") == "openrouter:web_search" or "search" in call.get("function", {}).get("name", ""):
            args = call.get("function", {}).get("arguments", "{}")
            try:
                parsed_args = json.loads(args) if isinstance(args, str) else args
                if "query" in parsed_args:
                    queries.append(parsed_args["query"])
            except Exception:
                pass

    # Fallback search query parsing from provider annotations if present
    if not queries and "annotations" in message:
        for ann in message["annotations"]:
            if ann.get("type") == "url_citation" and "url" in ann:
                queries.append(ann["url"])

    return {
        "question_id": q["question_id"],
        "search_query_count": len(queries) if queries else (1 if use_grounding and "provider_name" in data else 0), 
        # Note: Some models handle web_search transparently server-side 
        # without exposing raw query strings back in the message object.
        "search_queries": queries,
        "raw_text": raw_text,
        "usage": data.get("usage", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe OpenRouter web search server tool cost per question. "
        "Throwaway diagnostic, not a pipeline stage."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument(
        "--model",
        default="openai/gpt-4o-mini", # Change to a free or preferred model on OpenRouter
        help="Specify the OpenRouter model string (e.g., 'openai/gpt-4o-mini' or a free model).",
    )
    parser.add_argument("--sleep", type=float, default=3.0)
    parser.add_argument(
        "--no-grounding",
        action="store_true",
        help="Diagnostic only: omit the openrouter:web_search tool.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Set OPENROUTER_API_KEY first.")
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
    total_queries = 0
    for i, q in enumerate(questions, start=1):
        print(f"[{i}/{len(questions)}] probing {q['question_id']} with model {args.model}...")
        try:
            result = probe_question(
                api_key, args.model, q, use_grounding=not args.no_grounding
            )
        except Exception as exc:
            print(f"  [error] {exc}")
            continue

        total_queries += result["search_query_count"]
        results.append(result)
        n = result["search_query_count"]
        print(f"  -> {n} search item(s) tracked: {result['search_queries']}")

        if args.sleep:
            time.sleep(args.sleep)

    print("\n--- OpenRouter Grounding Cost Probe Summary ---")
    print(f"model: {args.model}")
    print(f"questions probed: {len(results)}")
    print(f"total search markers counted: {total_queries}")
    if results:
        print(f"avg search metrics/question: {total_queries / len(results):.2f}")
    
    out_path = args.input.parent / "_openrouter_grounding_probe_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()