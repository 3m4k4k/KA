"""
Throwaway probe: tests Ollama's native web_search API as a retrieval
source for the correctness-check stage, paired with a locally-running
model doing the actual judgment.

Deliberately NOT a pipeline stage -- same as the two probes before it,
this doesn't touch any QuestionRecord and isn't meant to be kept.

ARCHITECTURE NOTE: this does NOT let the model decide on its own
whether/when to call web_search as an autonomous tool-call loop --
that pattern exists in Ollama's docs, but its exact tool-calling
syntax wasn't independently confirmed here, and wiring it wrong would
make this probe untrustworthy in the same way trusting a provider's
docs already burned us twice tonight. Instead this does explicit,
inspectable retrieval-then-reason, same shape as the Wikipedia/PubMed
design:
    1. Build a search query from the question text
    2. Call ollama.web_search() directly -- this is the actual
       uncertain thing we're testing: does it return real content,
       is it really free, what does the response look like
    3. Paste whatever came back into the local model's prompt as
       evidence, and ask THAT model (running fully on your machine)
       to judge based on it

HONEST GAPS, read before running:
  - ollama.web_search() is a HOSTED call to Ollama's own servers, not
    fully local -- it needs OLLAMA_API_KEY even though the reasoning
    model runs on your machine. Not free of any third-party dependency,
    just a different one than Gemini/CodeCraft.
  - The exact shape of what web_search() returns wasn't independently
    verified before writing this -- the extraction below tries a few
    plausible field names defensively and always dumps the raw object
    too, so you can see the real shape rather than trust an assumption.
  - No confirmed exact free-tier quota number exists anywhere I found
    -- described as "generous" by third parties, nothing more precise.
    Watch for auth/quota errors same as we did with the other two.

Requires:
    pip install --upgrade ollama --break-system-packages
    ollama pull qwen3:4b          # or any tool/chat-capable model you
                                   # already have pulled -- pass via --model
    ollama serve                  # local server must be running
    export OLLAMA_API_KEY=...     # free key from ollama.com/settings/keys

Usage:
    python3 test_ollama_websearch.py --input ../reviewed/DOC-56eb58bdf252.questions.json --count 5
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import ollama

PROMPT_TEMPLATE = (
    "You are fact-checking ONE exam question. Below is evidence retrieved "
    "from a real web search -- base your judgment ONLY on this evidence, "
    "not on what you already believe you know.\n\n"
    "EVIDENCE:\n{evidence}\n\n"
    "Question: {question_text}\n"
    "Options:\n{options}\n"
    "Given answer: {answer}\n\n"
    "Respond with ONLY a JSON object, no markdown fences, in exactly this "
    "shape:\n"
    '{{"agrees": <true/false>, "confidence": <float 0.0-1.0>, '
    '"suggested_answer": <option label or null>, "evidence_notes": '
    '<short string citing what in the evidence supports your judgment>}}\n\n'
    "If the evidence above is empty, irrelevant, or doesn't actually "
    "address the question, say so honestly in evidence_notes and set "
    "confidence low -- do not fall back on your own unverified memory."
)


def format_options(options: list[dict]) -> str:
    return "\n".join(f"  {o['label']}) {o['text']}" for o in options)


def extract_search_results(raw) -> tuple[str, list[dict]]:
    """Defensively pulls (title, url, content) triples out of whatever
    ollama.web_search() actually returns. Field names weren't confirmed
    ahead of time, so this tries a few plausible shapes and falls back
    to dumping the raw object rather than silently returning nothing.
    """
    candidates = None
    if hasattr(raw, "results"):
        candidates = raw.results
    elif isinstance(raw, dict) and "results" in raw:
        candidates = raw["results"]
    elif isinstance(raw, list):
        candidates = raw

    if not candidates:
        return f"[no 'results' found -- raw object was: {raw!r}]", []

    parsed = []
    for item in candidates:
        title = getattr(item, "title", None) or (item.get("title") if isinstance(item, dict) else None) or "?"
        url = getattr(item, "url", None) or (item.get("url") if isinstance(item, dict) else None) or "?"
        content = (
            getattr(item, "content", None)
            or (item.get("content") if isinstance(item, dict) else None)
            or ""
        )
        parsed.append({"title": title, "url": url, "content": content})

    evidence_text = "\n\n".join(
        f"[{p['title']}]({p['url']})\n{p['content'][:600]}" for p in parsed
    )
    return evidence_text, parsed


def probe_question(model: str, q: dict, top_k: int) -> dict:
    query = q["question_text"]

    try:
        raw_search = ollama.web_search(query)
    except Exception as exc:  # noqa: BLE001 -- surface the real error,
        # don't let it kill the rest of the batch
        return {
            "question_id": q["question_id"],
            "search_error": str(exc),
            "raw_text": None,
        }

    evidence_text, parsed_results = extract_search_results(raw_search)

    prompt = PROMPT_TEMPLATE.format(
        evidence=evidence_text,
        question_text=q["question_text"],
        options=format_options(q["options"]),
        answer=q.get("verified_answer") or q.get("source_answer"),
    )

    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])

    return {
        "question_id": q["question_id"],
        "search_result_count": len(parsed_results),
        "search_results": parsed_results,
        "raw_text": response["message"]["content"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe Ollama's native web_search API for correctness-"
        "checking. Throwaway diagnostic, not a pipeline stage."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument(
        "--model",
        default="gemma3:latest",
        help="Local model for the reasoning step -- must already be pulled "
        "(ollama pull <name>). Defaults to gemma3:latest, cited as tested-compatible "
        "with this feature; override if you already have a different model.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Search results to use per question.")
    parser.add_argument("--sleep", type=float, default=1.0)
    args = parser.parse_args()

    if not os.environ.get("OLLAMA_API_KEY"):
        print(
            "Set OLLAMA_API_KEY first (free key from "
            "ollama.com/settings/keys) -- web_search is a hosted call and "
            "needs it even though the model itself runs locally."
        )
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
        result = probe_question(args.model, q, args.top_k)
        results.append(result)

        if result.get("search_error"):
            print(f"  [search error] {result['search_error']}")
        else:
            print(f"  -> {result['search_result_count']} search result(s)")
            print(f"     {result['raw_text']}")

        if args.sleep:
            time.sleep(args.sleep)

    print("\n--- Ollama web_search probe summary ---")
    print(f"model: {args.model}")
    print(f"questions probed: {len(results)}")
    errors = sum(1 for r in results if r.get("search_error"))
    print(f"search errors: {errors}")
    print(
        "\nCheck the saved file for each question's search_results -- if "
        "content fields are real, relevant snippets, this is genuinely "
        "usable. If they're empty or clearly irrelevant, the retrieval "
        "step isn't working even if no error was raised."
    )

    out_path = args.input.parent / "_ollama_websearch_probe_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()