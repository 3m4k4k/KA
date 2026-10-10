"""
Throwaway probe: quick sanity test of Groq (free tier) as the engine
for the answer-correctness / explanation stages.

Like the Gemini probe it replaces, this is deliberately NOT a pipeline
stage -- it doesn't write to any QuestionRecord, doesn't touch
reviewed/ files, and isn't meant to be kept around. It answers three
questions before the real stage gets built:

  1. Does the key work, and how fast/token-hungry is a call?
  2. On the real answer keys, how often does the model agree?
     (Disagreements are what you'd send to the review UI -- they're
     either real key errors or model errors, and you want to eyeball
     which.)
  3. Does it actually CATCH a wrong key? --corrupt re-runs each
     question with a deliberately wrong answer injected. If the model
     still "agrees" with a wrong key, it can't be trusted as a checker.

No web search -- this is first-year textbook content, so the model is
judged on its own knowledge. No new dependencies: uses `requests`,
which the pipeline already needs.

Requires:
    export GROQ_API_KEY=...        (console.groq.com/keys)

Usage:
    python test_groq.py --input ../reviewed/DOC-56eb58bdf252.questions.json --count 10
    python test_groq.py --input ../reviewed/DOC-56eb58bdf252.questions.json --count 10 --corrupt
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
from pathlib import Path

import requests

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PROMPT = (
    "You are checking ONE first-year medical school exam question "
    "(anatomy, histology, physiology, biochemistry). Judge using your own "
    "textbook knowledge.\n\n"
    "Respond with ONLY a JSON object, no markdown fences, in exactly this "
    "shape:\n"
    '{"agrees": <true/false>, "suggested_answer": <option label>, '
    '"explanation": <1-2 sentences on why the correct option is right>, '
    '"why_others_wrong": {<label>: <short reason>, ...}, '
    '"concerns": [<short strings, e.g. "garbled_option", '
    '"stem_truncated", "ambiguous_question">]}\n\n'
    "'agrees' means: is the GIVEN answer the correct option? Do not trust "
    "the given answer -- decide independently first. 'why_others_wrong' "
    "covers every option except the correct one. If the question text or "
    "options look garbled or cut off by OCR, say so in 'concerns' instead "
    "of guessing."
)

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.S)


def format_options(options: list[dict]) -> str:
    return "\n".join(f"  {o['label']}) {o['text']}" for o in options)


def effective_fields(q: dict) -> tuple[str, list[dict], str | None]:
    """Prefer human-corrected fields from the review UI when present."""
    text = q.get("verified_question_text") or q["question_text"]
    options = q.get("verified_options") or q["options"]
    answer = q.get("verified_answer") or q.get("source_answer")
    return text, options, answer


def wrong_answer_for(options: list[dict], real_answer: str) -> str | None:
    """Deterministic wrong label: the next option after the real one."""
    labels = [o["label"] for o in options]
    if real_answer not in labels or len(labels) < 2:
        return None
    return labels[(labels.index(real_answer) + 1) % len(labels)]


def call_groq(
    api_key: str,
    model: str,
    user_prompt: str,
    reasoning_effort: str | None,
    max_retries: int = 3,
) -> dict:
    """Returns {text, latency_s, total_tokens, retries_429, error}."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_completion_tokens": 2000,
    }
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort

    headers = {"Authorization": f"Bearer {api_key}"}
    retries_429 = 0
    start = time.time()

    for _attempt in range(max_retries + 1):
        try:
            resp = requests.post(
                GROQ_CHAT_URL, json=payload, headers=headers, timeout=90
            )
        except requests.RequestException as exc:  # noqa: BLE001
            return {"text": None, "error": f"network: {exc}", "retries_429": retries_429,
                    "latency_s": time.time() - start, "total_tokens": None}

        if resp.status_code == 429:
            retries_429 += 1
            wait = float(resp.headers.get("retry-after", "10"))
            print(f"    [429] rate limited, waiting {wait:.0f}s "
                  f"(retry {retries_429}/{max_retries})")
            time.sleep(min(wait, 60))
            continue

        if resp.status_code != 200:
            return {"text": None, "error": f"HTTP {resp.status_code}: {resp.text[:300]}",
                    "retries_429": retries_429, "latency_s": time.time() - start,
                    "total_tokens": None}

        body = resp.json()
        text = body["choices"][0]["message"].get("content") or ""
        usage = body.get("usage") or {}
        return {"text": text, "error": None, "retries_429": retries_429,
                "latency_s": time.time() - start,
                "total_tokens": usage.get("total_tokens")}

    return {"text": None, "error": "gave up after repeated 429s",
            "retries_429": retries_429, "latency_s": time.time() - start,
            "total_tokens": None}


def parse_response(raw: str | None) -> dict | None:
    if not raw:
        return None
    m = _JSON_BLOCK_RE.search(raw)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if "agrees" not in data:
        return None
    return data


def probe(api_key, model, q, given_answer, reasoning_effort) -> dict:
    text, options, _ = effective_fields(q)
    prompt = (
        f"Question: {text}\n"
        f"Options:\n{format_options(options)}\n"
        f"Given answer: {given_answer}"
    )
    call = call_groq(api_key, model, prompt, reasoning_effort)
    parsed = parse_response(call["text"])
    return {
        "question_id": q["question_id"],
        "given_answer": given_answer,
        "agrees": parsed.get("agrees") if parsed else None,
        "suggested_answer": parsed.get("suggested_answer") if parsed else None,
        "explanation": parsed.get("explanation") if parsed else None,
        "why_others_wrong": parsed.get("why_others_wrong") if parsed else None,
        "concerns": parsed.get("concerns") if parsed else None,
        "parsed_ok": parsed is not None,
        "error": call["error"],
        "latency_s": round(call["latency_s"], 2),
        "total_tokens": call["total_tokens"],
        "retries_429": call["retries_429"],
        "raw_text": call["text"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe Groq as the answer-check engine. Throwaway diagnostic, "
        "not a pipeline stage."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument(
        "--corrupt",
        action="store_true",
        help="Also re-run each question with a deliberately WRONG answer "
        "injected, to test whether the model catches bad keys.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default=None,
        help="Only for gpt-oss models. Lower = fewer tokens/faster. Omit for "
        "other models (an unsupported param returns HTTP 400).",
    )
    parser.add_argument("--sleep", type=float, default=2.0)
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Set GROQ_API_KEY first (console.groq.com/keys).")
        return

    doc = json.loads(args.input.read_text())
    questions = [
        q for q in doc["questions"]
        if effective_fields(q)[2] and (q.get("verified_options") or q.get("options"))
    ][: args.count]
    if not questions:
        print(f"No answerable questions found in {args.input}.")
        return

    real_runs: list[dict] = []
    corrupt_runs: list[dict] = []

    for i, q in enumerate(questions, start=1):
        _, options, answer = effective_fields(q)
        print(f"[{i}/{len(questions)}] {q['question_id']} (key={answer})")

        r = probe(api_key, args.model, q, answer, args.reasoning_effort)
        real_runs.append(r)
        if r["error"]:
            print(f"  real key   -> ERROR {r['error']}")
        else:
            flag = "" if r["agrees"] else "   <-- DISAGREES, check this one"
            print(f"  real key   -> agrees={r['agrees']} suggested={r['suggested_answer']} "
                  f"{r['latency_s']}s {r['total_tokens']} tok{flag}")
        time.sleep(args.sleep)

        if args.corrupt:
            wrong = wrong_answer_for(options, answer)
            if wrong is None:
                continue
            c = probe(api_key, args.model, q, wrong, args.reasoning_effort)
            corrupt_runs.append(c)
            if c["error"]:
                print(f"  wrong key  -> ERROR {c['error']}")
            else:
                caught = c["agrees"] is False
                print(f"  wrong key({wrong}) -> agrees={c['agrees']} "
                      f"{'CAUGHT' if caught else 'MISSED  <-- bad'}")
            time.sleep(args.sleep)

    def summarize(label: str, runs: list[dict]) -> None:
        ok = [r for r in runs if r["parsed_ok"]]
        lat = [r["latency_s"] for r in runs if not r["error"]]
        toks = [r["total_tokens"] for r in runs if r["total_tokens"]]
        print(f"\n[{label}] calls={len(runs)} parsed_ok={len(ok)} "
              f"errors={sum(1 for r in runs if r['error'])} "
              f"429_retries={sum(r['retries_429'] for r in runs)}")
        if lat:
            print(f"  avg latency {statistics.mean(lat):.1f}s")
        if toks:
            print(f"  avg tokens/call {statistics.mean(toks):.0f}")
        return ok

    print("\n--- Groq probe summary ---")
    print(f"model: {args.model}")
    ok = summarize("real keys", real_runs)
    if ok:
        agree = sum(1 for r in ok if r["agrees"])
        print(f"  agrees with the key on {agree}/{len(ok)}")
    if corrupt_runs:
        okc = summarize("wrong keys (injected)", corrupt_runs)
        if okc:
            caught = sum(1 for r in okc if r["agrees"] is False)
            print(f"  caught {caught}/{len(okc)} deliberately wrong keys")
            print("  -> if this isn't near 100%, don't use the model as a checker.")

    out_path = args.input.parent / "_groq_probe_results.json"
    out_path.write_text(
        json.dumps({"real": real_runs, "corrupt": corrupt_runs}, indent=2, default=str)
    )
    print(f"\nFull results (incl. explanations) written to {out_path}")


if __name__ == "__main__":
    main()