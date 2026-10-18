"""
AI layout discovery.

Asks a model ONE question about an unfamiliar document: "what does the
layout look like?" -- answered as a LayoutSpec (layout_spec.py). The model
never sees the whole document and never writes question content.

Guard rails, in order:
  1. The reply must parse into the strict LayoutSpec schema (unknown fields
     rejected, literals constrained, markers limited to the fixed grammar).
  2. Grounding: every evidence ref must exist in the real line stream, must
     not be a stripped header/footer line, and must actually match the rule
     it is cited for. Every section-header literal must exist in the document.
     A model that hallucinates a line, a ref or a header is rejected here,
     before any extraction runs.
  3. Whatever survives is executed by the deterministic engine and judged by
     layout_validation.py. The AI cannot bypass that.

Transport: the project's existing providers. `GroqClient` wraps
groq_router.GroqRouter (rate-limit aware, model fallback); `OllamaClient`
posts to the local Ollama chat API like ai_parser.py. No new dependencies.
Unit tests inject a scripted client instead.

Prompt size is bounded (default ~8k chars of excerpts) so a request fits
Groq free-tier per-minute token limits.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

sys.path.insert(0, str(Path(__file__).resolve().parent))

from layout_spec import CompiledSpec, LayoutSpec, compile_spec, normalize_literal  # noqa: E402
from layout_features import (  # noqa: E402
    LayoutFeatures,
    Line,
    content_lines,
    is_key_header,
    is_key_like_line,
    parse_ref,
)

PROMPT_VERSION = "1"


class AIUnavailable(Exception):
    """No key / budget exhausted / provider down. The batch continues
    without AI; affected documents are flagged, never silently dropped."""


@dataclass
class LLMReply:
    text: str
    model: str


class LLMClient(Protocol):
    def complete(self, system: str, user: str, *, max_tokens: int) -> LLMReply: ...


class GroqClient:
    def __init__(self, router):
        self.router = router

    def complete(self, system: str, user: str, *, max_tokens: int = 1800) -> LLMReply:
        from groq_router import AllModelsExhausted, FatalAPIError, RouterError

        try:
            res = self.router.chat(system, user, max_completion_tokens=max_tokens, est_completion_tokens=max_tokens // 2)
        except AllModelsExhausted as exc:
            raise AIUnavailable(f"groq budget exhausted (resume in {exc.resume_in_s / 60:.0f} min)") from exc
        except (FatalAPIError, RouterError) as exc:
            raise AIUnavailable(f"groq error: {exc}") from exc
        return LLMReply(res.text, f"groq:{res.model}")


class OllamaClient:
    def __init__(self, model: str = "phi4-mini", num_ctx: int = 8192, url: str = "http://localhost:11434/api/chat"):
        self.model, self.num_ctx, self.url = model, num_ctx, url

    def complete(self, system: str, user: str, *, max_tokens: int = 1800) -> LLMReply:
        import requests

        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "options": {"num_ctx": self.num_ctx, "temperature": 0.1, "num_predict": max_tokens},
        }
        try:
            resp = requests.post(self.url, json=payload, timeout=300)
        except requests.RequestException as exc:
            raise AIUnavailable(f"ollama unreachable: {exc}") from exc
        if resp.status_code != 200:
            raise AIUnavailable(f"ollama HTTP {resp.status_code}: {resp.text[:200]}")
        return LLMReply((resp.json().get("message") or {}).get("content", ""), f"ollama:{self.model}")


# ---------------------------------------------------------------------------
# Digest
# ---------------------------------------------------------------------------


def _fmt(ln: Line, width: int = 150) -> str:
    t = ln.text if len(ln.text) <= width else ln.text[: width - 1] + "\u2026"
    return f"{ln.ref}| {t}"


def build_digest(lines: list[Line], feats: LayoutFeatures, budget_chars: int = 8000) -> str:
    """Deterministic, bounded view of the document: computed statistics plus
    excerpts around structural landmarks. Every excerpt line carries its real
    ref so the model can cite it."""
    body = content_lines(lines)
    pages = sorted({ln.page for ln in body})
    stats = [
        f"pages={feats.n_pages} content_lines={feats.n_lines}",
        f"dominant stem style (prefix|open|closer)={feats.stem_style or 'none'} lines={feats.stem_count}",
        f"dominant option style (case|open|closer)={feats.opt_style or 'none'} lines={feats.opt_count}",
        f"numbering restarts at 1: {feats.restarts}x",
        f"key-like entry lines={feats.key_like_lines} (tokens={feats.key_tokens}); key-header-like lines={feats.key_header_lines}",
        f"inline 'Answer: x' lines={feats.inline_answer_lines}; options carrying a mark={feats.marked_option_lines} {feats.marked_option_symbol!r}",
        f"distinct heading-like lines={feats.heading_like_distinct}",
    ]
    if feats.header_candidates:
        stats.append("recurring/short header candidates (normalised, count, first ref): " + "; ".join(f"{n!r} x{c} {r}" for n, c, r in feats.header_candidates))

    # landmark windows (indices into body)
    idx = {ln.ref: i for i, ln in enumerate(body)}
    marks: list[tuple[int, int, int]] = []  # (centre, before, after)
    marks.append((0, 0, 24))
    for n, _c, ref in feats.header_candidates[:8]:
        if ref in idx:
            marks.append((idx[ref], 2, 8))
    for i, ln in enumerate(body):
        if is_key_header(ln.text):
            marks.append((i, 2, 14))
    first_key = next((i for i, ln in enumerate(body) if is_key_like_line(ln.text)), None)
    if first_key is not None:
        marks.append((first_key, 3, 10))
    if pages:
        last_page_start = next((i for i, ln in enumerate(body) if ln.page == pages[-1]), len(body) - 1)
        marks.append((last_page_start, 0, 14))
    if pages and len(pages) > 2:
        mid = next((i for i, ln in enumerate(body) if ln.page == pages[len(pages) // 2]), 0)
        marks.append((mid, 0, 16))

    spans = sorted((max(c - b, 0), min(c + a + 1, len(body))) for c, b, a in marks)
    merged: list[list[int]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    out_parts = ["DOCUMENT STATISTICS (computed by code):", *stats, "", "EXCERPTS (ref| text; '...' = skipped lines):"]
    used = sum(len(x) + 1 for x in out_parts)
    for s, e in merged:
        chunk = [_fmt(body[i]) for i in range(s, e)]
        size = sum(len(x) + 1 for x in chunk) + 4
        if used + size > budget_chars:
            # trim this chunk to fit, then stop
            room = budget_chars - used - 4
            keep = []
            for x in chunk:
                if room - len(x) - 1 < 0:
                    break
                keep.append(x)
                room -= len(x) + 1
            if keep:
                out_parts.extend(keep + ["..."])
            break
        out_parts.extend(chunk + ["..."])
        used += size
    return "\n".join(out_parts)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You analyse the LAYOUT of ONE exam question-bank document so a deterministic "
    "parser can extract its questions. You never extract, copy or rewrite questions. "
    "You describe layout rules in the JSON schema below and nothing else.\n\n"
    "Every line in the excerpts is shown as `ref| text` (e.g. `p3:14| 12- What is...`). "
    "Copy literal strings EXACTLY as they appear. Every rule needs `evidence`: refs of "
    "lines from the excerpts that show it. Never invent refs or lines.\n\n"
    "Reply with ONLY one JSON object, no markdown fences:\n"
    '{"expressible": true,\n'
    ' "question_marker": {"prefix": "", "open": "", "closers": "-.)", "evidence": ["p2:5","p2:11"]},\n'
    ' "option_marker": {"case": "lower", "open": "", "closers": ".)", "max_label": "d", "evidence": ["p2:6"]},\n'
    ' "sections": [{"header": "<exact line text>", "kind": "mcq|fill_in_blank|matching|answer_key", "evidence": ["p1:3"]}],\n'
    ' "default_content_kind": "mcq|fill_in_blank|matching|null",\n'
    ' "grouping": {"mode": "none|heading_restart"},\n'
    ' "answer_key": {"mode": "none|section|inline_label|marked_option", "separators": "-.", "allow_space": false, '
    '"grouped": false, "inline_prefixes": [], "option_mark": "", "option_mark_position": "suffix", "evidence": []},\n'
    ' "notes": ""}\n\n'
    "Rules:\n"
    "- question_marker: how a question stem line starts: optional literal `prefix` (e.g. \"Q\", \"Question\"), optional `open` "
    "(\"(\" or \"[\"), the NUMBER, then one or more `closers` chars from - . ) : ] (list every closer style used).\n"
    "- option_marker: how an option line starts: a letter (`case` lower/upper/either), optional `open` \"(\", then `closers`; "
    "`max_label` = highest letter used (a-h).\n"
    "- sections: only header LINES that really exist (kind says what follows: mcq / fill_in_blank / matching, or answer_key "
    "for an answer-key block). Omit headers that do not exist. If there are no content headers, set default_content_kind.\n"
    "- grouping heading_restart: numbering restarts at 1 under chapter/topic heading lines. Otherwise none.\n"
    "- answer_key.mode: `section` = a key block after a header (entries like `12-b`, `12. b`, `12 b`; list `separators`, set "
    "allow_space if entries are separated by spaces only; grouped=true if the block repeats the chapter headings); "
    "`inline_label` = a line after each question such as `Answer: b` (give the prefix words); `marked_option` = the correct "
    "option itself carries a literal mark such as `*` (give option_mark and suffix/prefix); `none` = no key anywhere.\n"
    "- If the layout cannot be described (options on one line, two-column pages, answers elsewhere, questions without numbers) "
    'reply {"expressible": false, "notes": "<why>"}.'
)


def build_user_prompt(digest: str, feedback: list[str] | None = None, previous: dict | None = None) -> str:
    parts = [digest]
    if feedback:
        parts.append("\nYOUR PREVIOUS ANSWER WAS REJECTED. Problems found by code:")
        parts.extend(f"- {f}" for f in feedback[:12])
        if previous is not None:
            parts.append("Previous answer: " + json.dumps(previous, ensure_ascii=False)[:1500])
        parts.append("Fix these problems and reply with the full corrected JSON object.")
    else:
        parts.append("\nDescribe this document's layout as the JSON object.")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Parsing + grounding
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.S)
_FENCE_RE = re.compile(r"```(?:json)?|```")
_JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_reply(raw: str | None) -> tuple[dict | None, list[str]]:
    if not raw:
        return None, ["empty reply"]
    txt = _FENCE_RE.sub("", _THINK_RE.sub("", raw))
    m = _JSON_RE.search(txt)
    if not m:
        return None, ["reply contained no JSON object"]
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        return None, [f"reply was not valid JSON ({exc.msg})"]
    if not isinstance(data, dict):
        return None, ["reply JSON was not an object"]
    return data, []


def parse_spec(data: dict) -> tuple[LayoutSpec | None, list[str]]:
    from pydantic import ValidationError

    try:
        return LayoutSpec(**data), []
    except ValidationError as exc:
        errs = []
        for e in exc.errors()[:8]:
            loc = ".".join(str(x) for x in e["loc"])
            errs.append(f"schema: {loc}: {e['msg']}")
        return None, errs
    except TypeError as exc:
        return None, [f"schema: {exc}"]


def _fuzzy_header_present(norm: str, body: list[Line]) -> bool:
    import difflib

    for ln in body:
        n = normalize_literal(ln.text)
        if n == norm:
            return True
    for ln in body:
        n = normalize_literal(ln.text)
        if len(norm) >= 4 and abs(len(n) - len(norm)) <= 3 and difflib.SequenceMatcher(None, n, norm).ratio() >= 0.9:
            return True
    return False


def ground_spec(spec: LayoutSpec, lines: list[Line], cspec: CompiledSpec) -> list[str]:
    """Hallucination tripwires. Returns human-readable problems (fed back to
    the model on a retry). Empty list = grounded."""
    problems: list[str] = []
    by_ref = {ln.ref: ln for ln in lines}
    body = content_lines(lines)

    def check_refs(label: str, refs: list[str], predicate, required: bool):
        if required and not refs:
            problems.append(f"{label}: no evidence refs given")
            return
        ok = 0
        for r in refs:
            if parse_ref(r) is None:
                problems.append(f"{label}: {r!r} is not a ref like p3:14")
                continue
            ln = by_ref.get(r.strip())
            if ln is None:
                problems.append(f"{label}: ref {r} does not exist in the document")
            elif ln.furniture:
                problems.append(f"{label}: ref {r} is a running header/footer, not content")
            elif not predicate(ln):
                problems.append(f"{label}: ref {r} (`{ln.text[:50]}`) does not match the rule you described")
            else:
                ok += 1
        if required and refs and ok == 0:
            problems.append(f"{label}: none of the cited lines match the rule")

    if spec.question_marker:
        check_refs("question_marker", spec.question_marker.evidence, lambda ln: bool(cspec.stem_re.match(ln.text)), True)
    if spec.option_marker and cspec.option_re is not None:
        check_refs("option_marker", spec.option_marker.evidence, lambda ln: bool(cspec.option_re.match(ln.text)), True)
    for s in spec.sections:
        n = normalize_literal(s.header)
        check_refs(f"section {s.header!r}", s.evidence, lambda ln, n=n: normalize_literal(ln.text) == n, True)
        if not _fuzzy_header_present(n, body):
            problems.append(f"section {s.header!r}: no such header line exists in the document")
    ak = spec.answer_key
    if ak.mode == "section" and cspec.key_token_re is not None:
        check_refs("answer_key", ak.evidence, lambda ln: bool(cspec.key_token_re.search(ln.text)) or bool(is_key_header(ln.text)) or is_key_like_line(ln.text), True)
    elif ak.mode == "inline_label" and cspec.inline_re is not None:
        check_refs("answer_key", ak.evidence, lambda ln: bool(cspec.inline_re.match(ln.text)), True)
    elif ak.mode == "marked_option" and cspec.mark:
        mk = cspec.mark
        check_refs("answer_key", ak.evidence, lambda ln: ln.text.rstrip().endswith(mk) or ln.text.lstrip().startswith(mk), True)
    return problems


@dataclass
class ReplyEvaluation:
    spec: LayoutSpec | None = None
    cspec: CompiledSpec | None = None
    problems: list[str] = field(default_factory=list)
    unexpressible: bool = False
    raw_data: dict | None = None


def evaluate_reply(raw: str, lines: list[Line]) -> ReplyEvaluation:
    """Parse -> schema -> compile -> ground. Extraction/validation happen
    in the orchestrator."""
    data, errs = parse_reply(raw)
    if data is None:
        return ReplyEvaluation(problems=errs)
    if data.get("expressible") is False:
        return ReplyEvaluation(unexpressible=True, raw_data=data, problems=[f"model says layout is not expressible: {data.get('notes', '')}"])
    spec, errs = parse_spec(data)
    if spec is None:
        return ReplyEvaluation(problems=errs, raw_data=data)
    try:
        cspec = compile_spec(spec)
    except ValueError as exc:
        return ReplyEvaluation(problems=[f"cannot compile spec: {exc}"], raw_data=data)
    problems = ground_spec(spec, lines, cspec)
    return ReplyEvaluation(spec=spec, cspec=cspec, problems=problems, raw_data=data)
