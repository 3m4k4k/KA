"""
Validation of an extraction -- independent of whoever produced it.

The same checks run on the builtin parser's output, on a stored profile's
output, and on an AI-discovered profile's output. A layout is only trusted
because its *output* looks right when compared with the *source lines*, never
because a model (or an earlier run) said so.

Verdicts:
    pass  -- every check passed; safe to accept without AI / further effort
    weak  -- nothing clearly broken but something is off (ambiguous)
    fail  -- the layout was misread

Record-level anomalies (a duplicated option, a missing key entry) are NOT
layout failures by themselves -- they are the pipeline's normal "flag it for
review" signal. Only a high *rate* of them says the layout was misread.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from question_schemas import QuestionDocument, QuestionType  # noqa: E402
from layout_features import (  # noqa: E402
    Line,
    LayoutFeatures,
    content_lines,
    is_inline_answer_line,
    is_key_header,
    is_key_like_line,
)
from layout_spec import normalize_literal  # noqa: E402

PASS, WEAK, FAIL = "pass", "weak", "fail"


@dataclass
class Thresholds:
    min_questions: int = 3
    coverage_pass: float = 0.93
    coverage_fail: float = 0.80
    sequence_pass: float = 0.90
    sequence_fail: float = 0.55
    option_shape_pass: float = 0.90
    option_shape_fail: float = 0.65
    option_count_pass: float = 0.90
    option_count_fail: float = 0.75
    key_cov_pass: float = 0.85
    key_cov_fail: float = 0.60
    key_resolve_fail: float = 0.30  # unmatched key entries / key entries
    key_resolve_warn: float = 0.10
    key_option_pass: float = 0.95
    key_option_fail: float = 0.80
    anomaly_warn: float = 0.25
    anomaly_fail: float = 0.50
    empty_text_fail: float = 0.10
    furniture_warn: float = 0.25
    furniture_fail: float = 0.40
    ignore_anomaly_prefixes: tuple = ("duplicate_question_merged",)


@dataclass
class Check:
    name: str
    status: str  # "pass" | "warn" | "fail"
    value: float | None
    detail: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "status": self.status, "value": None if self.value is None else round(self.value, 3), "detail": self.detail}


@dataclass
class ValidationReport:
    verdict: str
    score: float
    checks: list[Check] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.status == "fail"]

    @property
    def warned(self) -> list[Check]:
        return [c for c in self.checks if c.status == "warn"]

    def summary(self) -> str:
        bad = self.failed + self.warned
        if not bad:
            return "all checks passed"
        return "; ".join(f"{c.name}={c.value if c.value is None else round(c.value, 2)} ({c.status}) {c.detail}".strip() for c in bad)

    def to_dicts(self) -> list[dict]:
        return [c.to_dict() for c in self.checks]


def _grade(value: float, ok: float, bad: float, higher_is_better: bool = True) -> str:
    if higher_is_better:
        return "pass" if value >= ok else ("fail" if value < bad else "warn")
    return "pass" if value <= ok else ("fail" if value > bad else "warn")


# --- coverage ---------------------------------------------------------------

_GENERIC_HEADERS = re.compile(
    r"^(m\.?\s*c\.?\s*q'?s?|fill in the blanks?|(cross[\s-]*)?matching|(model )?answers?|answer key|key)$", re.I
)
_LEAD_MARK_RE = re.compile(r"^\(?\[?(?:q(?:uestion)?\s*)?[0-9a-h]{1,3}\s*[-.):\u00b7\]]+\s*", re.I)
_MARK_RE = re.compile(r"[*\u2713\u2714]|\(correct\)", re.I)


def _cov_norm(s: str) -> str:
    s = _MARK_RE.sub(" ", s.lower())
    return re.sub(r"[\W_]+", " ", s).strip()


def content_coverage(qdoc: QuestionDocument, body: list[Line]) -> tuple[float, int]:
    """Fraction (by characters) of the document's content lines whose text
    shows up in the extracted records. Key-entry lines, inline answer lines,
    section headers and chapter headings are excluded -- they are structure,
    not content. A parser that drops or mangles a third of the document
    cannot hide behind a high question count."""
    parts = []
    labels = set()
    for q in qdoc.questions:
        parts.append(q.question_text)
        parts.extend(o.text for o in q.options)
        for mp in q.matching_pairs:
            parts.extend([mp.left_text, mp.right_text])
        if q.source_answer:
            parts.append(q.source_answer)
        if q.group_label:
            labels.add(_cov_norm(q.group_label))
    blob = " " + " ".join(_cov_norm(p) for p in parts) + " "
    covered = total = 0
    for ln in body:
        t = ln.text
        if is_key_like_line(t) or is_inline_answer_line(t) or is_key_header(t) or _GENERIC_HEADERS.match(normalize_literal(t)):
            continue
        n = _cov_norm(_LEAD_MARK_RE.sub("", t))
        if not n or n in labels:
            continue
        total += len(n)
        if n in blob:
            covered += len(n)
    return (covered / total if total else 1.0), total


# --- helpers ----------------------------------------------------------------


def _actionable(anomaly: str, th: Thresholds) -> bool:
    return not anomaly.startswith(th.ignore_anomaly_prefixes)


def validate_extraction(
    qdoc: QuestionDocument,
    lines: list[Line],
    feats: LayoutFeatures,
    th: Thresholds | None = None,
) -> ValidationReport:
    th = th or Thresholds()
    body = content_lines(lines)
    qs = qdoc.questions
    checks: list[Check] = []
    metrics: dict = {"questions": len(qs)}

    # 1. enough questions
    st = "pass" if len(qs) >= th.min_questions else "fail"
    checks.append(Check("has_questions", st, float(len(qs)), f"need >= {th.min_questions}"))

    if not qs:
        return ValidationReport(FAIL, 0.0, checks, metrics)

    # 1b. furniture sanity: if a large share of the document was stripped as
    # running header/footer, content is probably being thrown away unseen.
    if lines:
        fr = sum(1 for ln in lines if ln.furniture) / len(lines)
        checks.append(Check("furniture_share", _grade(fr, th.furniture_warn, th.furniture_fail, False), fr, "lines stripped as running header/footer"))

    # 2. content coverage (output vs source)
    cov, eligible = content_coverage(qdoc, body)
    metrics["coverage_chars"] = eligible
    checks.append(Check("content_coverage", _grade(cov, th.coverage_pass, th.coverage_fail), cov, "source text found in records"))

    # 3. numbering density per (type, group)
    present = rng = dup = 0
    groups: dict[tuple, list[int]] = {}
    for q in qs:
        if q.question_number.isdigit():
            groups.setdefault((q.section_type, q.group_label), []).append(int(q.question_number))
    for nums in groups.values():
        s = set(nums)
        present += len(s)
        rng += max(s) - min(s) + 1
        dup += len(nums) - len(s)
    seq = (present / rng) if rng else 0.0
    if dup:
        seq = max(0.0, seq - dup / max(len(qs), 1))
    checks.append(Check("numbering_sequence", _grade(seq, th.sequence_pass, th.sequence_fail), seq, "contiguous question numbers"))

    # 4. MCQ option shape
    mcqs = [q for q in qs if q.section_type == QuestionType.MCQ]
    if mcqs:
        ok = 0
        for q in mcqs:
            lbls = [o.label.lower() for o in q.options]
            if 2 <= len(lbls) <= 6 and lbls == [chr(ord("a") + i) for i in range(len(lbls))]:
                ok += 1
        shape = ok / len(mcqs)
        checks.append(Check("option_shape", _grade(shape, th.option_shape_pass, th.option_shape_fail), shape, "2-6 options labelled a,b,c.."))
        # extracted options per MCQ vs what the source's option-looking lines
        # say: catches an option marker that swallows a, b, c, d into one.
        if feats.opts_per_stem >= 2:
            mean_opts = sum(len(q.options) for q in mcqs) / len(mcqs)
            ratio = mean_opts / feats.opts_per_stem
            checks.append(Check("option_count_consistency", _grade(ratio, th.option_count_pass, th.option_count_fail), ratio, "extracted options per MCQ vs option-like lines in source"))
        empty = sum(1 for q in mcqs if not q.question_text.strip()) / len(mcqs)
        checks.append(Check("empty_stems", "fail" if empty > th.empty_text_fail else "pass", empty, "MCQs with empty question text"))

    # 5. keys
    keyed_types = [q for q in qs if q.section_type in (QuestionType.MCQ, QuestionType.FILL_IN_BLANK, QuestionType.MATCHING)]
    answered = sum(1 for q in keyed_types if q.source_answer)
    key_cov = answered / len(keyed_types) if keyed_types else 1.0
    n_q = max(len(qs), 1)
    key_expected = (
        feats.key_header_lines > 0
        or feats.key_tokens >= max(3, 0.3 * n_q)
        or feats.inline_answer_lines >= max(3, 0.3 * n_q)
        or feats.marked_option_lines >= max(3, 0.3 * n_q)
    )
    metrics.update(key_coverage=round(key_cov, 3), key_expected=key_expected)
    if key_expected:
        checks.append(Check("key_coverage", _grade(key_cov, th.key_cov_pass, th.key_cov_fail), key_cov, "questions with a source_answer (key present in source)"))
    else:
        checks.append(Check("key_coverage", "pass", key_cov, "no key evidence in source; informational"))

    if key_expected:
        unanswered = len(keyed_types) - answered
        explained = len(qdoc.unmatched_answer_key_entries) + sum(1 for q in keyed_types if not q.source_answer and any(a.startswith("no_answer_key_entry") for a in q.anomalies))
        unexplained = max(0, unanswered - explained)
        checks.append(Check("key_gaps_explained", "warn" if unexplained else "pass", float(unexplained), "unanswered questions with no flag and no unmatched key entry (silent key loss)"))

    n_keys = len(qdoc.unmatched_answer_key_entries) + answered
    if n_keys:
        unres = len(qdoc.unmatched_answer_key_entries) / n_keys
        checks.append(Check("key_resolution", _grade(unres, th.key_resolve_warn, th.key_resolve_fail, False), unres, "key entries matching no question"))

    mcq_answered = [q for q in mcqs if q.source_answer and q.options]
    if mcq_answered:
        good = 0
        for q in mcq_answered:
            labels = {o.label.lower() for o in q.options}
            parts = [p for p in re.split(r"\s*[&,]\s*", q.source_answer.lower()) if p]
            good += all(p in labels for p in parts)
        r = good / len(mcq_answered)
        checks.append(Check("key_matches_options", _grade(r, th.key_option_pass, th.key_option_fail), r, "answers naming a real option"))

    # 6. anomaly rate
    flagged = sum(1 for q in qs if any(_actionable(a, th) for a in q.anomalies))
    rate = flagged / len(qs)
    checks.append(Check("anomaly_rate", _grade(rate, th.anomaly_warn, th.anomaly_fail, False), rate, "records carrying actionable anomalies"))

    # 7. provenance integrity (engine output only)
    refd = [q for q in qs if q.extraction and q.extraction.first_line_ref]
    if refd:
        index = {ln.ref: i for i, ln in enumerate(lines)}
        bad = 0
        for q in refd:
            a, b = index.get(q.extraction.first_line_ref), index.get(q.extraction.last_line_ref or q.extraction.first_line_ref)
            if a is None or b is None or a > b:
                bad += 1
        checks.append(Check("provenance_integrity", "fail" if bad else "pass", float(bad), "records with unresolvable/reversed line refs"))

    # verdict + score
    per = {"pass": 1.0, "warn": 0.5, "fail": 0.0}
    score = sum(per[c.status] for c in checks) / len(checks)
    if any(c.status == "fail" for c in checks):
        verdict = FAIL
    elif any(c.status == "warn" for c in checks):
        verdict = WEAK
    else:
        verdict = PASS
    return ValidationReport(verdict, score, checks, metrics)


def rank_key(verdict: str, score: float) -> tuple:
    return ({PASS: 2, WEAK: 1, FAIL: 0}[verdict], round(score, 3))
