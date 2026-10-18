"""
Layout engine -- executes a LayoutSpec (layout_spec.py) against a
SourceDocument and produces a QuestionDocument.

This is the generic, spec-driven counterpart of segmentation.py. Same
principles, same output schema, same question-id shapes:

  - pure text-structure parsing, no AI calls here, ever
  - flag, don't repair: ambiguity becomes an anomaly, never a silent fix
  - nothing is invented: every character of question_text / option text /
    source_answer is taken from a real source line, and each record carries
    first/last line refs so a reviewer can find the original
  - source_answer is only what the key said; verified_answer stays None

What the engine does NOT do is decide whether its own output is any good.
That is layout_validation.py's job -- the engine will happily execute a
wrong spec and produce plausible-looking garbage, which is exactly why
every extraction is validated before it is trusted.

segmentation.py's text-only helpers (heading detection, duplicate merging,
group-label matching, matching-question parsing) are reused rather than
copied, so those behaviours have a single definition.
"""

from __future__ import annotations

import bisect
import difflib
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from schemas import SourceDocument  # noqa: E402
import segmentation as _seg  # noqa: E402
from question_schemas import (  # noqa: E402
    ExtractionProvenance,
    Option,
    QuestionDocument,
    QuestionRecord,
    QuestionType,
    SourceSpan,
    UnmatchedAnswerKeyEntry,
)
from layout_spec import CompiledSpec, normalize_literal  # noqa: E402
from layout_features import Line  # noqa: E402

_QT = {
    "mcq": QuestionType.MCQ,
    "fill_in_blank": QuestionType.FILL_IN_BLANK,
    "matching": QuestionType.MATCHING,
}
_FUZZY_HEADER_RATIO = 0.9


@dataclass
class _Sec:
    kind: str  # "content" | "answer_key"
    target: str | None  # content kind this section holds / the key applies to
    header_ref: str | None = None
    lines: list[Line] = field(default_factory=list)


@dataclass
class _Block:
    number: str
    group_idx: int = 0
    stem_line: Line | None = None
    lines: list[tuple[Line, str]] = field(default_factory=list)  # (source line, text to use)

    @property
    def first(self) -> Line | None:
        return self.stem_line or (self.lines[0][0] if self.lines else None)

    @property
    def last(self) -> Line | None:
        return self.lines[-1][0] if self.lines else self.stem_line


# ---------------------------------------------------------------------------
# Sectioning
# ---------------------------------------------------------------------------


def _header_kind(text: str, cspec: CompiledSpec, anomalies: list[str]) -> str | None:
    if len(text) > 80:
        return None
    n = normalize_literal(text)
    if not n:
        return None
    if n in cspec.header_kinds:
        return cspec.header_kinds[n]
    # OCR-corrupted header ("M.C.Q's" -> "M.C.Q s"): accept a near-exact match,
    # but say so -- a fuzzy hit is a fact a reviewer may want to know.
    for h, kind in cspec.header_kinds.items():
        if len(h) >= 4 and abs(len(n) - len(h)) <= 3:
            if difflib.SequenceMatcher(None, n, h).ratio() >= _FUZZY_HEADER_RATIO:
                note = f"fuzzy_section_header:{text.strip()[:40]}->{h}"
                if note not in anomalies:
                    anomalies.append(note)
                return kind
    return None


def _split_sections(lines: list[Line], cspec: CompiledSpec, anomalies: list[str]) -> list[_Sec]:
    secs: list[_Sec] = []
    cur: _Sec | None = None
    last_content: str | None = None
    for ln in lines:
        if ln.furniture:
            continue
        kind = _header_kind(ln.text, cspec, anomalies)
        if kind is not None:
            if kind == "answer_key":
                cur = _Sec("answer_key", last_content, ln.ref)
            else:
                cur = _Sec("content", kind, ln.ref)
                last_content = kind
            secs.append(cur)
            continue
        if cur is None:
            cur = _Sec("content", cspec.spec.default_content_kind)
            if cur.target:
                last_content = cur.target
            secs.append(cur)
        cur.lines.append(ln)
    return secs


# ---------------------------------------------------------------------------
# Question blocks
# ---------------------------------------------------------------------------


def _is_heading(text: str, cspec: CompiledSpec) -> bool:
    if not _seg._looks_like_heading(text):
        return False
    if cspec.stem_re.match(text):
        return False
    if cspec.option_re is not None and cspec.option_re.match(text):
        return False
    return True


def _split_blocks(lines: list[Line], cspec: CompiledSpec) -> list[_Block]:
    blocks: list[_Block] = []
    cur: _Block | None = None
    for ln in lines:
        m = cspec.stem_re.match(ln.text)
        if m:
            cur = _Block(m.group(1), stem_line=ln)
            blocks.append(cur)
            rest = m.group(2).strip()
            if rest:
                cur.lines.append((ln, rest))
            continue
        if cur is None:
            cur = _Block("0")
            blocks.append(cur)
        cur.lines.append((ln, ln.text))
    return blocks


def _pop_heading(block: _Block | None, cspec: CompiledSpec) -> str | None:
    if block is None or not block.lines:
        return None
    ln, _t = block.lines[-1]
    if ln is block.stem_line:
        return None
    if _is_heading(ln.text, cspec):
        block.lines.pop()
        return ln.text.strip()
    return None


def _split_grouped(lines: list[Line], cspec: CompiledSpec) -> tuple[list[_Block], list[str | None]]:
    """Chapter-grouped numbering: a new group starts at the first stem and
    again whenever numbering restarts at 1. The group's label is the heading
    line just before that stem (taken off the end of the previous block,
    where it would otherwise be swallowed into its last option). Labels are
    None where no heading is found -- never invented."""
    blocks: list[_Block] = []
    labels: list[str | None] = []
    cur: _Block | None = None
    gi = -1
    for ln in lines:
        m = cspec.stem_re.match(ln.text)
        if m:
            num = m.group(1)
            restart = gi == -1 or (num == "1" and cur is not None and cur.number not in ("0", "1"))
            if restart:
                labels.append(_pop_heading(cur, cspec))
                gi += 1
            cur = _Block(num, group_idx=gi, stem_line=ln)
            blocks.append(cur)
            rest = m.group(2).strip()
            if rest:
                cur.lines.append((ln, rest))
            continue
        if cur is None:
            cur = _Block("0")
            blocks.append(cur)
        cur.lines.append((ln, ln.text))
    return blocks, labels


def _take_mark(text: str, cspec: CompiledSpec) -> tuple[str, bool]:
    mk = cspec.mark
    if not mk:
        return text, False
    if cspec.mark_position == "suffix":
        t = text.rstrip()
        if t.endswith(mk):
            return t[: -len(mk)].rstrip(), True
    else:
        t = text.lstrip()
        if t.startswith(mk):
            return t[len(mk):].lstrip(), True
    return text, False


def _norm_answer(raw: str) -> str:
    return re.sub(r"\s+", " ", raw.strip().lower())


def _parse_mcq_blocks(
    blocks: list[_Block], doc_id: str, labels: list[str | None] | None, cspec: CompiledSpec
) -> list[QuestionRecord]:
    multi_group = labels is not None and len(labels) > 1
    parsed = []
    for b in blocks:
        if b.number == "0":
            continue
        text_parts: list[str] = []
        options: list[Option] = []
        marked: list[str] = []
        inline_answer = None
        inline_ref = None
        pages: set[int] = set()
        for ln, txt in b.lines:
            pages.add(ln.page)
            if cspec.inline_re is not None:
                im = cspec.inline_re.match(txt.strip())
                if im:
                    inline_answer, inline_ref = _norm_answer(im.group(1)), ln.ref
                    continue
            work, is_marked = _take_mark(txt, cspec)
            om = cspec.option_re.match(work) if cspec.option_re else None
            if om:
                options.append(Option(label=om.group(1), text=om.group(2).strip()))
                if is_marked:
                    marked.append(om.group(1))
            elif options:
                last = options[-1]
                options[-1] = Option(label=last.label, text=f"{last.text} {work}".strip())
                if is_marked:
                    marked.append(last.label)
            else:
                text_parts.append(txt)
        if b.stem_line is not None:
            pages.add(b.stem_line.page)
        parsed.append((b, text_parts, options, marked, inline_answer, inline_ref, pages))

    # the modal option count of this document is "expected"; flag deviations
    # (same rule shape as segmentation.py's 4-or-a..e, but adaptive).
    mode_counts = Counter(len(p[2]) for p in parsed if p[2])
    mode = mode_counts.most_common(1)[0][0] if mode_counts else 4

    records = []
    for b, text_parts, options, marked, inline_answer, inline_ref, pages in parsed:
        label = labels[b.group_idx] if labels else None
        if multi_group:
            qid = f"{doc_id}-mcq-{_seg._slug(label) if label else f'g{b.group_idx}'}-{b.number}"
        else:
            qid = f"{doc_id}-mcq-{b.number}"
        rec = QuestionRecord(
            question_id=qid,
            document_id=doc_id,
            section_type=QuestionType.MCQ,
            question_number=b.number,
            group_label=label,
            question_text=" ".join(text_parts).strip(),
            options=options,
            source_spans=[SourceSpan(pages=sorted(pages))],
        )
        lbls = [o.label.lower() for o in options]
        if not (len(options) == mode or lbls == list("abcde")):
            rec.anomalies.append(f"unexpected_option_count:{len(options)}")
        prov = ExtractionProvenance(
            method="layout_profile",
            first_line_ref=b.first.ref if b.first else None,
            last_line_ref=b.last.ref if b.last else None,
        )
        if inline_answer:
            rec.source_answer = inline_answer
            prov.answer_ref = inline_ref
        elif marked:
            uniq = []
            for x in marked:
                if x.lower() not in [u.lower() for u in uniq]:
                    uniq.append(x)
            rec.source_answer = " & ".join(u.lower() for u in uniq)
            if len(uniq) > 1:
                rec.anomalies.append("multiple_marked_options:" + ",".join(u.lower() for u in uniq))
            prov.answer_ref = b.first.ref if b.first else None
        rec.extraction = prov
        records.append(rec)
    return records


def _parse_fib_blocks(blocks: list[_Block], doc_id: str, cspec: CompiledSpec) -> list[QuestionRecord]:
    records = []
    for b in blocks:
        if b.number == "0":
            continue
        parts = []
        answer = None
        answer_ref = None
        for ln, txt in b.lines:
            if cspec.inline_text_re is not None:
                im = cspec.inline_text_re.match(txt.strip())
                if im:
                    answer, answer_ref = im.group(1).strip(), ln.ref
                    continue
            parts.append(txt)
        pages = sorted({ln.page for ln, _ in b.lines} | ({b.stem_line.page} if b.stem_line else set()))
        rec = QuestionRecord(
            question_id=f"{doc_id}-fib-{b.number}",
            document_id=doc_id,
            section_type=QuestionType.FILL_IN_BLANK,
            question_number=b.number,
            question_text=" ".join(p for p in parts if p).strip(),
            source_spans=[SourceSpan(pages=pages)],
        )
        rec.source_answer = answer
        rec.extraction = ExtractionProvenance(
            method="layout_profile",
            first_line_ref=b.first.ref if b.first else None,
            last_line_ref=b.last.ref if b.last else None,
            answer_ref=answer_ref,
        )
        records.append(rec)
    return records


def _parse_matching(lines: list[Line], doc_id: str) -> list[QuestionRecord]:
    recs = _seg._parse_matching_blocks([(ln.page, ln.text) for ln in lines], doc_id)
    for r in recs:
        r.extraction = ExtractionProvenance(method="layout_profile")
    return recs


# ---------------------------------------------------------------------------
# Answer keys
# ---------------------------------------------------------------------------


def _joined_with_offsets(lines: list[Line]) -> tuple[str, list[int], list[Line]]:
    joined, starts = "", []
    for ln in lines:
        starts.append(len(joined))
        joined += ln.text + " "
    return joined, starts, lines


def _line_at(offset: int, starts: list[int], lines: list[Line]) -> Line:
    i = max(bisect.bisect_right(starts, offset) - 1, 0)
    return lines[i]


def _parse_number_key(sec: _Sec, cspec: CompiledSpec, anomalies: list[str]):
    """Numbered key: number and letter aren't always on the same *line*
    (pdf text of a multi-column table puts "50-" and "c" on consecutive
    lines) but always on the same page, so match per page over joined lines."""
    resolved: dict[str, tuple[str, int, str]] = {}
    # Join the WHOLE key section, not page by page: a "7-" at the bottom of a
    # page and its "c" at the top of the next is one entry (segmentation.py's
    # per-page join silently loses those).
    joined, starts, ls = _joined_with_offsets(sec.lines)
    for m in cspec.key_token_re.finditer(joined):
        num, ans = m.group(1), _norm_answer(m.group(2))
        ln = _line_at(m.start(), starts, ls)
        if num in resolved and resolved[num][0] != ans:
            anomalies.append(f"answer_key_repeat_conflict:{num}:{resolved[num][0]}_vs_{ans}")
        else:
            resolved.setdefault(num, (ans, ln.page, ln.ref))
    return resolved


_KEYCHARS_RE = re.compile(r"^[\s|:\-.&,a-fA-F0-9]+$")


def _parse_grouped_key(sec: _Sec, cspec: CompiledSpec, anomalies: list[str]):
    """Chapter-grouped key: heading, then entries, then the next heading.
    Returns {norm_label: (label, {number: (answer, ref)})} in document order.
    A group can continue across a page break and can appear twice (vision
    loops); repeats merge, and a repeat that disagrees is flagged."""
    groups: dict[str, tuple[str, dict[str, tuple[str, str]]]] = {}
    label: str | None = None
    buf: list[Line] = []

    def flush():
        if label is None or not buf:
            return
        key = _seg._norm_label(label)
        _, answers = groups.setdefault(key, (label, {}))
        joined, starts, ls = _joined_with_offsets(buf)
        for m in cspec.key_token_re.finditer(joined):
            num, val = m.group(1), _norm_answer(m.group(2))
            ref = _line_at(m.start(), starts, ls).ref
            if num in answers and answers[num][0] != val:
                anomalies.append(f"answer_key_repeat_conflict:{label}-{num}:{answers[num][0]}_vs_{val}")
            else:
                answers.setdefault(num, (val, ref))

    for ln in sec.lines:
        t = ln.text
        if _KEYCHARS_RE.match(t) and any(ch.isdigit() for ch in t):
            buf.append(ln)
        elif _is_heading(t, cspec):
            flush()
            label, buf = t.strip(), []
    flush()
    return {k: v for k, v in groups.items() if v[1]}


def _attach_keys(
    records: list[QuestionRecord],
    secs: list[_Sec],
    cspec: CompiledSpec,
    doc_id: str,
    doc_anomalies: list[str],
    unmatched: list[UnmatchedAnswerKeyEntry],
) -> None:
    ak = cspec.spec.answer_key
    key_secs = [s for s in secs if s.kind == "answer_key"]
    if ak.mode != "section":
        return
    if not key_secs:
        doc_anomalies.append("answer_key_section_not_found")
        return

    by_number = {
        (r.section_type, r.question_number): r for r in records if not cspec.grouped_mcq or r.section_type != QuestionType.MCQ
    }
    matching_numbers = sorted(
        (r.question_number for r in records if r.section_type == QuestionType.MATCHING), key=_seg._numeric_sort_key
    )

    def set_answer(rec: QuestionRecord, ans: str, ref: str | None, conflict_tag: str | None = None):
        if rec.source_answer is not None and rec.source_answer != ans:
            rec.anomalies.append(conflict_tag or f"conflicting_answer_key_value:{rec.source_answer}_vs_{ans}")
        rec.source_answer = ans
        if rec.extraction is not None and ref:
            rec.extraction.answer_ref = ref

    for sec in key_secs:
        if sec.target is None:
            doc_anomalies.append(f"answer_key_without_content_kind@{sec.header_ref}")
            continue
        first_page = sec.lines[0].page if sec.lines else 0
        if sec.target == "mcq" and ak.grouped:
            groups = _parse_grouped_key(sec, cspec, doc_anomalies)
            content_norms = sorted({_seg._norm_label(r.group_label) for r in records if r.group_label})
            matched_content: set[str] = set()
            for key_norm, (key_label, answers) in groups.items():
                target = _seg._match_group_label(key_norm, content_norms)
                if target is None:
                    doc_anomalies.append(f"answer_key_group_unmatched:{key_label}")
                    continue
                matched_content.add(target)
                recs = {
                    r.question_number: r
                    for r in records
                    if r.group_label and _seg._norm_label(r.group_label) == target
                }
                for num, (ans, ref) in answers.items():
                    rec = recs.get(num)
                    if rec is None:
                        unmatched.append(
                            UnmatchedAnswerKeyEntry(
                                raw_token=f"{key_label}:{num}",
                                raw_value=ans,
                                section_type=QuestionType.MCQ,
                                page=first_page,
                            )
                        )
                        continue
                    set_answer(rec, ans, ref)
            for cn in content_norms:
                if cn not in matched_content:
                    lab = next(r.group_label for r in records if r.group_label and _seg._norm_label(r.group_label) == cn)
                    doc_anomalies.append(f"no_answer_key_for_group:{lab}")
        elif sec.target == "mcq":
            resolved = _parse_number_key(sec, cspec, doc_anomalies)
            for number, (ans, pg, ref) in resolved.items():
                rec = by_number.get((QuestionType.MCQ, number))
                if rec is not None:
                    set_answer(rec, ans, ref)
                else:
                    unmatched.append(
                        UnmatchedAnswerKeyEntry(raw_token=number, raw_value=ans, section_type=QuestionType.MCQ, page=pg)
                    )
        elif sec.target == "fill_in_blank":
            tuples = [(ln.page, ln.text) for ln in sec.lines]
            for b in _seg._split_numbered_blocks(tuples, cspec.stem_re):
                if b.number == "0":
                    continue
                text = " ".join(t for _, t in b.lines).strip()
                if not text:
                    continue
                rec = by_number.get((QuestionType.FILL_IN_BLANK, b.number))
                if rec is not None:
                    set_answer(rec, text, None, "conflicting_answer_key_value")
                else:
                    unmatched.append(
                        UnmatchedAnswerKeyEntry(
                            raw_token=b.number, raw_value=text, section_type=QuestionType.FILL_IN_BLANK, page=first_page
                        )
                    )
        elif sec.target == "matching":
            expected = {
                str(i + 1): len(by_number[(QuestionType.MATCHING, num)].matching_pairs)
                for i, num in enumerate(matching_numbers)
                if (QuestionType.MATCHING, num) in by_number
            }
            pseudo = _seg._Section("answer_key", QuestionType.MATCHING)
            pseudo.lines = [(ln.page, ln.text) for ln in sec.lines]
            resolved_groups, ambiguous = _seg._parse_matching_answer_key(pseudo, expected)
            for gnum, ans in resolved_groups.items():
                idx = int(gnum) - 1
                rec = None
                if 0 <= idx < len(matching_numbers):
                    rec = by_number.get((QuestionType.MATCHING, matching_numbers[idx]))
                if rec is not None:
                    set_answer(rec, ans, None, "conflicting_answer_key_value")
                    if gnum in ambiguous:
                        rec.anomalies.append("matching_answer_key_possibly_misaligned:answer_key_batch_short_a_token")
                else:
                    unmatched.append(
                        UnmatchedAnswerKeyEntry(
                            raw_token=f"({gnum})", raw_value=ans, section_type=QuestionType.MATCHING, page=first_page
                        )
                    )


def _flag_missing_answers(records: list[QuestionRecord], cspec: CompiledSpec) -> None:
    """The spec says this document has a key. A record that ended up with no
    source_answer must say so itself -- otherwise it would sit in auto_import
    looking complete."""
    if cspec.spec.answer_key.mode == "none":
        return
    for r in records:
        if r.source_answer is None:
            r.anomalies.append("no_answer_key_entry")


def _check_answers_vs_options(records: list[QuestionRecord]) -> None:
    """A key letter that names no option is a flag, not a repair."""
    for r in records:
        if r.section_type != QuestionType.MCQ or not r.source_answer or not r.options:
            continue
        labels = {o.label.lower() for o in r.options}
        parts = [p for p in re.split(r"\s*[&,]\s*", r.source_answer.lower()) if p]
        if any(p not in labels for p in parts):
            r.anomalies.append(f"answer_not_in_options:{r.source_answer}")


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------


def extract_with_spec(
    doc: SourceDocument,
    lines: list[Line],
    cspec: CompiledSpec,
    *,
    method: str = "layout_profile",
    profile_id: str | None = None,
    profile_state: str | None = None,
) -> QuestionDocument:
    doc_anomalies: list[str] = []
    unmatched: list[UnmatchedAnswerKeyEntry] = []
    secs = _split_sections(lines, cspec, doc_anomalies)

    records: list[QuestionRecord] = []
    grouped_mode = False
    for sec in secs:
        if sec.kind != "content" or sec.target is None:
            continue
        if sec.target == "mcq":
            if cspec.grouped_mcq:
                grouped_mode = True
                blocks, labels = _split_grouped(sec.lines, cspec)
                records.extend(_parse_mcq_blocks(blocks, doc.document_id, labels, cspec))
                for i, lab in enumerate(labels):
                    if lab is None:
                        doc_anomalies.append(f"group_heading_not_found:group_{i}")
            else:
                records.extend(_parse_mcq_blocks(_split_blocks(sec.lines, cspec), doc.document_id, None, cspec))
        elif sec.target == "fill_in_blank":
            records.extend(_parse_fib_blocks(_split_blocks(sec.lines, cspec), doc.document_id, cspec))
        elif sec.target == "matching":
            records.extend(_parse_matching(sec.lines, doc.document_id))

    records, dup_anoms = _seg._dedupe_or_flag_questions(records, keep_order=grouped_mode)
    doc_anomalies.extend(dup_anoms)

    _attach_keys(records, secs, cspec, doc.document_id, doc_anomalies, unmatched)
    _flag_missing_answers(records, cspec)
    _check_answers_vs_options(records)

    if not any(s.kind == "content" and s.target for s in secs):
        doc_anomalies.append("no_sections_detected")

    for r in records:
        if r.extraction is not None:
            r.extraction.method = method
            r.extraction.profile_id = profile_id
            r.extraction.profile_state = profile_state

    qdoc = QuestionDocument(
        document_id=doc.document_id,
        original_filename=doc.original_filename,
        subject=doc.subject,
        topic=doc.topic,
        subtopic=None,
        questions=records,
        unmatched_answer_key_entries=unmatched,
        document_anomalies=doc_anomalies,
    )
    qdoc.compute_rollup()
    return qdoc
