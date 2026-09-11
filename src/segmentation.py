"""
Question segmentation/parsing stage.

Reads SourceDocument JSON (produced by the ingestion/reconstruction
layer) and produces a QuestionDocument: individual question records
with text, options, and a source_answer parsed from a MODEL ANSWERS
block, all with page provenance.

Pure text-structure parsing -- regex and line/block heuristics only.
No LLM calls happen here; that's a later, separate stage.

Same "flag, don't repair" discipline as the rest of the pipeline:
  - Anything ambiguous (OCR-glitched answer-key tokens, conflicting
    duplicate blocks, unmatched entries) becomes a structured anomaly
    or Suggestion. Nothing is silently dropped, merged, or guessed
    into a canonical field.
"""

from __future__ import annotations

import difflib
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from schemas import SourceDocument  # noqa: E402
from question_schemas import (  # noqa: E402
    MatchingPair,
    Option,
    QuestionDocument,
    QuestionRecord,
    QuestionType,
    SourceSpan,
    Suggestion,
    UnmatchedAnswerKeyEntry,
)

# ---------------------------------------------------------------------------
# Section header detection
# ---------------------------------------------------------------------------

_SECTION_HEADERS: list[tuple[re.Pattern, QuestionType]] = [
    (re.compile(r"^m\.?\s*c\.?\s*q'?s?\.?$", re.I), QuestionType.MCQ),
    (re.compile(r"^fill\s+in\s+the\s+blanks?$", re.I), QuestionType.FILL_IN_BLANK),
    (re.compile(r"^(cross[\s-]*)?matching$", re.I), QuestionType.MATCHING),
]
_ANSWER_KEY_HEADER = re.compile(r"^model\s+answers?$", re.I)

# ---------------------------------------------------------------------------
# Line-level patterns
# ---------------------------------------------------------------------------

# A numbered question stem, e.g. "1-", "26.", "32)", "33-)", "28:..",
# possibly preceded by a short stray OCR token ("X 16-").
_STEM_RE = re.compile(r"^(?:\S{1,2}\s+)?(\d{1,3})\s*[-:.)]+\.?\s*(.*)$")

# An MCQ option line: "a-", "a.", "a)", any case, optionally indented.
_OPTION_RE = re.compile(r"^([a-eA-E])\s*[-.)]\s*(.*)$")

# A bare page-number footer line -- digits only, no trailing
# punctuation (question stems always carry a trailing separator).
_PAGE_NUM_RE = re.compile(r"^\d{1,4}$")

# Answer-key token: a question number (or the "so-" OCR misread of
# "50-") paired with a single-letter MCQ answer.
_MCQ_KEY_TOKEN_RE = re.compile(r"(\d{1,3}|\bso\b)\s*[-.]\s*([a-eA-E])\b", re.I)

# Matching-section stem: requires the word "match" so numbered
# column-A items ("1- Supine") don't get mistaken for new questions.
# Captures the rest of the line too (group 2) -- without it the stem
# sentence itself (everything from "Match..." onward) would match the
# lookahead but never get carried into the question record.
_MATCHING_STEM_RE = re.compile(r"^(\d{1,3})\s*[-.)]+\s*(match.*)$", re.I)

# Matching-answer-key tokens, matched in one interleaved pass so
# ordering between group markers and pair values is preserved even
# though (like the MCQ key) the number and its letter often land on
# separate lines: either a group marker "(1)" or a "digit-letter" pair.
_MATCHING_TOKEN_RE = re.compile(r"\((\d{1,3})\)|(\d{1,2})\s*[-.]+\s*([A-Za-z])\b")


# ---------------------------------------------------------------------------
# Furniture stripping
# ---------------------------------------------------------------------------


def _clean_page_lines(doc: SourceDocument) -> dict[int, list[str]]:
    """Per-page lines with running headers/footers and bare page-number
    lines removed. Furniture is detected generically (frequency across
    pages), not hardcoded to this document's specific title text --
    the pipeline has to work on files it hasn't seen yet.
    """
    page_lines: dict[int, list[str]] = {}
    for p in doc.pages:
        if not p.text:
            continue
        lines = [ln.strip() for ln in p.text.split("\n")]
        page_lines[p.page] = [ln for ln in lines if ln]

    n_pages = max(len(page_lines), 1)
    norm_counts: Counter[str] = Counter()
    for lines in page_lines.values():
        seen = set()
        for ln in lines:
            norm = re.sub(r"\s+", " ", ln.lower()).strip(" .")
            if norm and len(norm) <= 60 and norm not in seen:
                norm_counts[norm] += 1
                seen.add(norm)

    # A line that repeats near-verbatim across a healthy fraction of
    # pages is a running header/footer, UNLESS it's one of the
    # section headers themselves (those legitimately repeat 2-3 times
    # across a whole document but are load-bearing structure, not
    # furniture) -- guarded by requiring the ratio to be fairly high.
    furniture = {
        norm
        for norm, cnt in norm_counts.items()
        if cnt >= 3 and cnt / n_pages >= 0.3
    }

    cleaned: dict[int, list[str]] = {}
    for pg, lines in page_lines.items():
        kept = []
        for ln in lines:
            norm = re.sub(r"\s+", " ", ln.lower()).strip(" .")
            if norm in furniture:
                continue
            if _PAGE_NUM_RE.match(ln):
                continue
            kept.append(ln)
        cleaned[pg] = kept
    return cleaned


# ---------------------------------------------------------------------------
# Sectioning
# ---------------------------------------------------------------------------


class _Section:
    __slots__ = ("kind", "target_type", "lines")

    def __init__(self, kind: str, target_type: QuestionType | None):
        self.kind = kind  # "content" or "answer_key"
        self.target_type = target_type
        self.lines: list[tuple[int, str]] = []


def _split_into_sections(cleaned: dict[int, list[str]]) -> list[_Section]:
    sections: list[_Section] = []
    current: _Section | None = None
    last_content_type: QuestionType | None = None

    for pg in sorted(cleaned):
        for ln in cleaned[pg]:
            header_type = None
            for pattern, qtype in _SECTION_HEADERS:
                if pattern.match(ln):
                    header_type = qtype
                    break

            if header_type is not None:
                current = _Section("content", header_type)
                sections.append(current)
                last_content_type = header_type
                continue

            if _ANSWER_KEY_HEADER.match(ln):
                current = _Section("answer_key", last_content_type)
                sections.append(current)
                continue

            if current is None:
                # Text before any recognized header -- shouldn't
                # normally happen on these documents, but don't drop
                # it silently; stash it as an untyped content section
                # so it surfaces rather than vanishing.
                current = _Section("content", None)
                sections.append(current)

            current.lines.append((pg, ln))

    return sections


# ---------------------------------------------------------------------------
# Raw question-block splitting (MCQ / fill-in-blank)
# ---------------------------------------------------------------------------


class _RawBlock:
    def __init__(self, number: str):
        self.number = number
        self.lines: list[tuple[int, str]] = []  # (page, raw line, incl. options)


def _split_numbered_blocks(
    lines: list[tuple[int, str]], stem_re: re.Pattern
) -> list[_RawBlock]:
    blocks: list[_RawBlock] = []
    current: _RawBlock | None = None

    for pg, ln in lines:
        m = stem_re.match(ln)
        if m:
            current = _RawBlock(number=m.group(1))
            rest = m.group(2).strip() if m.lastindex and m.lastindex >= 2 else ""
            blocks.append(current)
            if rest:
                current.lines.append((pg, rest))
            continue
        if current is None:
            # Content before the first recognizable stem in this
            # section -- keep it as block "0" rather than dropping it.
            current = _RawBlock(number="0")
            blocks.append(current)
        current.lines.append((pg, ln))

    return blocks


def _parse_mcq_blocks(
    blocks: list[_RawBlock], document_id: str
) -> list[QuestionRecord]:
    records = []
    for b in blocks:
        if b.number == "0":
            continue  # preamble noise before question 1, e.g. section title remnants
        text_parts: list[str] = []
        options: list[Option] = []
        pages: set[int] = set()
        for pg, ln in b.lines:
            pages.add(pg)
            om = _OPTION_RE.match(ln)
            if om:
                options.append(Option(label=om.group(1), text=om.group(2).strip()))
            else:
                text_parts.append(ln)

        record = QuestionRecord(
            question_id=f"{document_id}-mcq-{b.number}",
            document_id=document_id,
            section_type=QuestionType.MCQ,
            question_number=b.number,
            question_text=" ".join(text_parts).strip(),
            options=options,
            source_spans=[SourceSpan(pages=sorted(pages))],
        )
        if len(options) != 4:
            record.anomalies.append(f"unexpected_option_count:{len(options)}")
        records.append(record)
    return records


def _parse_fill_in_blank_blocks(
    blocks: list[_RawBlock], document_id: str
) -> list[QuestionRecord]:
    records = []
    for b in blocks:
        if b.number == "0":
            continue
        pages = sorted({pg for pg, _ in b.lines})
        text = " ".join(ln for _, ln in b.lines).strip()
        records.append(
            QuestionRecord(
                question_id=f"{document_id}-fib-{b.number}",
                document_id=document_id,
                section_type=QuestionType.FILL_IN_BLANK,
                question_number=b.number,
                question_text=text,
                source_spans=[SourceSpan(pages=pages)],
            )
        )
    return records


# ---------------------------------------------------------------------------
# Matching-section parsing
# ---------------------------------------------------------------------------


def _parse_matching_blocks(
    lines: list[tuple[int, str]], document_id: str
) -> list[QuestionRecord]:
    blocks = _split_numbered_blocks(lines, _MATCHING_STEM_RE)
    records = []
    for b in blocks:
        if b.number == "0":
            continue
        pages: set[int] = set()
        stem_parts: list[str] = []
        pairs: list[MatchingPair] = []
        pending_left: tuple[str, str] | None = None

        for pg, ln in b.lines:
            pages.add(pg)
            if ln in ("(A)", "(B)"):
                continue
            lm = re.match(r"^(\d{1,2})\s*[-.)]\s*(.*)$", ln)
            rm = re.match(r"^([A-Za-z])\s*[-.)]\s*(.*)$", ln)
            if lm and pending_left is None:
                pending_left = (lm.group(1), lm.group(2).strip())
                continue
            if rm and pending_left is not None:
                pairs.append(
                    MatchingPair(
                        left_label=pending_left[0],
                        left_text=pending_left[1],
                        right_label=rm.group(1).upper(),
                        right_text=rm.group(2).strip(),
                    )
                )
                pending_left = None
                continue
            # Doesn't look like a pair line yet -- still part of the
            # stem sentence (e.g. "Match the position in column (A)
            # with its description in column (B):").
            if not pairs and pending_left is None:
                stem_parts.append(ln)

        record = QuestionRecord(
            question_id=f"{document_id}-match-{b.number}",
            document_id=document_id,
            section_type=QuestionType.MATCHING,
            question_number=b.number,
            question_text=" ".join(stem_parts).strip(),
            matching_pairs=pairs,
            source_spans=[SourceSpan(pages=sorted(pages))],
        )
        if pending_left is not None:
            record.anomalies.append(f"unpaired_left_item:{pending_left[0]}")
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# Answer-key parsing
# ---------------------------------------------------------------------------


def _parse_mcq_answer_key(
    section: _Section,
) -> tuple[dict[str, tuple[str, int]], list[UnmatchedAnswerKeyEntry]]:
    """Returns (number -> (answer, page)), plus unmatched entries for
    tokens that couldn't be resolved to a question number (e.g. the
    "so-" OCR misread of "50-")."""
    resolved: dict[str, tuple[str, int]] = {}
    unmatched: list[UnmatchedAnswerKeyEntry] = []
    ordered_numeric: list[int] = []

    # The number and its answer letter aren't always on the same
    # *line* -- clean pdf-text extraction of a multi-column layout
    # often puts "50-" and its "c" on consecutive lines rather than
    # side by side. They ARE always on the same *page*, so join each
    # page's lines before matching rather than matching line-by-line.
    page_order: list[int] = []
    page_lines: dict[int, list[str]] = {}
    for pg, ln in section.lines:
        page_lines.setdefault(pg, []).append(ln)
        if pg not in page_order:
            page_order.append(pg)

    for pg in page_order:
        joined = " ".join(page_lines[pg])
        matches = list(_MCQ_KEY_TOKEN_RE.finditer(joined))

        # First pass: collect every resolvable digit token on this
        # page so the gap-inference below can see the WHOLE page's
        # sequence, not just tokens preceding a given glitch -- a
        # glitch near the start of a page would otherwise see almost
        # nothing "seen so far" and every later number would look
        # like a gap.
        page_numeric: list[int] = []
        for m in matches:
            token = m.group(1)
            if token.isdigit():
                page_numeric.append(int(token))

        for m in matches:
            token, value = m.group(1), m.group(2).lower()
            if token.isdigit():
                resolved[token] = (value, pg)
                ordered_numeric.append(int(token))
            else:
                # Non-numeric token ("so") -- infer the missing number
                # from the full run of numbers on this page rather
                # than guessing blindly.
                entry = UnmatchedAnswerKeyEntry(
                    raw_token=token,
                    raw_value=value,
                    section_type=QuestionType.MCQ,
                    page=pg,
                )
                gap = _infer_sequence_gap(page_numeric)
                if gap is not None:
                    entry.suggestions.append(
                        Suggestion(
                            kind="answer_key_ocr_correction",
                            message=(
                                f"'{token}-' likely misread of '{gap}-' "
                                f"(gap in surrounding answer sequence)"
                            ),
                            confidence=0.7,
                        )
                    )
                unmatched.append(entry)
    return resolved, unmatched


def _infer_sequence_gap(seen: list[int]) -> int | None:
    if len(seen) < 2:
        return None
    full_range = set(range(min(seen), max(seen) + 1))
    missing = sorted(full_range - set(seen))
    return missing[0] if len(missing) == 1 else None


def _parse_fill_in_blank_answer_key(section: _Section) -> dict[str, str]:
    blocks = _split_numbered_blocks(section.lines, _STEM_RE)
    out = {}
    for b in blocks:
        if b.number == "0":
            continue
        text = " ".join(ln for _, ln in b.lines).strip()
        if text:
            out[b.number] = text
    return out


def _parse_matching_answer_key(
    section: _Section, expected_counts: dict[str, int]
) -> tuple[dict[str, str], set[str]]:
    """Returns matching-sub-question-number -> serialized answer
    string, e.g. "1-B,2-C,3-D,4-A,5-E".

    The source layout prints several matching sub-questions' answer
    keys side by side as columns -- e.g. group headers "(1) (2) (3)
    (4)" followed by row-major data: all four groups' "1-" answers,
    then all four groups' "2-" answers, and so on. That's fine when
    every group in the batch has the same number of pairs, but real
    documents don't guarantee that (group 3 here has 4 pairs while its
    neighbors have 5) -- a group with fewer pairs simply stops
    appearing in later rows, shifting which group each token belongs
    to for anyone assuming a fixed column count.

    `expected_counts` (group number -> how many pairs that matching
    question actually has, from the already-parsed content section)
    is what makes this resolvable: walk the token stream in order,
    advancing a per-active-group row counter, and only expect a token
    for a group once its own row counter is still within its expected
    count. This is still a heuristic reconstruction of a layout, not a
    guarantee -- if expected_counts is missing or wrong for a group,
    misattribution is possible, which is why it's just as important
    that resolved answers here can still be flagged as
    conflicting_answer_key_value upstream if they don't match anything
    plausible.
    """
    groups: dict[str, list[str]] = {}
    ambiguous_groups: set[str] = set()
    active_batch: list[str] = []
    pending_tokens: list[str] = []

    def flush_batch():
        if not active_batch or not pending_tokens:
            return
        remaining = {g: expected_counts.get(g, 5) for g in active_batch}
        expected_total = sum(remaining.values())
        if len(pending_tokens) < expected_total:
            # The batch is short at least one token relative to what
            # the content section says these questions should have.
            # Position-based reconstruction can still make a
            # best-effort assignment, but WHICH group is actually
            # missing its answer isn't determinable from structure
            # alone -- flag every group in this batch rather than
            # silently trusting whichever one the round-robin left
            # short (that's an artifact of processing order, not
            # evidence about the source).
            ambiguous_groups.update(active_batch)
        group_cycle = list(active_batch)
        gi = 0
        for tok in pending_tokens:
            # advance to the next group in the row that still expects
            # a token for this row position
            attempts = 0
            while remaining.get(group_cycle[gi], 0) <= 0 and attempts < len(group_cycle):
                gi = (gi + 1) % len(group_cycle)
                attempts += 1
            g = group_cycle[gi]
            groups.setdefault(g, []).append(tok)
            remaining[g] -= 1
            gi = (gi + 1) % len(group_cycle)
        pending_tokens.clear()

    # Group markers and pair values are frequently split across
    # separate lines (same layout quirk as the MCQ key), so join each
    # page's lines into one string and walk group markers + pair
    # tokens in a single ordered pass, instead of matching line by
    # line where a lone "1-" or "B" would never match anything.
    page_order: list[int] = []
    page_lines: dict[int, list[str]] = {}
    for pg, ln in section.lines:
        page_lines.setdefault(pg, []).append(ln)
        if pg not in page_order:
            page_order.append(pg)

    for pg in page_order:
        joined = " ".join(page_lines[pg])
        for m in _MATCHING_TOKEN_RE.finditer(joined):
            if m.group(1) is not None:
                group_num = m.group(1)
                if pending_tokens:
                    flush_batch()
                    active_batch = []
                active_batch.append(group_num)
            else:
                if not active_batch:
                    continue
                pending_tokens.append(f"{m.group(2)}-{m.group(3).upper()}")
    flush_batch()

    return {k: ",".join(v) for k, v in groups.items() if v}, ambiguous_groups


# ---------------------------------------------------------------------------
# Cross-block conflict / duplicate detection
# ---------------------------------------------------------------------------


def _dedupe_or_flag_questions(
    records: list[QuestionRecord],
) -> tuple[list[QuestionRecord], list[str]]:
    """Real vision-transcribed pages have shown the model looping and
    re-emitting a question a second time mid-page (not always
    separated by a blank line, so the ingestion-layer duplicate
    detector doesn't always catch it). Two questions with the same
    section_type + question_number are a duplication, not two
    different questions -- but WHICH copy is correct isn't something
    text structure alone can decide, so this only merges near-identical
    duplicates (keeping one, flagging it) and otherwise surfaces a
    conflict rather than silently picking one.
    """
    by_key: dict[tuple[QuestionType, str], list[QuestionRecord]] = {}
    for r in records:
        by_key.setdefault((r.section_type, r.question_number), []).append(r)

    out: list[QuestionRecord] = []
    doc_anomalies: list[str] = []
    for (qtype, number), group in by_key.items():
        if len(group) == 1:
            out.append(group[0])
            continue

        base = group[0]
        for other in group[1:]:
            ratio = difflib.SequenceMatcher(
                None, base.question_text, other.question_text
            ).ratio()
            merged_pages = sorted(
                {p for span in base.source_spans + other.source_spans for p in span.pages}
            )
            if ratio >= 0.85:
                base.anomalies.append(
                    f"duplicate_question_merged:sim_{ratio:.2f}"
                )
            else:
                base.anomalies.append(
                    f"duplicate_question_number_conflict:sim_{ratio:.2f}"
                )
                doc_anomalies.append(
                    f"{qtype.value}_q{number}_conflicting_duplicates"
                )
            base.source_spans = [SourceSpan(pages=merged_pages)]
        out.append(base)

    out.sort(key=lambda r: (r.section_type.value, _numeric_sort_key(r.question_number)))
    return out, doc_anomalies


def _numeric_sort_key(number: str):
    return int(number) if number.isdigit() else float("inf")


# ---------------------------------------------------------------------------
# Top-level assembly
# ---------------------------------------------------------------------------


def segment_document(doc: SourceDocument) -> QuestionDocument:
    cleaned = _clean_page_lines(doc)
    sections = _split_into_sections(cleaned)

    all_records: list[QuestionRecord] = []
    unmatched_all: list[UnmatchedAnswerKeyEntry] = []
    doc_anomalies: list[str] = []

    # answer-key parsing needs to happen after we've parsed every
    # content section, so it can key back into MATCHING sub-questions
    # by sequential position -- but content sections must be walked in
    # document order to assign consistent numbers. Two passes: first
    # collect content records, then attach answer keys.
    content_sections = [s for s in sections if s.kind == "content" and s.target_type]
    key_sections = [s for s in sections if s.kind == "answer_key"]

    matching_key_group_counter = 0

    for section in content_sections:
        if section.target_type == QuestionType.MCQ:
            blocks = _split_numbered_blocks(section.lines, _STEM_RE)
            all_records.extend(_parse_mcq_blocks(blocks, doc.document_id))
        elif section.target_type == QuestionType.FILL_IN_BLANK:
            blocks = _split_numbered_blocks(section.lines, _STEM_RE)
            all_records.extend(_parse_fill_in_blank_blocks(blocks, doc.document_id))
        elif section.target_type == QuestionType.MATCHING:
            all_records.extend(_parse_matching_blocks(section.lines, doc.document_id))

    all_records, dup_anomalies = _dedupe_or_flag_questions(all_records)
    doc_anomalies.extend(dup_anomalies)

    by_number: dict[tuple[QuestionType, str], QuestionRecord] = {
        (r.section_type, r.question_number): r for r in all_records
    }

    # Track how many MATCHING questions we've seen in document order,
    # so the (1)/(2)/(3)... groups in a matching answer-key block can
    # be mapped back to the right sub-question even though they're
    # numbered by group position, not by the matching question's own
    # printed number (which also happens to coincide here, but isn't
    # guaranteed on messier sources).
    matching_numbers_in_order = sorted(
        (r.question_number for r in all_records if r.section_type == QuestionType.MATCHING),
        key=_numeric_sort_key,
    )

    for section in key_sections:
        if section.target_type == QuestionType.MCQ:
            resolved, unmatched = _parse_mcq_answer_key(section)
            unmatched_all.extend(unmatched)
            for number, (answer, page) in resolved.items():
                rec = by_number.get((QuestionType.MCQ, number))
                if rec:
                    if rec.source_answer is not None and rec.source_answer != answer:
                        rec.anomalies.append(
                            f"conflicting_answer_key_value:{rec.source_answer}_vs_{answer}"
                        )
                    rec.source_answer = answer
                else:
                    unmatched_all.append(
                        UnmatchedAnswerKeyEntry(
                            raw_token=number,
                            raw_value=answer,
                            section_type=QuestionType.MCQ,
                            page=page,
                        )
                    )
            # Try to resolve any unmatched OCR-glitch tokens (like
            # "so-") against questions that came back with no answer
            # at all -- if exactly one MCQ question in range has no
            # source_answer, that's a strong (but still flagged, not
            # silently applied) suggestion target.
            for entry in unmatched:
                for sugg in entry.suggestions:
                    if sugg.kind == "answer_key_ocr_correction":
                        pass  # suggestion already carries the guess; left for a human/reviewer to accept

        elif section.target_type == QuestionType.FILL_IN_BLANK:
            resolved_fib = _parse_fill_in_blank_answer_key(section)
            for number, answer in resolved_fib.items():
                rec = by_number.get((QuestionType.FILL_IN_BLANK, number))
                if rec:
                    if rec.source_answer is not None and rec.source_answer != answer:
                        rec.anomalies.append("conflicting_answer_key_value")
                    rec.source_answer = answer
                else:
                    unmatched_all.append(
                        UnmatchedAnswerKeyEntry(
                            raw_token=number,
                            raw_value=answer,
                            section_type=QuestionType.FILL_IN_BLANK,
                            page=section.lines[0][0] if section.lines else 0,
                        )
                    )

        elif section.target_type == QuestionType.MATCHING:
            expected_counts = {
                str(i + 1): len(by_number[(QuestionType.MATCHING, num)].matching_pairs)
                for i, num in enumerate(matching_numbers_in_order)
                if (QuestionType.MATCHING, num) in by_number
            }
            resolved_groups, ambiguous_groups = _parse_matching_answer_key(
                section, expected_counts
            )
            for group_num, answer in resolved_groups.items():
                idx = int(group_num) - 1
                if 0 <= idx < len(matching_numbers_in_order):
                    q_number = matching_numbers_in_order[idx]
                    rec = by_number.get((QuestionType.MATCHING, q_number))
                    if rec:
                        if rec.source_answer is not None and rec.source_answer != answer:
                            rec.anomalies.append("conflicting_answer_key_value")
                        rec.source_answer = answer
                        if group_num in ambiguous_groups:
                            rec.anomalies.append(
                                "matching_answer_key_possibly_misaligned:"
                                "answer_key_batch_short_a_token"
                            )
                        continue
                unmatched_all.append(
                    UnmatchedAnswerKeyEntry(
                        raw_token=f"({group_num})",
                        raw_value=answer,
                        section_type=QuestionType.MATCHING,
                        page=section.lines[0][0] if section.lines else 0,
                    )
                )

    if not content_sections:
        doc_anomalies.append("no_sections_detected")

    qdoc = QuestionDocument(
        document_id=doc.document_id,
        original_filename=doc.original_filename,
        subject=doc.subject,
        topic=doc.topic,
        subtopic=None,
        questions=all_records,
        unmatched_answer_key_entries=unmatched_all,
        document_anomalies=doc_anomalies,
    )
    qdoc.compute_rollup()
    return qdoc


def main() -> None:
    import argparse
    import json as _json

    parser = argparse.ArgumentParser(
        description="Segment reconstructed SourceDocuments into QuestionDocuments."
    )
    parser.add_argument("--input", default="../output", type=Path)
    parser.add_argument("--output", default="../questions", type=Path)
    args = parser.parse_args()

    input_root: Path = args.input.resolve()
    output_root: Path = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    doc_paths = sorted(input_root.glob("DOC-*.json"))
    if not doc_paths:
        print(f"No SourceDocuments found in {input_root}. Run main.py first.")
        return

    stats = {"processed": 0, "needs_review": 0, "total_questions": 0}
    for doc_path in doc_paths:
        doc = SourceDocument(**_json.loads(doc_path.read_text()))
        qdoc = segment_document(doc)

        out_path = output_root / f"{doc.document_id}.questions.json"
        out_path.write_text(_json.dumps(qdoc.model_dump(), indent=2, default=str))

        stats["processed"] += 1
        stats["total_questions"] += len(qdoc.questions)
        if qdoc.needs_review:
            stats["needs_review"] += 1
            print(
                f"[OK, FLAGGED] {doc.original_filename} -> {out_path.name} "
                f"({len(qdoc.questions)} questions, {len(qdoc.review_reasons)} reasons)"
            )
        else:
            print(f"[OK] {doc.original_filename} -> {out_path.name} ({len(qdoc.questions)} questions)")

    print("\n--- Segmentation summary ---")
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
