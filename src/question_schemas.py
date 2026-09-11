"""
Data structures for the question segmentation/parsing layer.

This layer reads ONLY from SourceDocument/PageExtraction (schemas.py) --
never from raw files. Its job: turn page-shaped text into
question-shaped records, using pure text-structure parsing (regex,
line/block heuristics). No LLM calls happen here -- that's a later,
separate stage, same "don't collapse stages" discipline as the rest of
the pipeline.

Same "flag, don't repair" principle as text_quality.py:
  - source_answer (parsed from an answer-key page) and verified_answer
    (filled in by a future verification stage) are kept permanently
    separate. Nothing here ever guesses verified_answer.
  - Anything ambiguous, conflicting, or malformed becomes a structured
    anomaly or Suggestion, never a silent repair or a dropped record.

subtopic/taxonomy: still explicitly out of scope. QuestionDocument
carries subject/topic straight through from SourceDocument; subtopic
stays null.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class QuestionType(str, Enum):
    MCQ = "mcq"
    FILL_IN_BLANK = "fill_in_blank"
    MATCHING = "matching"


class SourceSpan(BaseModel):
    """Where in the source document a piece of content came from.

    `pages` is a list (not a single int) specifically so a question
    that starts on one page and finishes on the next can carry both
    page numbers -- multi-page stitching produces a SourceSpan with
    len(pages) > 1 rather than picking just one page and losing the
    other.
    """

    pages: list[int]


class Option(BaseModel):
    """One answer choice in an MCQ."""

    label: str  # as it appeared in source: "a", "A", "1", etc. -- not normalized to a fixed set
    text: str


class MatchingPair(BaseModel):
    """One left/right pair in a matching-section sub-question, as
    presented in column (A) / column (B) form. Kept as raw source
    pairs -- this is NOT the answer key; the correct pairing (if any)
    comes from a MODEL ANSWERS block and is attached as source_answer,
    same as MCQ/fill-in-blank.
    """

    left_label: str  # "1", "2", ... from column (A)
    left_text: str
    right_label: str  # "A", "B", ... from column (B)
    right_text: str


class Suggestion(BaseModel):
    """A non-binding hint generated during parsing -- e.g. 'this
    garbled answer-key token is probably question 50, based on its
    position in the sequence'. Suggestions never get written into
    canonical fields automatically; a human or a later stage decides
    whether to accept one.
    """

    kind: str  # e.g. "answer_key_ocr_correction", "possible_duplicate"
    message: str
    confidence: Optional[float] = None


class UnmatchedAnswerKeyEntry(BaseModel):
    """An answer-key line that couldn't be confidently tied to a
    question number -- e.g. the 'so-' OCR glitch instead of '50-'.
    Recorded as its own top-level record rather than silently dropped
    or silently guessed into a question's source_answer.
    """

    raw_token: str  # the raw key text as it appeared, e.g. "so-"
    raw_value: str  # the answer value paired with it, e.g. "c"
    section_type: QuestionType
    page: int
    suggestions: list[Suggestion] = Field(default_factory=list)


class QuestionRecord(BaseModel):
    """One parsed question, of whatever type."""

    question_id: str  # f"{document_id}-q{stable_index}"
    document_id: str
    section_type: QuestionType

    # The number as printed in the source (string, not int -- some
    # sources print "4" for a matching sub-question restart, and OCR
    # glitches like "so-" belong in UnmatchedAnswerKeyEntry, not here).
    question_number: str

    question_text: str
    options: list[Option] = Field(default_factory=list)
    matching_pairs: list[MatchingPair] = Field(default_factory=list)

    # Parsed straight from a MODEL ANSWERS block, kept as source text
    # (e.g. "b", "Anatomical position", "1-B,2-C,3-D,4-A,5-E"). Never
    # hand-edited, never used as a stand-in for verified_answer.
    source_answer: Optional[str] = None

    # Reserved for a future verification stage. Always null here --
    # this layer has no way to verify anything and must not pretend to.
    verified_answer: Optional[str] = None

    source_spans: list[SourceSpan] = Field(default_factory=list)

    # Structured anomalies (e.g. "multi_block_answer_conflict",
    # "mid_question_duplicate_content") -- deduplicated, never silently
    # repaired away.
    anomalies: list[str] = Field(default_factory=list)
    suggestions: list[Suggestion] = Field(default_factory=list)


class QuestionDocument(BaseModel):
    """Output of the segmentation stage for one SourceDocument."""

    document_id: str
    original_filename: str
    subject: Optional[str] = None
    topic: Optional[str] = None
    subtopic: Optional[str] = None  # still a placeholder -- not this layer's job

    questions: list[QuestionRecord] = Field(default_factory=list)
    unmatched_answer_key_entries: list[UnmatchedAnswerKeyEntry] = Field(
        default_factory=list
    )

    # Document-level parsing anomalies not tied to one question, e.g.
    # "no_sections_detected", "duplicate_MODEL_ANSWERS_block_conflict".
    document_anomalies: list[str] = Field(default_factory=list)

    needs_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)

    def compute_rollup(self) -> None:
        reasons: list[str] = list(self.document_anomalies)
        for q in self.questions:
            for a in q.anomalies:
                reasons.append(f"{q.question_id}:{a}")
        for u in self.unmatched_answer_key_entries:
            reasons.append(f"unmatched_answer_key:{u.raw_token}@p{u.page}")
        self.review_reasons = reasons
        self.needs_review = len(reasons) > 0
