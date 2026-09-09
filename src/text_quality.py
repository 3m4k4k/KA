"""
Heuristics for deciding whether extracted text is actually usable.

This is deliberately NOT just "is the string empty". Bad PDF fonts,
broken ligatures, and mis-decoded encodings frequently produce text
that is non-empty but useless -- e.g. lots of replacement characters
or symbol soup where prose should be.

Deliberately NOT a proxy for "is this prose". Answer-key pages,
option lists, and numeric tables are legitimate, correctly-extracted
content that happens to have very few English "words" in them --
penalizing that would flag good extractions as garbled. The checks
below look at whether the *characters themselves* look like real
content (letters, digits, common punctuation) versus corruption
(control chars, mojibake, repeated spam), never at word density.

Nothing here tries to fix the text. It only decides: trust it, or
route it to OCR/vision.
"""

from __future__ import annotations

import difflib
import unicodedata
from dataclasses import dataclass

_REPLACEMENT_CHAR = "\ufffd"
_ALLOWED_WHITESPACE = {"\n", "\r", "\t"}

# Punctuation/symbols that show up constantly in legitimate exam
# content -- option markers, ranges, units, math -- and should never
# by themselves count as "noise".
_ALLOWED_PUNCTUATION = set(
    ".,;:!?()[]{}'\"-_/\\%&+=<>*#@°ºµ±×÷√∞≈≠≤≥→←↔$€£¥|~^`"
)


@dataclass
class TextQualityResult:
    is_garbled: bool
    reason: str | None
    confidence: float  # heuristic confidence that this text is clean
    char_count: int


def _is_noise_char(ch: str) -> bool:
    if ch.isalnum():  # any script's letters/digits, not just ASCII
        return False
    if ch.isspace():
        return False
    if ch in _ALLOWED_PUNCTUATION:
        return False
    return True


def assess(text: str | None, *, min_chars: int = 20) -> TextQualityResult:
    if text is None:
        return TextQualityResult(True, "empty_extraction", 0.0, 0)

    stripped = text.strip()
    char_count = len(stripped)

    if char_count < min_chars:
        return TextQualityResult(True, "too_short", 0.1, char_count)

    replacement_ratio = stripped.count(_REPLACEMENT_CHAR) / char_count
    if replacement_ratio > 0.01:
        return TextQualityResult(
            True, f"replacement_chars:{replacement_ratio:.3f}", 0.1, char_count
        )

    # Control/unassigned/private-use characters (excluding normal
    # newlines/tabs) almost always mean a broken font or encoding
    # mapping -- real documents don't contain these.
    control_or_unassigned = sum(
        1
        for ch in stripped
        if unicodedata.category(ch) in ("Cc", "Cn", "Co", "Cs")
        and ch not in _ALLOWED_WHITESPACE
    )
    control_ratio = control_or_unassigned / char_count
    if control_ratio > 0.02:
        return TextQualityResult(
            True, f"control_chars:{control_ratio:.3f}", 0.15, char_count
        )

    # Repeated single character spam (e.g. a scan of a mostly-blank
    # page that OCR'd into garbage, or a broken font mapping every
    # glyph to the same byte).
    most_common_char_ratio = max(
        stripped.count(c) for c in set(stripped)
    ) / char_count
    if most_common_char_ratio > 0.4:
        return TextQualityResult(
            True,
            f"repeated_char_spam:{most_common_char_ratio:.2f}",
            0.15,
            char_count,
        )

    # Fraction of characters that are neither letters/digits (any
    # script), whitespace, nor expected exam punctuation/symbols. A
    # high ratio here means actual mojibake/symbol soup -- unlike word
    # density, this doesn't penalize answer keys ("1- b", "2- c" is
    # ~100% letters/digits/punctuation) or option lists.
    noise_count = sum(1 for ch in stripped if _is_noise_char(ch))
    noise_ratio = noise_count / char_count
    if noise_ratio > 0.15:
        return TextQualityResult(
            True, f"high_noise_ratio:{noise_ratio:.2f}", 0.2, char_count
        )

    # Passed all checks. Confidence scales gently with cleanliness and
    # length, capped well below 1.0 -- this is a heuristic gate, not a
    # ground-truth judgment, and should not be confused with a
    # calibrated probability. Treat it as a routing signal only.
    length_factor = min(char_count / 500, 1.0)
    confidence = 0.55 + 0.35 * (1 - noise_ratio) * length_factor
    confidence = min(confidence, 0.9)

    return TextQualityResult(False, None, round(confidence, 3), char_count)


def detect_repeated_block(
    text: str,
    *,
    min_block_chars: int = 60,
    similarity_threshold: float = 0.97,
) -> str | None:
    """Looks for a paragraph-sized chunk of text that reappears
    elsewhere on the same page.

    This exists specifically for vision-model transcriptions: small
    local vision models occasionally loop and re-emit content they
    already produced, so a page comes back as "questions 1-5" followed
    immediately by "questions 1-5" again, verbatim. That's a model
    artifact, not something in the source document -- confirmed by
    testing against a real scanned exam page where this happened
    (that real case scored a similarity of 1.00, an exact repeat).

    The threshold is set high (0.97, near-exact match only) rather
    than "pretty similar", because exam banks legitimately contain
    consecutive questions sharing most of their wording -- e.g. two
    questions with the identical option list testing different joint
    types. Real testing against actual exam content showed such pairs
    scoring ~0.89, which a looser threshold would wrongly flag as
    duplication. Only near-verbatim repeats should trip this.

    Deliberately block-based (split on blank lines), not line-based,
    so it does NOT fire on legitimately repeated short strings like a
    running header/footer ("Basis of Human Anatomy" at the bottom of
    every page) -- those are shorter than min_block_chars and get
    filtered out before comparison.

    Returns a short machine-readable description of what matched, or
    None if nothing looks duplicated. This only flags -- it never
    removes or edits the duplicated content, consistent with the rest
    of this pipeline never silently repairing source data.
    """
    raw_blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    blocks = [b for b in raw_blocks if len(b) >= min_block_chars]

    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            ratio = difflib.SequenceMatcher(None, blocks[i], blocks[j]).ratio()
            if ratio >= similarity_threshold:
                preview = blocks[i][:60].replace("\n", " ")
                return f"blocks_{i}_and_{j}_sim_{ratio:.2f}:'{preview}...'"
    return None

