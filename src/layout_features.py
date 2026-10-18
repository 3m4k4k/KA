"""
Line stream + structural features of a SourceDocument.

Two jobs, both purely deterministic (no AI, no I/O):

1. `build_lines` turns a SourceDocument into a list of `Line`s that carry a
   stable reference ("p7:12" = page 7, 12th non-empty line) and a flag for
   running headers/footers. Furniture is detected by segmentation.py's own
   frequency logic (not re-implemented), so "what counts as furniture" has
   exactly one definition in the pipeline.

2. `extract_features` / `signature` / `similarity` describe the *shape* of a
   document: which stem style dominates ("1-", "Q1.", "(1)"...), which
   option style, whether numbering restarts, whether there is a key block
   or per-question answer lines, and so on. A layout profile is matched to a
   new document by comparing these structural signatures -- never by file
   name, course, folder or content hash. A signature match is only a
   *candidate*; layout_validation.py decides whether the profile really
   fits the document.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from schemas import SourceDocument  # noqa: E402
import segmentation as _seg  # noqa: E402
from layout_spec import (  # noqa: E402
    OptionMarker,
    QuestionMarker,
    normalize_literal,
    option_regex,
    stem_regex,
)


@dataclass(frozen=True)
class Line:
    page: int
    idx: int  # 1-based among the page's non-empty lines, before furniture removal
    text: str
    furniture: bool = False

    @property
    def ref(self) -> str:
        return f"p{self.page}:{self.idx}"


def parse_ref(ref: str) -> tuple[int, int] | None:
    m = re.fullmatch(r"\s*p(\d+):(\d+)\s*", str(ref))
    return (int(m.group(1)), int(m.group(2))) if m else None


def build_lines(doc: SourceDocument) -> list[Line]:
    """All non-empty lines of every page in page order. Lines removed by
    segmentation's furniture stripping are kept but flagged, so refs stay
    stable and the AI/validators can still see (and refuse to cite) them."""
    cleaned = _seg._clean_page_lines(doc)
    out: list[Line] = []
    for p in sorted(doc.pages, key=lambda p: p.page):
        if not p.text:
            continue
        raw = [ln.strip() for ln in p.text.split("\n")]
        raw = [ln for ln in raw if ln]
        kept = cleaned.get(p.page, [])
        j = 0
        for i, text in enumerate(raw, start=1):
            if j < len(kept) and kept[j] == text:
                out.append(Line(p.page, i, text, False))
                j += 1
            elif is_inline_answer_line(text) or is_key_like_line(text) or _KEY_FRAGMENT_RE.match(text):
                # segmentation's frequency-based furniture stripper can't tell
                # a repeated "Answer: b" / "3-a" line from a running footer
                # (there are only a handful of distinct letters). Those lines
                # are content, so rescue them here; segmentation.py itself is
                # left untouched.
                out.append(Line(p.page, i, text, False))
            else:
                out.append(Line(p.page, i, text, True))
    return out


def content_lines(lines: list[Line]) -> list[Line]:
    return [ln for ln in lines if not ln.furniture]


# ---------------------------------------------------------------------------
# Patterns used for *description* (counting what a document looks like)
# ---------------------------------------------------------------------------

_KEYLINE_RE = re.compile(r"^[\s|:\-.&,a-fA-F0-9)(]+$")
_KEY_TOKEN_ANY = re.compile(r"(?<![\w])\d{1,3}\s*[-.):|]?\s*[a-fA-F](?:\s*[&,]\s*[a-fA-F])*(?![\w])")
_INLINE_ANSWER_RE = re.compile(
    r"^(?:correct\s+answer|answer|ans|key)\s*[:.\-=)]*\s*\(?[a-fA-F](?:\s*[&,]\s*[a-fA-F])*\)?\.?\s*$", re.I
)
_KEY_HEADER_RE = re.compile(r"^(?:model\s+)?(?:answers?|answer\s+key|key|solutions?)(?:\s+sheet)?$", re.I)
_KIND_HINTS = {
    "mcq": re.compile(r"m\.?\s*c\.?\s*q|multiple\s+choice|choose", re.I),
    "fill_in_blank": re.compile(r"fill\s+in|blank|complete", re.I),
    "matching": re.compile(r"match", re.I),
}
# A key entry split across lines: a bare letter ("c", "a & c") or a dangling "50-".
_KEY_FRAGMENT_RE = re.compile(r"^(?:[a-fA-F](?:\s*[&,]\s*[a-fA-F])*|\d{1,3}\s*[-.):|])$")
_MARK_SYMBOLS = ("*", "\u2713", "\u2714", "(correct)")


def is_key_like_line(text: str) -> bool:
    """A line made only of key entries ("12-b", "| 3-a | 4-c |", "5 D")."""
    return bool(
        _KEYLINE_RE.match(text)
        and any(ch.isdigit() for ch in text)
        and any(ch.isalpha() for ch in text)
        and _KEY_TOKEN_ANY.search(text)
    )


def key_token_count(text: str) -> int:
    return len(_KEY_TOKEN_ANY.findall(text))


def is_inline_answer_line(text: str) -> bool:
    return bool(_INLINE_ANSWER_RE.match(text))


def is_key_header(text: str) -> bool:
    return bool(_KEY_HEADER_RE.match(normalize_literal(text)))


# Candidate marker styles enumerated for counting. Single closer each, so a
# catch-all style can't win by matching everything.
def _candidate_stems() -> list[tuple[str, QuestionMarker]]:
    out = []
    for prefix in ("", "Q", "Question"):
        for op, closers in (("", "-"), ("", "."), ("", ")"), ("", ":"), ("(", ")"), ("[", "]")):
            m = QuestionMarker(prefix=prefix, open=op, closers=closers)
            out.append((f"{prefix}|{op}|{closers}", m))
    return out


def _candidate_options() -> list[tuple[str, OptionMarker]]:
    out = []
    for case in ("lower", "upper"):
        for op, closers in (("", "."), ("", ")"), ("", "-"), ("", ":"), ("(", ")")):
            out.append((f"{case}|{op}|{closers}", OptionMarker(case=case, open=op, closers=closers)))
    return out


_STEM_CANDIDATES = None
_OPT_CANDIDATES = None


def _candidates():
    global _STEM_CANDIDATES, _OPT_CANDIDATES
    if _STEM_CANDIDATES is None:
        _STEM_CANDIDATES = [(i, m, stem_regex(m)) for i, m in _candidate_stems()]
        _OPT_CANDIDATES = [(i, m, option_regex(m)) for i, m in _candidate_options()]
    return _STEM_CANDIDATES, _OPT_CANDIDATES


@dataclass
class LayoutFeatures:
    n_pages: int = 0
    n_lines: int = 0
    stem_style: str = ""
    stem_count: int = 0
    stem_runner_up: int = 0
    opt_style: str = ""
    opt_count: int = 0
    restarts: int = 0
    key_like_lines: int = 0
    key_tokens: int = 0
    key_header_lines: int = 0
    inline_answer_lines: int = 0
    marked_option_lines: int = 0
    marked_option_symbol: str = ""
    heading_like_distinct: int = 0
    kind_header_hits: dict = field(default_factory=dict)
    header_candidates: list = field(default_factory=list)  # (norm text, count, first ref)

    @property
    def opts_per_stem(self) -> float:
        return self.opt_count / self.stem_count if self.stem_count else 0.0

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["opts_per_stem"] = round(self.opts_per_stem, 2)
        return d


def extract_features(lines: list[Line]) -> LayoutFeatures:
    body = content_lines(lines)
    f = LayoutFeatures(n_pages=len({ln.page for ln in body}), n_lines=len(body))
    stems, opts = _candidates()

    keylike = [ln for ln in body if is_key_like_line(ln.text)]
    key_ids = {id(ln) for ln in keylike}
    f.key_like_lines = len(keylike)
    f.key_tokens = sum(key_token_count(ln.text) for ln in keylike)
    f.inline_answer_lines = sum(1 for ln in body if is_inline_answer_line(ln.text))
    f.key_header_lines = sum(1 for ln in body if is_key_header(ln.text))

    non_key = [ln for ln in body if id(ln) not in key_ids]

    # dominant stem style
    best = ("", 0, 0)
    counts = []
    for sid, _m, rx in stems:
        n = sum(1 for ln in non_key if rx.match(ln.text))
        counts.append((n, sid))
    counts.sort(key=lambda t: (-t[0], len(t[1]), t[1]))
    if counts and counts[0][0] > 0:
        f.stem_style, f.stem_count = counts[0][1], counts[0][0]
        f.stem_runner_up = counts[1][0] if len(counts) > 1 else 0
        rx = next(r for sid, _m, r in stems if sid == f.stem_style)
        prev = None
        for ln in non_key:
            m = rx.match(ln.text)
            if m:
                num = int(m.group(1))
                if num == 1 and prev is not None and prev > 1:
                    f.restarts += 1
                prev = num

    # dominant option style
    ocounts = []
    for oid, _m, rx in opts:
        n = sum(1 for ln in non_key if rx.match(ln.text))
        ocounts.append((n, oid))
    ocounts.sort(key=lambda t: (-t[0], t[1]))
    if ocounts and ocounts[0][0] > 0:
        f.opt_style, f.opt_count = ocounts[0][1], ocounts[0][0]
        orx = next(r for oid, _m, r in opts if oid == f.opt_style)
        sym = Counter()
        for ln in non_key:
            if orx.match(ln.text) or orx.match(_strip_marks(ln.text)[0]):
                for s in _MARK_SYMBOLS:
                    t = ln.text.rstrip()
                    if t.endswith(s) or t.lstrip().startswith(s):
                        sym[s] += 1
        if sym:
            f.marked_option_symbol, f.marked_option_lines = sym.most_common(1)[0]

    # headings and header candidates
    heads = Counter()
    first_ref = {}
    for ln in non_key:
        t = ln.text
        if len(t) > 40 or any(ch.isdigit() for ch in t):
            continue
        if any(rx.match(t) for _i, _m, rx in stems[:6]) or any(rx.match(t) for _i, _m, rx in opts[:3]):
            continue
        n = normalize_literal(t)
        if not n:
            continue
        heads[n] += 1
        first_ref.setdefault(n, ln.ref)
    hl = {normalize_literal(ln.text) for ln in non_key if _seg._looks_like_heading(ln.text) and not is_key_header(ln.text)}
    f.heading_like_distinct = len(hl)
    for kind, rx in _KIND_HINTS.items():
        f.kind_header_hits[kind] = sum(c for n, c in heads.items() if rx.search(n) and len(n.split()) <= 5)
    cands = []
    for n, c in heads.items():
        if is_key_header(n) or any(rx.search(n) for rx in _KIND_HINTS.values()) or n in hl or c >= 2:
            cands.append((n, c, first_ref[n]))
    cands.sort(key=lambda t: (-t[1], t[2]))
    f.header_candidates = cands[:14]
    return f


def _strip_marks(text: str) -> tuple[str, str]:
    t = text.rstrip()
    for s in _MARK_SYMBOLS:
        if t.endswith(s):
            return t[: -len(s)].rstrip(), s
        if t.lstrip().startswith(s):
            return t.lstrip()[len(s):].lstrip(), s
    return text, ""


# ---------------------------------------------------------------------------
# Signature + similarity
# ---------------------------------------------------------------------------


def _bucket(n: float, edges: list[float], labels: list[str]) -> str:
    for e, lab in zip(edges, labels):
        if n <= e:
            return lab
    return labels[-1]


def signature(f: LayoutFeatures) -> dict:
    """Quantised structural description. Coarse on purpose: two documents
    from the same source family should produce the same signature even
    though their page counts, chapter names and question counts differ."""
    n_q = max(f.stem_count, 1)
    return {
        "stem": f.stem_style or "none",
        "opt": f.opt_style or "none",
        "restarts": _bucket(f.restarts, [0, 2], ["0", "1-2", "3+"]) if f.stem_count else "na",
        "opts_per_stem": _bucket(f.opts_per_stem, [0.5, 3.5, 4.5, 5.5], ["0", "1-3", "4", "5", "6+"]),
        "key_header": "1" if f.key_header_lines else "0",
        "key_like": _bucket(f.key_tokens / n_q, [0.05, 0.5], ["0", "low", "high"]),
        "inline_ans": _bucket(f.inline_answer_lines / n_q, [0.05, 0.5], ["0", "some", "most"]),
        "marked_opt": f.marked_option_symbol or ("0" if not f.marked_option_lines else "?"),
        "headings": _bucket(f.heading_like_distinct, [0, 3], ["0", "few", "many"]),
    }


_SIG_WEIGHTS = {
    "stem": 3.0,
    "opt": 2.0,
    "key_header": 1.5,
    "key_like": 1.5,
    "inline_ans": 1.0,
    "marked_opt": 1.0,
    "restarts": 1.0,
    "opts_per_stem": 1.0,
    "headings": 0.5,
}


def similarity(a: dict, b: dict) -> float:
    total = sum(_SIG_WEIGHTS.values())
    got = sum(w for k, w in _SIG_WEIGHTS.items() if a.get(k) == b.get(k))
    return got / total
