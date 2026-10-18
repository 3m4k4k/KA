"""
Adaptive question segmentation.

Wraps segmentation.py (unchanged) with a resolve-validate-learn loop:

  1. Try the builtin, hand-written parsers. If their output validates, done:
     no profile lookup, no AI, behaviour identical to plain segmentation.py
     (plus provenance fields).
  2. Otherwise (unfamiliar / ambiguous / failed validation) look for stored
     layout profiles with a similar structural signature. A profile is only
     used if it is applicable to THIS document (its literal headers exist,
     its markers match real lines) and its output validates.
  3. Otherwise ask the AI to describe the layout (layout_ai.py), execute that
     description with the deterministic engine, validate. One repair retry
     with the validator's complaints is allowed.
  4. A validated AI-discovered layout is saved as a *provisional* profile.
  5. Every document is validated every time, profile or not.
  6. If nothing validates, the best candidate is still written, flagged at
     document level (`layout_unresolved`) -- nothing is dropped or repaired.

Usage:
    python adaptive_segmentation.py --input ../output --output ../questions
    python adaptive_segmentation.py --no-ai          # builtin + stored profiles only
    python adaptive_segmentation.py --provider ollama --model phi4-mini
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from schemas import SourceDocument  # noqa: E402
from question_schemas import ExtractionProvenance, LayoutResolution, QuestionDocument  # noqa: E402
import segmentation  # noqa: E402
from layout_spec import LayoutSpec, compile_spec, normalize_literal  # noqa: E402
from layout_features import Line, build_lines, content_lines, extract_features, signature  # noqa: E402
from layout_engine import extract_with_spec  # noqa: E402
from layout_validation import FAIL, PASS, WEAK, Thresholds, ValidationReport, rank_key, validate_extraction  # noqa: E402
from layout_store import LayoutStore, PROVISIONAL, TRUSTED, LifecyclePolicy  # noqa: E402
import layout_ai  # noqa: E402
from layout_ai import AIUnavailable, PROMPT_VERSION, build_digest, build_user_prompt, evaluate_reply  # noqa: E402


@dataclass
class AdaptiveConfig:
    use_ai: bool = True
    force_ai: bool = False  # ignore the proposal cache and ask again
    max_ai_attempts: int = 2
    min_signature_similarity: float = 0.7
    max_profile_candidates: int = 3
    digest_chars: int = 8000
    ai_max_tokens: int = 1800
    thresholds: Thresholds = field(default_factory=Thresholds)


@dataclass
class _Candidate:
    source: str  # builtin | profile | ai
    qdoc: QuestionDocument
    report: ValidationReport
    profile_id: str | None = None
    profile_state: str | None = None
    note: str = ""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def profile_applicable(spec: LayoutSpec, lines: list[Line]) -> tuple[bool, str]:
    """Cheap precondition for a stored profile: its own literals must exist in
    this document. Failing this means "not applicable", which is NOT counted
    as a failure of the profile."""
    cspec = compile_spec(spec)
    body = content_lines(lines)
    present = {normalize_literal(ln.text) for ln in body if len(ln.text) <= 80}
    for h in cspec.header_kinds:
        if h not in present:
            return False, f"header {h!r} not in document"
    if sum(1 for ln in body if cspec.stem_re.match(ln.text)) < 3:
        return False, "stem marker matches fewer than 3 lines"
    if cspec.option_re is not None and sum(1 for ln in body if cspec.option_re.match(ln.text)) < 6:
        return False, "option marker matches fewer than 6 lines"
    return True, ""


class AdaptiveSegmenter:
    def __init__(self, store: LayoutStore, llm: layout_ai.LLMClient | None = None, config: AdaptiveConfig | None = None):
        self.store = store
        self.llm = llm
        self.cfg = config or AdaptiveConfig()
        self.ai_disabled_reason: str | None = None if llm is not None else "no AI client configured"
        self.stats = {
            "builtin": 0, "profile": 0, "ai_discovered": 0, "unresolved": 0,
            "ai_calls": 0, "ai_cache_hits": 0, "ai_avoided_by_profile": 0,
        }

    # -- public -------------------------------------------------------------

    def segment(self, doc: SourceDocument) -> QuestionDocument:
        lines = build_lines(doc)
        feats = extract_features(lines)
        sig = signature(feats)
        th = self.cfg.thresholds
        cands: list[_Candidate] = []

        # 1. builtin (fast path)
        try:
            bq = segmentation.segment_document(doc)
            brep = validate_extraction(bq, lines, feats, th)
        except Exception as exc:  # noqa: BLE001 -- a parser crash is a failed candidate
            traceback.print_exc()
            bq = QuestionDocument(document_id=doc.document_id, original_filename=doc.original_filename,
                                  subject=doc.subject, topic=doc.topic, document_anomalies=[f"builtin_parser_crashed:{exc}"])
            bq.compute_rollup()
            brep = ValidationReport(FAIL, 0.0, [])
        builtin = _Candidate("builtin", bq, brep)
        cands.append(builtin)
        if brep.verdict == PASS:
            self.stats["builtin"] += 1
            return self._finalize(builtin, cands, doc)

        # 2. stored profiles
        for prof, sim in self.store.match(sig, min_similarity=self.cfg.min_signature_similarity, top_k=self.cfg.max_profile_candidates):
            spec = self.store.spec_of(prof)
            if spec is None:
                continue
            ok, why = profile_applicable(spec, lines)
            if not ok:
                cands.append(_Candidate("profile", bq, ValidationReport(FAIL, 0.0, []), prof["profile_id"], self.store.state(prof), f"not applicable: {why}"))
                cands[-1].report = ValidationReport("n/a", -1.0, [])
                continue
            cand = self._run_profile(doc, lines, feats, sig, prof, spec)
            cands.append(cand)
            if cand.report.verdict == PASS:
                self.stats["profile"] += 1
                if self.cfg.use_ai and self.llm is not None:
                    self.stats["ai_avoided_by_profile"] += 1
                return self._finalize(cand, cands, doc)

        # 3. AI discovery
        ai_meta = {"attempts": 0, "cache_hit": False, "model": None, "note": None}
        if self.cfg.use_ai:
            cand = self._discover(doc, lines, feats, sig, builtin, ai_meta)
            if cand is not None:
                cands.append(cand)
                if cand.report.verdict == PASS:
                    self.stats["ai_discovered"] += 1
                    return self._finalize(cand, cands, doc, ai_meta)

        # 4. nothing validated: best candidate, flagged
        real = [c for c in cands if c.report.verdict in (PASS, WEAK, FAIL)]
        best = max(real, key=lambda c: (rank_key(c.report.verdict, c.report.score), c.source == "builtin"))
        self.stats["unresolved"] += 1
        return self._finalize(best, cands, doc, ai_meta, unresolved=True)

    # -- candidates ---------------------------------------------------------

    def _run_profile(self, doc, lines, feats, sig, prof, spec) -> _Candidate:
        cspec = compile_spec(spec)
        qd = extract_with_spec(doc, lines, cspec, method="layout_profile", profile_id=prof["profile_id"])
        rep = validate_extraction(qd, lines, feats, self.cfg.thresholds)
        # PASS counts as a success, FAIL as a failure of this profile; WEAK neither.
        if rep.verdict in (PASS, FAIL):
            prof = self.store.record_outcome(prof["profile_id"], file_hash=doc.file_hash, document_id=doc.document_id,
                                             verdict=rep.verdict, score=rep.score, n_questions=len(qd.questions), signature=sig)
        state = self.store.state(prof)
        for q in qd.questions:
            if q.extraction:
                q.extraction.profile_state = state
        return _Candidate("profile", qd, rep, prof["profile_id"], state)

    def _discover(self, doc, lines, feats, sig, builtin: _Candidate, meta: dict) -> _Candidate | None:
        key = hashlib.sha1(f"{doc.file_hash}|{PROMPT_VERSION}".encode()).hexdigest()[:20]
        cached = None if self.cfg.force_ai else self.store.load_proposals(key)
        digest = build_digest(lines, feats, self.cfg.digest_chars)
        feedback: list[str] | None = None
        if builtin.report.checks:
            feedback = None  # builtin context goes in the first prompt, see below
        attempts: list[dict] = list(cached or [])
        best: _Candidate | None = None
        replay = cached is not None
        meta["cache_hit"] = replay
        if replay:
            self.stats["ai_cache_hits"] += 1

        n = 0
        previous = None
        while True:
            if replay:
                if n >= len(attempts):
                    break
                raw, model = attempts[n]["reply"], attempts[n].get("model")
            else:
                if n >= self.cfg.max_ai_attempts:
                    break
                if self.ai_disabled_reason:
                    meta["note"] = f"ai unavailable: {self.ai_disabled_reason}"
                    break
                prompt_digest = digest
                if n == 0 and builtin.report.checks:
                    prompt_digest += "\n\nNOTE: the existing hand-written parser could not handle this document: " + builtin.report.summary()[:400]
                try:
                    reply = self.llm.complete(layout_ai.SYSTEM_PROMPT, build_user_prompt(prompt_digest, feedback, previous), max_tokens=self.cfg.ai_max_tokens)
                except AIUnavailable as exc:
                    self.ai_disabled_reason = str(exc)
                    meta["note"] = f"ai unavailable: {exc}"
                    break
                self.stats["ai_calls"] += 1
                raw, model = reply.text, reply.model
                attempts.append({"reply": raw, "model": model})
            n += 1
            meta["attempts"] = n
            meta["model"] = model

            ev = evaluate_reply(raw, lines)
            if ev.unexpressible:
                meta["note"] = ev.problems[0]
                break
            previous = ev.raw_data
            if ev.spec is None or ev.problems:
                feedback = ev.problems or ["spec rejected"]
                continue

            qd = extract_with_spec(doc, lines, ev.cspec, method="ai_discovered_layout", profile_id=ev.spec.profile_id())
            rep = validate_extraction(qd, lines, feats, self.cfg.thresholds)
            cand = _Candidate("ai", qd, rep, ev.spec.profile_id(), None)
            cand.note = rep.summary()
            if best is None or rank_key(rep.verdict, rep.score) > rank_key(best.report.verdict, best.report.score):
                best = cand
                best_spec = ev.spec
            if rep.verdict == PASS:
                self._learn(doc, sig, ev.spec, rep, qd, model, cand)
                break
            feedback = [f"extraction validated as {rep.verdict}: {rep.summary()}"]
            if qd.questions:
                feedback.append(f"it produced {len(qd.questions)} questions; first question text: {qd.questions[0].question_text[:80]!r}")

        if not replay and attempts:
            self.store.save_proposals(key, attempts)
        return best

    def _learn(self, doc, sig, spec: LayoutSpec, rep, qd, model, cand: _Candidate) -> None:
        pid = spec.profile_id()
        existed = self.store.get(pid) is not None
        prof = self.store.save_new(spec, sig, origin={"method": "ai", "model": model, "discovered_on": doc.file_hash, "prompt_version": PROMPT_VERSION})
        prof = self.store.record_outcome(pid, file_hash=doc.file_hash, document_id=doc.document_id, verdict=PASS, score=rep.score, n_questions=len(qd.questions), signature=sig)
        cand.profile_state = self.store.state(prof)
        cand.profile_id = pid
        for q in qd.questions:
            if q.extraction:
                q.extraction.profile_state = cand.profile_state

    # -- output -------------------------------------------------------------

    def _finalize(self, chosen: _Candidate, cands: list[_Candidate], doc, ai_meta: dict | None = None, unresolved: bool = False) -> QuestionDocument:
        qd = chosen.qdoc
        ai_meta = ai_meta or {"attempts": 0, "cache_hit": False, "model": None, "note": None}
        method = "unresolved" if unresolved else {"builtin": "builtin", "profile": "layout_profile", "ai": "ai_discovered_layout"}[chosen.source]
        if chosen.source == "builtin":
            for q in qd.questions:
                q.extraction = ExtractionProvenance(method="builtin")
        notes = []
        if ai_meta.get("note"):
            notes.append(ai_meta["note"])
        res = LayoutResolution(
            method=method,
            profile_id=chosen.profile_id if chosen.source != "builtin" else None,
            profile_state=chosen.profile_state,
            validation_verdict=chosen.report.verdict,
            validation_score=round(chosen.report.score, 3),
            checks=chosen.report.to_dicts(),
            considered=[
                {"source": c.source, "profile_id": c.profile_id, "verdict": c.report.verdict, "score": round(c.report.score, 3), "note": c.note or c.report.summary()[:200]}
                for c in cands
            ],
            ai_model=ai_meta.get("model"),
            ai_attempts=ai_meta.get("attempts", 0),
            ai_cache_hit=ai_meta.get("cache_hit", False),
            notes=notes,
            resolved_at=_now(),
        )
        qd.layout_resolution = res
        # document-level flags (never per-question: that would drag every
        # question of a freshly learned layout into needs_review)
        if unresolved:
            qd.document_anomalies.append(f"layout_unresolved:{chosen.report.verdict}:{chosen.report.summary()[:160]}")
        elif chosen.source != "builtin":
            if chosen.profile_state == PROVISIONAL:
                qd.document_anomalies.append(f"layout_profile_provisional:{chosen.profile_id}")
        if chosen.source == "builtin" and chosen.report.verdict == WEAK:
            qd.document_anomalies.append(f"layout_weak_validation:{chosen.report.summary()[:160]}")
        qd.compute_rollup()
        return qd


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _make_client(args):
    if args.no_ai or args.provider == "none":
        return None
    if args.provider == "ollama":
        return layout_ai.OllamaClient(args.model or "phi4-mini")
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        print("GROQ_API_KEY not set -- running without AI (builtin parsers + stored profiles only).")
        return None
    from groq_router import GroqRouter

    return layout_ai.GroqClient(GroqRouter(key, max_wait_s=args.max_wait))


def main() -> None:
    ap = argparse.ArgumentParser(description="Segment SourceDocuments into QuestionDocuments, adapting to unseen layouts.")
    ap.add_argument("--input", default="../output", type=Path)
    ap.add_argument("--output", default="../questions", type=Path)
    ap.add_argument("--store", default="../layout_profiles", type=Path)
    ap.add_argument("--provider", choices=["groq", "ollama", "none"], default="groq")
    ap.add_argument("--model", help="Ollama model name (provider=ollama)")
    ap.add_argument("--no-ai", action="store_true", help="Builtin parsers + stored profiles only.")
    ap.add_argument("--force-ai", action="store_true", help="Ignore cached AI proposals and ask again.")
    ap.add_argument("--max-attempts", type=int, default=2)
    ap.add_argument("--max-wait", type=float, default=120.0)
    args = ap.parse_args()

    in_root, out_root = args.input.resolve(), args.output.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    paths = sorted(in_root.glob("DOC-*.json"))
    if not paths:
        print(f"No SourceDocuments found in {in_root}. Run main.py first.")
        return

    cfg = AdaptiveConfig(use_ai=not (args.no_ai or args.provider == "none"), force_ai=args.force_ai, max_ai_attempts=args.max_attempts)
    seg = AdaptiveSegmenter(LayoutStore(args.store.resolve()), _make_client(args), cfg)
    if seg.llm is None:
        cfg.use_ai = False

    for p in paths:
        doc = SourceDocument(**json.loads(p.read_text()))
        qd = seg.segment(doc)
        (out_root / f"{doc.document_id}.questions.json").write_text(json.dumps(qd.model_dump(), indent=2, default=str))
        r = qd.layout_resolution
        tag = f"{r.method}" + (f" {r.profile_id} ({r.profile_state})" if r.profile_id else "")
        print(f"[{tag}] {doc.original_filename} -> {len(qd.questions)} questions, validation={r.validation_verdict} {r.validation_score}"
              + (f"  FLAGGED: {qd.document_anomalies[-1][:100]}" if r.method == "unresolved" else ""))

    print("\n--- Adaptive segmentation summary ---")
    for k, v in seg.stats.items():
        print(f"{k}: {v}")
    if seg.ai_disabled_reason and cfg.use_ai:
        print(f"AI stopped: {seg.ai_disabled_reason}  (re-run later; proposals already fetched are cached)")


if __name__ == "__main__":
    main()
