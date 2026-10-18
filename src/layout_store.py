"""
Persistent layout-profile store.

    layout_profiles/
      profiles/LP-<hash>.json     one file per profile (content-addressed)
      proposals/<key>.json        raw AI replies per document (so re-runs
                                  never re-bill for the same document)

Profile identity is the hash of the *canonical spec* (no evidence refs, no
notes), so the same layout discovered twice is one profile -- and the second
discovery on a different document counts as corroboration, not a duplicate.

Lifecycle -- derived from recorded evidence, never hand-set:

    provisional  validated (PASS) on 1 distinct document, i.e. only the one
                 it was learned from. Reused to save AI calls; documents
                 extracted with it are flagged at document level.
    validated    PASS on >= `validated_after` (2) distinct documents.
    trusted      PASS on >= `trusted_after` (3) distinct documents with a
                 failure rate <= 10%. No document-level flag.
    quarantined  more failing documents than passing (and >= 3 failures), or
                 quarantined by hand. Never matched again.

"Distinct document" = distinct file_hash (sha256 of the raw file), and the
latest outcome per document wins -- so re-running the pipeline on the same
files can never promote a profile, and a document that was failing and is
later fixed stops counting against it. Any observed failure removes
`trusted` until the failure rate is back under 10%.

Only a PASS verdict counts as a success and only a FAIL verdict as a
failure; WEAK is recorded but counts neither way.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from layout_spec import LayoutSpec  # noqa: E402
from layout_features import similarity  # noqa: E402

PROVISIONAL, VALIDATED, TRUSTED, QUARANTINED = "provisional", "validated", "trusted", "quarantined"
_STATE_ORDER = {TRUSTED: 3, VALIDATED: 2, PROVISIONAL: 1, QUARANTINED: 0}


@dataclass
class LifecyclePolicy:
    validated_after: int = 2
    trusted_after: int = 3
    trusted_max_failure_rate: float = 0.10
    quarantine_min_failures: int = 3


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def derive_state(profile: dict, policy: LifecyclePolicy) -> str:
    if profile.get("manual_quarantine"):
        return QUARANTINED
    outcomes = profile.get("documents", {})
    passed = sum(1 for o in outcomes.values() if o["verdict"] == "pass")
    failed = sum(1 for o in outcomes.values() if o["verdict"] == "fail")
    if failed >= policy.quarantine_min_failures and failed > passed:
        return QUARANTINED
    if passed >= policy.trusted_after and failed <= int(policy.trusted_max_failure_rate * (passed + failed)):
        return TRUSTED
    if passed >= policy.validated_after and failed < passed:
        return VALIDATED
    return PROVISIONAL


class LayoutStore:
    def __init__(self, root: Path, policy: LifecyclePolicy | None = None, now_fn=_now):
        self.root = Path(root)
        self.policy = policy or LifecyclePolicy()
        self.now = now_fn
        (self.root / "profiles").mkdir(parents=True, exist_ok=True)
        (self.root / "proposals").mkdir(parents=True, exist_ok=True)

    # -- io -----------------------------------------------------------------

    def _path(self, profile_id: str) -> Path:
        return self.root / "profiles" / f"{profile_id}.json"

    def _write(self, path: Path, data: dict) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True))
        tmp.replace(path)

    def get(self, profile_id: str) -> dict | None:
        p = self._path(profile_id)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return None  # a corrupt profile is skipped, never fatal

    def list_profiles(self) -> list[dict]:
        out = []
        for p in sorted((self.root / "profiles").glob("LP-*.json")):
            d = self.get(p.stem)
            if d:
                out.append(d)
        return out

    def spec_of(self, profile: dict) -> LayoutSpec | None:
        try:
            return LayoutSpec(**profile["spec"])
        except Exception:  # noqa: BLE001 -- schema drift: ignore profile, don't crash
            return None

    def state(self, profile: dict) -> str:
        return derive_state(profile, self.policy)

    # -- matching -----------------------------------------------------------

    def match(self, signature: dict, *, min_similarity: float = 0.7, top_k: int = 3) -> list[tuple[dict, float]]:
        """Candidates by structural similarity only. Best state first, then
        similarity. A candidate is NOT a match until it validates."""
        found = []
        for prof in self.list_profiles():
            st = self.state(prof)
            if st == QUARANTINED:
                continue
            sim = max((similarity(signature, s) for s in prof.get("signatures", [])), default=0.0)
            if sim >= min_similarity:
                found.append((prof, sim, st))
        found.sort(key=lambda t: (-_STATE_ORDER[t[2]], -t[1], t[0]["profile_id"]))
        return [(p, s) for p, s, _ in found[:top_k]]

    # -- recording ----------------------------------------------------------

    def save_new(self, spec: LayoutSpec, signature: dict, *, origin: dict) -> dict:
        """Create the profile if its canonical spec is new; otherwise return
        the existing one (identity = canonical spec hash)."""
        pid = spec.profile_id()
        existing = self.get(pid)
        if existing:
            return existing
        prof = {
            "profile_id": pid,
            "schema_version": spec.schema_version,
            "created_at": self.now(),
            "origin": origin,
            "spec": spec.stripped().model_dump(mode="json"),
            "signatures": [signature],
            "documents": {},
            "manual_quarantine": False,
        }
        self._write(self._path(pid), prof)
        return prof

    def record_outcome(self, profile_id: str, *, file_hash: str, document_id: str, verdict: str, score: float, n_questions: int, signature: dict) -> dict:
        prof = self.get(profile_id)
        if prof is None:
            raise KeyError(profile_id)
        prof["documents"][file_hash] = {
            "document_id": document_id,
            "verdict": verdict,
            "score": round(score, 3),
            "questions": n_questions,
            "at": self.now(),
        }
        if verdict == "pass" and signature not in prof["signatures"]:
            prof["signatures"].append(signature)
            prof["signatures"] = prof["signatures"][-8:]
        prof["updated_at"] = self.now()
        prof["state"] = derive_state(prof, self.policy)  # informational copy
        self._write(self._path(profile_id), prof)
        return prof

    def set_quarantine(self, profile_id: str, value: bool) -> dict:
        prof = self.get(profile_id)
        if prof is None:
            raise KeyError(profile_id)
        prof["manual_quarantine"] = value
        prof["state"] = derive_state(prof, self.policy)
        self._write(self._path(profile_id), prof)
        return prof

    # -- AI proposal cache --------------------------------------------------

    def load_proposals(self, key: str) -> list[dict] | None:
        p = self.root / "proposals" / f"{key}.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text()).get("attempts")
        except (json.JSONDecodeError, OSError):
            return None

    def save_proposals(self, key: str, attempts: list[dict]) -> None:
        self._write(self.root / "proposals" / f"{key}.json", {"attempts": attempts, "saved_at": self.now()})


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Inspect / manage the layout-profile store.")
    ap.add_argument("--store", default="../layout_profiles", type=Path)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--show")
    ap.add_argument("--quarantine")
    ap.add_argument("--reinstate")
    a = ap.parse_args()
    st = LayoutStore(a.store)
    if a.quarantine:
        print(st.set_quarantine(a.quarantine, True)["state"])
    elif a.reinstate:
        print(st.set_quarantine(a.reinstate, False)["state"])
    elif a.show:
        print(json.dumps(st.get(a.show), indent=2))
    else:
        for p in st.list_profiles():
            docs = p["documents"]
            print(f"{p['profile_id']}  {st.state(p):<11} pass={sum(d['verdict']=='pass' for d in docs.values())} fail={sum(d['verdict']=='fail' for d in docs.values())}  origin={p['origin'].get('method')}")


if __name__ == "__main__":
    main()
