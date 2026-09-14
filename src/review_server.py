"""
Human-review stage -- Layer 4.

Serves a small local web page for reviewing the `needs_review` bucket
computed by ai_parser.py's routing rule, with the original scanned
page image shown next to the parsed question -- important because a
real chunk of needs_review entries here are vision-transcription
issues on the scanned document, where "does this parsed text actually
match the page" is a question a human can only answer by looking at
the page.

This stage is deliberately NOT another parsing/AI pass. It doesn't
re-derive question_text/options/matching_pairs/source_answer -- those
stay exactly as segmentation produced them, permanently, for audit.
What it writes are two independent kinds of human decision, tracked
with their own timestamp each (a reviewer might resolve one without
having an opinion on the other yet):

    verified_answer / verified_at / verification_note
        "is the answer correct" -- verified_answer is already reserved
        on QuestionRecord from Layer 3 onward, always null until now.

    duplicate_of_question_id / verified_question_text /
    verified_options / verified_matching_pairs / structural_reviewed_at
        "is the record itself well-formed" -- structural corrections,
        added for the case where a question's *text or options* are
        broken (duplicated, garbled, mis-merged), not its answer.
        duplicate_of_question_id points at another question_id
        (possibly in a different document) that this one is a repeat
        of; the "how many times has this been duplicated" count is
        never stored -- it's computed fresh from these pointers every
        time the queue is built (see ReviewStore._duplicate_components),
        so it can't drift when a third copy turns up later.
        verified_question_text/verified_options/verified_matching_pairs
        hold a reviewer's correction ALONGSIDE the untouched original
        -- same non-destructive discipline as verified_answer next to
        source_answer.

Nothing else on the record is touched. Writes go to the SAME canonical
per-document QuestionDocument JSON files that ai_parser.py writes to
(--reviewed-dir, matching ai_parser.py's --output) -- never to the
ephemeral _needs_review.json bucket view, which is just a computed
snapshot and gets regenerated from scratch by ai_parser.py on its next
run anyway.

Routing logic (which questions are "in the queue") is imported
directly from ai_parser.route() rather than re-implemented here, so
there is exactly one place in the whole pipeline that decides
auto_import vs needs_review.

Idempotent like every other stage: a question with verified_at already
set is left out of the queue on a normal run (skip already-verified,
matching the pattern in ollama_vision_fill.py / ai_parser.py) unless
--force is passed, which re-shows every needs_review question
regardless of prior verification -- useful for a second review pass,
not a routine re-run.

Image lookup: QuestionRecord doesn't carry rendered_image_path itself
(that lives on SourceDocument/PageExtraction, from the ingestion
layer) -- source_spans gives page numbers, and this stage cross-
references --source-dir/<document_id>.json (main.py's output) to find
each page's rendered_image_path, if any. Pages that extracted cleanly
from a text layer have no render at all -- the UI shows "no page image
available" rather than erroring, since that's expected and correct for
clean-text documents.

Why a bespoke http.server instead of a framework: this is a single-
user local tool run by one person reviewing their own files, not a
deployed service -- no auth, no concurrency concerns beyond "don't
corrupt a JSON file if two writes race", which a simple lock handles.
Keeping it stdlib-only means no new dependency for something this
small.

Usage:
    python review_server.py --reviewed-dir ../reviewed --source-dir ../output
    # then open http://localhost:8765 in a browser

Flags:
    --port           default 8765
    --threshold      must match the threshold ai_parser.py was run
                      with, so the queue reflects the same
                      auto_import/needs_review split (default 0.75,
                      same default as ai_parser.py)
    --force          include already-verified questions in the queue too
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

from question_schemas import (  # noqa: E402
    MatchingPair,
    Option,
    QuestionDocument,
    QuestionRecord,
)
from schemas import SourceDocument  # noqa: E402
from ai_parser import route  # noqa: E402  -- single source of truth for routing


def _options_key(opts: list[Option] | None) -> tuple | None:
    if opts is None:
        return None
    return tuple((o.label, o.text) for o in opts)


def _matching_pairs_key(pairs: list[MatchingPair] | None) -> tuple | None:
    if pairs is None:
        return None
    return tuple(
        (p.left_label, p.left_text, p.right_label, p.right_text) for p in pairs
    )

STATIC_DIR = Path(__file__).resolve().parent / "review_static"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ReviewStore:
    """In-memory index over the reviewed QuestionDocuments and their
    source SourceDocuments, with disk as the durable copy. Loaded once
    at startup; every verification write updates memory AND rewrites
    the owning document's JSON file immediately, so a crash never loses
    more than the single in-flight write, and there is never a moment
    where memory and disk disagree except mid-write (guarded by
    `_lock`).
    """

    def __init__(self, reviewed_dir: Path, source_dir: Path, threshold: float, force: bool):
        self.reviewed_dir = reviewed_dir
        self.source_dir = source_dir
        self.threshold = threshold
        self.force = force
        self._lock = threading.Lock()

        self._doc_paths: dict[str, Path] = {}
        self._docs: dict[str, QuestionDocument] = {}
        self._source_docs: dict[str, SourceDocument] = {}
        # Flat index over EVERY question in EVERY loaded document,
        # regardless of routing/verification status -- duplicate
        # marking is cross-document and a duplicate target may well be
        # a question that's already auto_import or already verified,
        # not just something currently sitting in the needs_review
        # queue. Built once at load time; safe to hold onto because
        # QuestionRecord objects are mutated in place (question_id
        # never changes), so this index never goes stale.
        self._question_index: dict[str, tuple[str, QuestionRecord]] = {}

        self._load()

    def _load(self) -> None:
        doc_paths = sorted(self.reviewed_dir.glob("DOC-*.json"))
        if not doc_paths:
            print(
                f"[warning] no QuestionDocument files found in {self.reviewed_dir} "
                "-- run ai_parser.py first."
            )
        for p in doc_paths:
            doc = QuestionDocument(**json.loads(p.read_text()))
            self._doc_paths[doc.document_id] = p
            self._docs[doc.document_id] = doc

        for p in sorted(self.source_dir.glob("DOC-*.json")):
            src = SourceDocument(**json.loads(p.read_text()))
            self._source_docs[src.document_id] = src

        for document_id, doc in self._docs.items():
            for q in doc.questions:
                self._question_index[q.question_id] = (document_id, q)

    # -- duplicates ----------------------------------------------------

    def _duplicate_components(self) -> dict[str, set[str]]:
        """Cross-document duplicate groups, computed fresh every call --
        same "never store a derived value" discipline as ai_parser.route().
        A manually-maintained count/group would drift the moment a third
        copy of a question turns up; recomputing from the raw
        duplicate_of_question_id pointers means it never can.

        Uses union-find over ALL questions in the corpus (not just the
        current queue) so a duplicate group is correct even if some
        members have already been verified or auto-imported. Returns,
        for every question that's part of a >1-member group, the full
        set of member question_ids (including itself). Questions with
        no duplicate relationship at all are simply absent from the
        returned dict.
        """
        parent: dict[str, str] = {qid: qid for qid in self._question_index}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for qid, (_, q) in self._question_index.items():
            target = q.duplicate_of_question_id
            if target and target in parent:
                union(qid, target)

        groups: dict[str, set[str]] = {}
        for qid in parent:
            groups.setdefault(find(qid), set()).add(qid)

        by_member: dict[str, set[str]] = {}
        for members in groups.values():
            if len(members) > 1:
                for m in members:
                    by_member[m] = members
        return by_member

    # -- queue -----------------------------------------------------

    def _in_queue(self, q: QuestionRecord) -> bool:
        if route(q, self.threshold) != "needs_review":
            return False
        if q.verified_at is not None and not self.force:
            return False
        return True

    def build_queue(self) -> list[dict]:
        items: list[dict] = []
        with self._lock:
            for document_id, doc in self._docs.items():
                for q in doc.questions:
                    if not self._in_queue(q):
                        continue
                    items.append(self._to_queue_item(doc, q))
        # Stable, predictable order: by document, then question number
        # numerically where possible.
        def sort_key(item):
            num = item["question_number"]
            num_key = int(num) if num.isdigit() else float("inf")
            return (item["document_id"], item["section_type"], num_key)

        items.sort(key=sort_key)
        return items

    def _to_queue_item(self, doc: QuestionDocument, q: QuestionRecord) -> dict:
        pages = sorted({p for span in q.source_spans for p in span.pages})
        image_urls = [
            f"/api/image?doc={doc.document_id}&page={pg}" for pg in pages
        ]

        dup_components = self._duplicate_components()
        dup_members = dup_components.get(q.question_id)
        duplicate_group = []
        if dup_members:
            for member_id in sorted(dup_members):
                if member_id == q.question_id:
                    continue
                entry = self._question_index.get(member_id)
                if entry is None:
                    continue
                member_doc_id, member_q = entry
                duplicate_group.append(
                    {
                        "question_id": member_id,
                        "document_id": member_doc_id,
                        "question_number": member_q.question_number,
                        "section_type": member_q.section_type.value,
                    }
                )
        # "how many times has this question been duplicated" -- total
        # copies in the group including this one, so a lone question
        # with no duplicates at all reads as 0, not 1.
        duplicate_count = len(dup_members) if dup_members else 0

        return {
            "question_id": q.question_id,
            "document_id": doc.document_id,
            "original_filename": doc.original_filename,
            "subject": doc.subject,
            "topic": doc.topic,
            "section_type": q.section_type.value,
            "question_number": q.question_number,
            "question_text": q.question_text,
            "options": [o.model_dump() for o in q.options],
            "matching_pairs": [m.model_dump() for m in q.matching_pairs],
            "source_answer": q.source_answer,
            "verified_answer": q.verified_answer,
            "verified_at": q.verified_at,
            "verification_note": q.verification_note,
            "anomalies": q.anomalies,
            "ai_confidence": q.ai_confidence,
            "ai_concerns": q.ai_concerns,
            "ai_notes": q.ai_notes,
            "pages": pages,
            "image_urls": image_urls,
            # -- structural review (Layer 4 addition) --
            "duplicate_of_question_id": q.duplicate_of_question_id,
            "duplicate_group": duplicate_group,
            "duplicate_count": duplicate_count,
            "verified_question_text": q.verified_question_text,
            "verified_options": (
                [o.model_dump() for o in q.verified_options]
                if q.verified_options is not None
                else None
            ),
            "verified_matching_pairs": (
                [m.model_dump() for m in q.verified_matching_pairs]
                if q.verified_matching_pairs is not None
                else None
            ),
            "structural_reviewed_at": q.structural_reviewed_at,
        }

    # -- images ------------------------------------------------------

    def find_image_path(self, document_id: str, page: int) -> Path | None:
        src = self._source_docs.get(document_id)
        if src is None:
            return None
        for p in src.pages:
            if p.page != page or not p.rendered_image_path:
                continue
            stored = Path(p.rendered_image_path)
            if stored.exists():
                return stored
            # Portability fallback: the path was recorded on whatever
            # machine ran ingestion. If this tool is running somewhere
            # else (or the project folder moved), try the same
            # basename under this machine's --source-dir/_renders.
            fallback = self.source_dir / "_renders" / stored.name
            if fallback.exists():
                return fallback
            return None
        return None

    # -- writes --------------------------------------------------------

    def verify(
        self,
        question_id: str,
        document_id: str,
        verified_answer: str,
        note: str,
        duplicate_of_question_id: str,
        verified_question_text: str,
        verified_options: list[dict] | None,
        verified_matching_pairs: list[dict] | None,
    ) -> dict:
        """Writes both the answer-verification fields (always) and the
        structural-review fields (only if actually different from what's
        already stored). Every field here is sent in full on every save
        -- same "always send the full current state, empty clears it"
        convention verified_answer already used -- so there's no
        partial-update ambiguity to resolve.

        Raises ValueError for a bad duplicate_of_question_id (self-
        reference or unknown target) -- distinct from KeyError, which
        means the record being edited itself couldn't be found.
        """
        with self._lock:
            doc = self._docs.get(document_id)
            if doc is None:
                raise KeyError(f"unknown document_id: {document_id}")
            target: QuestionRecord | None = None
            for q in doc.questions:
                if q.question_id == question_id:
                    target = q
                    break
            if target is None:
                raise KeyError(f"unknown question_id: {question_id}")

            target.verified_answer = verified_answer if verified_answer != "" else None
            target.verification_note = note if note != "" else None
            target.verified_at = _now_iso()

            structural_changed = False

            dup_id = duplicate_of_question_id or None
            if dup_id == question_id:
                raise ValueError("a question cannot be marked as a duplicate of itself")
            if dup_id is not None and dup_id not in self._question_index:
                raise ValueError(f"unknown duplicate_of_question_id target: {dup_id}")
            if dup_id != target.duplicate_of_question_id:
                target.duplicate_of_question_id = dup_id
                structural_changed = True

            new_text = verified_question_text or None
            if new_text != target.verified_question_text:
                target.verified_question_text = new_text
                structural_changed = True

            new_opts = (
                [Option(**o) for o in verified_options]
                if verified_options is not None
                else None
            )
            if _options_key(new_opts) != _options_key(target.verified_options):
                target.verified_options = new_opts
                structural_changed = True

            new_pairs = (
                [MatchingPair(**m) for m in verified_matching_pairs]
                if verified_matching_pairs is not None
                else None
            )
            if _matching_pairs_key(new_pairs) != _matching_pairs_key(
                target.verified_matching_pairs
            ):
                target.verified_matching_pairs = new_pairs
                structural_changed = True

            if structural_changed:
                target.structural_reviewed_at = _now_iso()

            path = self._doc_paths[document_id]
            path.write_text(json.dumps(doc.model_dump(), indent=2, default=str))

            return self._to_queue_item(doc, target)


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------


def make_handler(store: ReviewStore):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # noqa: A002 -- quieter default logging
            print(f"[http] {self.address_string()} {fmt % args}")

        def _send_json(self, payload: dict | list, status: int = 200) -> None:
            body = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, path: Path, content_type: str) -> None:
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802 -- stdlib method name
            parsed = urlparse(self.path)

            if parsed.path == "/" or parsed.path == "/index.html":
                self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
                return

            if parsed.path == "/app.js":
                self._send_file(STATIC_DIR / "app.js", "application/javascript; charset=utf-8")
                return

            if parsed.path == "/api/queue":
                self._send_json(
                    {
                        "threshold": store.threshold,
                        "force": store.force,
                        "items": store.build_queue(),
                    }
                )
                return

            if parsed.path == "/api/image":
                qs = parse_qs(parsed.query)
                doc_id = (qs.get("doc") or [None])[0]
                page_raw = (qs.get("page") or [None])[0]
                if not doc_id or page_raw is None:
                    self._send_json({"error": "missing doc/page"}, status=400)
                    return
                try:
                    page = int(page_raw)
                except ValueError:
                    self._send_json({"error": "page must be an integer"}, status=400)
                    return
                image_path = store.find_image_path(doc_id, page)
                if image_path is None:
                    self._send_json(
                        {"error": "no page image available for this page"}, status=404
                    )
                    return
                content_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
                self._send_file(image_path, content_type)
                return

            self._send_json({"error": "not found"}, status=404)

        def do_POST(self):  # noqa: N802 -- stdlib method name
            parsed = urlparse(self.path)
            if parsed.path != "/api/verify":
                self._send_json({"error": "not found"}, status=404)
                return

            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                self._send_json({"error": "invalid JSON body"}, status=400)
                return

            question_id = body.get("question_id")
            document_id = body.get("document_id")
            verified_answer = body.get("verified_answer", "")
            note = body.get("verification_note", "")
            duplicate_of_question_id = body.get("duplicate_of_question_id", "")
            verified_question_text = body.get("verified_question_text", "")
            verified_options = body.get("verified_options")  # list[dict] or None
            verified_matching_pairs = body.get("verified_matching_pairs")  # list[dict] or None

            if not question_id or not document_id:
                self._send_json(
                    {"error": "question_id and document_id are required"}, status=400
                )
                return

            try:
                updated = store.verify(
                    question_id,
                    document_id,
                    verified_answer,
                    note,
                    duplicate_of_question_id,
                    verified_question_text,
                    verified_options,
                    verified_matching_pairs,
                )
            except KeyError as exc:
                self._send_json({"error": str(exc)}, status=404)
                return
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=400)
                return

            self._send_json({"ok": True, "item": updated})

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local web tool for human review of the needs_review bucket."
    )
    parser.add_argument(
        "--reviewed-dir",
        default="../reviewed",
        type=Path,
        help="Directory of QuestionDocument JSON files written by ai_parser.py "
        "(its --output). Verification decisions are written back here, in place.",
    )
    parser.add_argument(
        "--source-dir",
        default="../output",
        type=Path,
        help="Directory of SourceDocument JSON files written by main.py (its "
        "--output). Used to look up rendered_image_path for each question's pages.",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Must match the --threshold ai_parser.py was run with, so the queue "
        "reflects the same auto_import/needs_review split.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Include already-verified questions in the queue too (a second "
        "review pass), instead of only unreviewed needs_review questions.",
    )
    args = parser.parse_args()

    reviewed_dir: Path = args.reviewed_dir.resolve()
    source_dir: Path = args.source_dir.resolve()

    if not reviewed_dir.exists():
        print(f"Reviewed-document directory not found: {reviewed_dir}")
        print("Run ai_parser.py first (its --output is this stage's --reviewed-dir).")
        return

    if not STATIC_DIR.exists():
        print(f"Missing static assets directory: {STATIC_DIR}")
        return

    store = ReviewStore(reviewed_dir, source_dir, args.threshold, args.force)
    queue_len = len(store.build_queue())
    print(f"Loaded {len(store._docs)} document(s) from {reviewed_dir}")
    print(f"Loaded {len(store._source_docs)} source document(s) from {source_dir}")
    print(f"Pending review queue: {queue_len} question(s) (threshold={args.threshold})")

    handler = make_handler(store)
    server = ThreadingHTTPServer(("localhost", args.port), handler)
    print(f"\nReview tool running at http://localhost:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
