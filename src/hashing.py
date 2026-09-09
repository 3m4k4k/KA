"""
Content-hash based dedup so re-running the pipeline over the same input
folder never reprocesses (and re-bills LLM/OCR calls for) a file that
hasn't changed. This is the "cheap, resumable, cost-aware" layer.

The manifest is a flat JSON file. For thousands of documents this is
fine; if it ever gets unwieldy, swap this module for a SQLite-backed
version without touching any caller.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from schemas import ManifestEntry


def hash_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


class Manifest:
    def __init__(self, manifest_path: Path):
        self.path = manifest_path
        self._entries: dict[str, ManifestEntry] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            raw = json.loads(self.path.read_text())
            self._entries = {
                k: ManifestEntry(**v) for k, v in raw.items()
            }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        raw = {k: v.model_dump() for k, v in self._entries.items()}
        self.path.write_text(json.dumps(raw, indent=2))

    def already_processed(self, file_hash: str) -> ManifestEntry | None:
        return self._entries.get(file_hash)

    def record(self, entry: ManifestEntry) -> None:
        self._entries[entry.file_hash] = entry
        self.save()
