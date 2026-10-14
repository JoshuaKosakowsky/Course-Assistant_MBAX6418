"""Content-addressed document store for course materials (Task 1).

Owns the library lifecycle:
- add(): copies the file into the corpus, records a library entry, and
  de-duplicates by SHA-256 — loading the same file twice returns the
  existing document and never duplicates it.
- remove(): cascades — deletes the corpus copy, parsed artifacts, and the
  record itself, then fires removal hooks so future search indexes
  (bm25s/chromadb, Phase 2) purge this document's content.

All data lives under DATA_ROOT (gitignored).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .config import DATA_ROOT


def _make_writable(path: Path) -> None:
    """Clear read-only attributes (OneDrive marks synced directories and
    their contents read-only on Windows, which blocks deletion)."""
    targets = [path]
    if path.is_dir():
        targets = [
            Path(root) / name
            for root, dirs, files in os.walk(path)
            for name in dirs + files
        ]
    for target in targets:
        try:
            os.chmod(target, stat.S_IWRITE)
        except OSError:
            pass


def _delete_with_retry(path: Path, retries: int = 3, delays=(0.5, 2.0, 4.0)) -> Path | None:
    """Delete a file or directory, retrying on transient locks (Windows
    file/OneDrive sync can hold a handle for several seconds). Returns the
    path if it still exists after all attempts, else None."""
    for attempt in range(retries):
        try:
            _make_writable(path)
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            return None
        except OSError:
            if attempt < retries - 1:
                time.sleep(delays[attempt])
    return path if path.exists() else None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Document:
    doc_id: str          # first 16 hex chars of the content sha256
    filename: str
    format: str          # lower-case extension, e.g. ".pdf"
    sha256: str
    size_bytes: int
    added_at: str        # ISO-8601 UTC

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Document":
        return cls(**d)


class DocumentNotFoundError(KeyError):
    pass


class DocumentStore:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else DATA_ROOT
        self.library_dir = self.root / "library"
        self.corpus_dir = self.root / "corpus"
        self.artifacts_dir = self.root / "artifacts"
        for d in (self.library_dir, self.corpus_dir, self.artifacts_dir):
            d.mkdir(parents=True, exist_ok=True)
        self._removal_hooks: list[Callable[[str], None]] = []

    # -- lifecycle ----------------------------------------------------------

    def add(self, source: Path) -> Document:
        """Register a file in the library. Idempotent by content: a second
        load of the same bytes returns the existing Document unchanged."""
        self.cleanup_orphans()  # sweep leftovers from earlier removes
        source = Path(source)
        if not source.is_file():
            raise FileNotFoundError(source)
        digest = _sha256(source)
        existing = self._find_by_sha(digest)
        if existing is not None:
            return existing
        doc = Document(
            doc_id=digest[:16],
            filename=source.name,
            format=source.suffix.lower(),
            sha256=digest,
            size_bytes=source.stat().st_size,
            added_at=_now_iso(),
        )
        dest = self.corpus_dir / doc.doc_id / source.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        self._write_record(doc)
        return doc

    def list(self) -> list[Document]:
        """All library documents, oldest first."""
        docs = []
        for record in sorted(self.library_dir.glob("*.json")):
            docs.append(Document.from_dict(json.loads(record.read_text(encoding="utf-8"))))
        return docs

    def get(self, doc_id: str) -> Document:
        record = self.library_dir / f"{doc_id}.json"
        if not record.is_file():
            raise DocumentNotFoundError(doc_id)
        return Document.from_dict(json.loads(record.read_text(encoding="utf-8")))

    def find(self, source: Path) -> Document | None:
        """Return the library document for a file if its content is already
        stored, else None. Powers the UI's 'already in library' message."""
        source = Path(source)
        return self._find_by_sha(_sha256(source)) if source.is_file() else None

    def corpus_path(self, doc_id: str) -> Path:
        """Path of the stored corpus copy for a document (the authoritative
        file to parse — content-addressed, independent of the upload path)."""
        directory = self.corpus_dir / doc_id
        files = list(directory.glob("*")) if directory.is_dir() else []
        if not files:
            raise DocumentNotFoundError(doc_id)
        return files[0]

    def remove(self, doc_id: str) -> None:
        """Delete a document and ALL of its searchable content: corpus copy,
        parsed artifacts, and the library record. The record is always
        removed (a remaining entry would let later answers rely on a deleted
        document); artifacts/corpus deletion retries on transient Windows
        locks. Raises DocumentNotFoundError for unknown ids."""
        self.get(doc_id)  # raises DocumentNotFoundError when unknown
        for hook in self._removal_hooks:
            hook(doc_id)
        for target in (self.artifacts_dir / doc_id, self.corpus_dir / doc_id):
            _delete_with_retry(target)
        record = self.library_dir / f"{doc_id}.json"
        if record.exists() and _delete_with_retry(record):
            raise OSError(f"Could not delete library record {record}")
        # NOTE: artifact/corpus dirs still locked after retries are swept by
        # cleanup_orphans() on the next add(); indexes are purged via hooks.

    # -- removal hooks (search indexes register here in Phase 2) ------------

    def register_removal_hook(self, hook: Callable[[str], None]) -> None:
        """Register a callable invoked with the doc_id on every remove().
        Phase 2 indexes (bm25s, chromadb) register here so a removed
        document's chunks and embeddings are purged."""
        self._removal_hooks.append(hook)

    def cleanup_orphans(self) -> list[Path]:
        """Delete artifacts/corpus directories whose library record no
        longer exists (leftovers from removes that hit transient locks).
        Returns the paths removed. Cheap enough to run on every add()."""
        known = {d.doc_id for d in self.list()}
        removed: list[Path] = []
        for base in (self.artifacts_dir, self.corpus_dir):
            for entry in list(base.glob("*")):
                if entry.is_dir() and entry.name not in known:
                    if _delete_with_retry(entry) is None:
                        removed.append(entry)
        return removed

    # -- internals ----------------------------------------------------------

    def _find_by_sha(self, digest: str) -> Document | None:
        for doc in self.list():
            if doc.sha256 == digest:
                return doc
        return None

    def _write_record(self, doc: Document) -> None:
        (self.library_dir / f"{doc.doc_id}.json").write_text(
            json.dumps(doc.to_dict(), indent=2), encoding="utf-8"
        )
