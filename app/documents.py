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
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .config import DATA_ROOT


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

    def remove(self, doc_id: str) -> None:
        """Delete a document and ALL of its searchable content: corpus copy,
        parsed artifacts, and the library record. Removes nothing if the
        document is unknown. Returns the removed document."""
        self.get(doc_id)  # raises DocumentNotFoundError when unknown
        for hook in self._removal_hooks:
            hook(doc_id)
        artifacts = self.artifacts_dir / doc_id
        corpus = self.corpus_dir / doc_id
        record = self.library_dir / f"{doc_id}.json"
        if artifacts.is_dir():
            shutil.rmtree(artifacts)
        if corpus.is_dir():
            shutil.rmtree(corpus)
        if record.is_file():
            record.unlink()

    # -- removal hooks (search indexes register here in Phase 2) ------------

    def register_removal_hook(self, hook: Callable[[str], None]) -> None:
        """Register a callable invoked with the doc_id on every remove().
        Phase 2 indexes (bm25s, chromadb) register here so a removed
        document's chunks and embeddings are purged."""
        self._removal_hooks.append(hook)

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
