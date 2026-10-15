"""End-to-end indexing workflow (Issue #2 / Phase 2).

Connects the team's ingestion to both text indexes:

    DocumentStore -> parse (or existing artifacts/manifests)
                  -> per-page text -> chunking -> BM25 + ChromaDB text index

Design decisions:

- The DocumentStore is the single source of truth. The indexer subscribes
  to the store's EXISTING removal-hook mechanism so deleting a document
  purges BOTH indexes; the UI calls ``index_document`` after a successful
  parse (see app/ui.py).
- Parsed artifacts (``artifacts/<doc_id>/manifest.json``) are reused when
  present; a document is parsed only when no manifest exists (e.g. the
  CLI indexing an unparsed library).
- BM25 is in-memory: it is REBUILT from the library + artifacts at
  startup (``reconcile``) and purged on removal, so a restart can never
  resurrect deleted content.
- ChromaDB is persistent: ``reconcile`` also prunes vectors whose doc_id
  is no longer in the library, so deleted documents cannot resurface.
- Embedding failures degrade gracefully: the keyword index always builds;
  vector indexing for the affected document is reported as skipped and is
  retriable; the document library is never modified by indexing.

Command line:  python -m app.indexing [--reindex] [--keyword-only]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from app.chunking import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    PageText,
    chunk_pages,
)
from app.documents import Document, DocumentStore
from app.embeddings import EmbeddingsError, TextEmbeddingClient
from app.keyword import Hit, KeywordIndex
from app.parse import PageRecord, parse_file
from app.vector import TEXT_COLLECTION, TextVectorIndex
from app import config

DEFAULT_PERSIST_DIR = config.DATA_ROOT / "indexes"


@dataclass
class IndexStats:
    """Counts and warnings for one indexing operation."""

    doc_ids: int = 0
    pages: int = 0          # pages that actually yielded text (empty pages skipped)
    chunks: int = 0
    keyword_chunks: int = 0
    vector_chunks: int = 0
    visual_pages: int = 0   # page images embedded into the visual index
    visual_added: int = 0   # == visual_pages today (kept for clarity/summing)
    visual_repaired: int = 0  # vectors restored after detected loss (integrity)
    skipped: List[str] = field(default_factory=list)      # e.g. vector service down
    errors: List[str] = field(default_factory=list)       # e.g. unparseable document
    visual_skipped: List[str] = field(default_factory=list)  # already-indexed pages
    visual_failed: List[str] = field(default_factory=list)   # embed errors (honest)

    def summary(self) -> Dict[str, object]:
        return {
            "docs": self.doc_ids,
            "pages": self.pages,
            "chunks": self.chunks,
            "keyword_chunks": self.keyword_chunks,
            "vector_chunks": self.vector_chunks,
            "visual_pages": self.visual_pages,
            "visual_repaired": self.visual_repaired,
            "skipped": self.skipped,
            "errors": self.errors,
            "visual_skipped": self.visual_skipped,
            "visual_failed": self.visual_failed,
        }

    def __add__(self, other: "IndexStats") -> "IndexStats":
        return IndexStats(
            doc_ids=self.doc_ids + other.doc_ids,
            pages=self.pages + other.pages,
            chunks=self.chunks + other.chunks,
            keyword_chunks=self.keyword_chunks + other.keyword_chunks,
            vector_chunks=self.vector_chunks + other.vector_chunks,
            visual_pages=self.visual_pages + other.visual_pages,
            visual_added=self.visual_added + other.visual_added,
            visual_repaired=self.visual_repaired + other.visual_repaired,
            skipped=self.skipped + other.skipped,
            errors=self.errors + other.errors,
            visual_skipped=self.visual_skipped + other.visual_skipped,
            visual_failed=self.visual_failed + other.visual_failed,
        )

    def __iadd__(self, other: "IndexStats") -> "IndexStats":
        merged = self + other
        self.__dict__.update(merged.__dict__)
        return self


class CorpusIndexer:
    """Builds and maintains BM25 + ChromaDB text indexes for a library.

    ``visual`` is an OPTIONAL app.visual.VisualIndexer (None = text-only).
    It is duck-typed and lazy: indexing.py never imports app.visual at
    module load, so the application works before PR #11 merges.
    """

    def __init__(
        self,
        store: DocumentStore,
        embedder=None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        persist_dir: Optional[Path | str] = None,
        visual=None,
    ) -> None:
        self.store = store
        self.embedder = embedder  # any object exposing embed(list[str]); None = keyword-only
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.keyword = KeywordIndex()
        self.vector = TextVectorIndex(
            self.embedder,
            persist_dir=persist_dir if persist_dir is not None else DEFAULT_PERSIST_DIR,
        )
        self.visual = visual  # VisualIndexer | None; text-only mode when None
        self._wired = False

    # ---- lifecycle ----------------------------------------------------

    def register_hooks(self) -> None:
        """Subscribe to the store's removal hook (existing mechanism).

        This is what makes document deletion purge the text indexes AND
        the visual index in the real application flow. No add-hook:
        indexing after a successful upload+parse is driven explicitly by
        the caller (UI or CLI). Idempotent: registering twice never adds
        a duplicate hook.
        """
        if self._wired:
            return
        self.store.register_removal_hook(self.remove_document)
        self._wired = True

    def reconcile(self, visual_backfill: bool = False) -> IndexStats:
        """Startup reconciliation (offline-safe apart from missing-vector
        repair, which needs the embedder):

        1. Rebuild the in-memory BM25 index from the library + artifacts,
           so deleted documents cannot reappear after a restart.
        2. Prune ChromaDB vectors whose doc_id is no longer in the library.
        3. Repair: embed documents that have NO stored vectors yet (e.g.
           the embedding service was down during a previous session).
           Only missing documents are embedded — existing vectors are
           preserved and never re-embedded just because the app restarted.
        4. Visual index (when configured): page-level incremental
           reconcile — unchanged pages are skipped with zero embedding
           calls; only missing/changed pages are embedded. Pass
           ``visual_backfill=True`` to force re-embedding of everything
           (the explicit ``python -m app.indexing --visual-backfill``
           command).
        """
        stats = IndexStats()
        self.keyword = KeywordIndex()
        for doc in self.store.list():
            stats += self._index_one(doc, to_keyword=True, to_vector=False)
        # prune orphan vectors in the persistent store
        known = {d.doc_id for d in self.store.list()}
        for doc_id in self.vector.stored_doc_ids():
            if doc_id not in known:
                self.vector.remove_document(doc_id)
        # repair missing vectors (only for documents that lack them)
        if self.embedder is not None:
            present = self.vector.stored_doc_ids()
            for doc in self.store.list():
                if doc.doc_id not in present:
                    try:
                        stats += self._index_one(
                            doc, to_keyword=False, to_vector=True
                        )
                    except Exception as exc:  # pragma: no cover - defensive
                        stats.skipped.append(f"vector:{doc.doc_id}: {exc}")
        stats.vector_chunks = self.vector.count
        # visual index: incremental by default, forced backfill on request
        if self.visual is not None:
            try:
                stats += self.visual.reconcile(force=visual_backfill)
            except Exception as exc:  # pragma: no cover - never block startup
                stats.visual_failed.append(f"reconcile: {exc}")
        return stats

    # ---- per-document indexing ----------------------------------------

    def index_document(self, doc_id: str) -> IndexStats:
        """Index (or reindex) one document: BM25 + text vectors when the
        embedder is available (keyword-only otherwise), PLUS visual page
        images when a visual indexer is configured. Idempotent; failures
        in one index never mask success in the others, and nothing is
        claimed as indexed unless it actually was."""
        stats = IndexStats()
        try:
            doc = self.store.get(doc_id)
        except Exception as exc:  # DocumentNotFoundError
            stats.errors.append(f"{doc_id}: {exc}")
            return stats
        stats += self._index_one(doc, to_keyword=True, to_vector=True)
        if self.visual is not None:
            try:
                stats += self.visual.index_document(doc_id)
            except Exception as exc:  # pragma: no cover - defensive
                stats.visual_failed.append(f"{doc_id}: {exc}")
        return stats

    def _index_one(
        self, doc: Document, to_keyword: bool, to_vector: bool
    ) -> IndexStats:
        stats = IndexStats(doc_ids=1)
        try:
            pages = self._pages_for(doc)
            non_empty = [p for p in pages if p.text.strip()]
            chunks = chunk_pages(
                doc.doc_id,
                doc.filename,
                pages,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                format=doc.format,
            )
        except Exception as exc:
            stats.errors.append(f"{doc.doc_id}: unparseable: {exc}")
            return stats
        stats.pages = len(non_empty)
        stats.chunks = len(chunks)

        if to_keyword:
            # purge-then-add: stale chunks can never survive a reindex
            self.keyword.remove_document(doc.doc_id)
            self.keyword.add(chunks)
            stats.keyword_chunks = len(chunks)  # per-doc count (summable)
        if to_vector:
            if self.embedder is None:
                stats.skipped.append(
                    f"vector:{doc.doc_id}: keyword-only mode (no embedding configured)"
                )
                return stats
            try:
                self.vector.reindex_document(doc.doc_id, chunks)
                stats.vector_chunks = len(chunks)  # per-doc count (summable)
            except EmbeddingsError as exc:
                stats.skipped.append(f"vector:{doc.doc_id}: {exc}")
            except Exception as exc:  # pragma: no cover - defensive
                stats.skipped.append(f"vector:{doc.doc_id}: {exc}")
        return stats

    def index_library(self) -> IndexStats:
        """Index every document currently in the library."""
        total = IndexStats()
        for doc in self.store.list():
            total += self._index_one(doc, to_keyword=True, to_vector=True)
        return total

    def remove_document(self, doc_id: str) -> None:
        """Removal-hook target: purge the document from ALL indexes.

        ChromaDB/visual deletes are purely local — this works even when
        the embedding service is unavailable. A visual-index failure is
        reported (and the visual state dropped) but never blocks the text
        purge.
        """
        self.keyword.remove_document(doc_id)
        self.vector.remove_document(doc_id)
        if self.visual is not None:
            self.visual.remove_document(doc_id)

    # ---- queries ------------------------------------------------------

    def search(
        self, query: str, top_k: int = 5, doc_ids: Optional[Sequence[str]] = None
    ) -> Dict[str, List[Hit]]:
        """Run the query against both indexes (fusion is Phase 3)."""
        return {
            "keyword": self.keyword.search(query, top_k=top_k, doc_ids=doc_ids),
            "vector": self.vector.search(query, top_k=top_k, doc_ids=doc_ids),
        }

    def status(self) -> Dict[str, object]:
        return {
            "library_docs": len(self.store.list()),
            "keyword_chunks": self.keyword.size,
            "vector_chunks": self.vector.count,
            "collection": self.vector.collection_name,
            "visual": "enabled" if self.visual is not None else "unavailable",
            "visual_pages": self.visual.count() if self.visual is not None else 0,
        }

    # ---- helpers ------------------------------------------------------

    def _pages_for(self, doc: Document) -> List[PageText]:
        """Page records from the existing manifest, or parse if missing."""
        manifest = self.store.artifacts_dir / doc.doc_id / "manifest.json"
        if manifest.is_file():
            raw = json.loads(manifest.read_text(encoding="utf-8"))
            records = [
                PageRecord(
                    number=int(p.get("number", 0)),
                    text=str(p.get("text", "") or ""),
                    image_path=str(p.get("image_path", "") or ""),
                    embedded_images=int(p.get("embedded_images", 0) or 0),
                )
                for p in raw.get("pages", [])
            ]
        else:
            source = self.store.corpus_path(doc.doc_id)
            result = parse_file(source, doc.doc_id, self.store.artifacts_dir / doc.doc_id)
            records = result.pages
        return [
            PageText(page_no=r.number, text=r.text or "", image_path=r.image_path or None)
            for r in records
        ]


def _stats_line(stats: IndexStats) -> str:
    s = stats.summary()
    extras = []
    if s["skipped"]:
        extras.append(f"skipped({len(s['skipped'])})")
    if s["errors"]:
        extras.append(f"errors({len(s['errors'])})")
    if s["visual_skipped"]:
        extras.append(f"visual_skipped({len(s['visual_skipped'])})")
    if s["visual_failed"]:
        extras.append(f"visual_failed({len(s['visual_failed'])})")
    tail = f" [{', '.join(extras)}]" if extras else ""
    base = (
        f"docs={s['docs']} pages={s['pages']} chunks={s['chunks']} "
        f"keyword_chunks={s['keyword_chunks']} vector_chunks={s['vector_chunks']}"
    )
    visual = f" visual_pages={s['visual_pages']}" if s["visual_pages"] else ""
    return base + visual + tail


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.indexing",
        description=(
            "Index the document library into BM25 + ChromaDB text indexes, "
            "and (when configured) page images into the visual index. "
            "Reuses existing parse artifacts; parses only documents without "
            "a manifest. Semantic (vector) indexing needs CLASS_API_KEY in "
            ".env; without it the run is keyword-only and says so."
        ),
    )
    parser.add_argument("--reindex", action="store_true",
                        help="clear both indexes before indexing")
    parser.add_argument("--keyword-only", action="store_true",
                        help="skip semantic (vector) indexing")
    parser.add_argument(
        "--visual-backfill", action="store_true",
        help="(re)index ALL page images into the visual index, ignoring "
             "the incremental state (explicit command; startup never does "
             "a full visual reindex automatically)",
    )
    parser.add_argument(
        "--doc", default=None,
        help="index only this document id (text + visual)",
    )
    args = parser.parse_args(argv)

    store = DocumentStore()
    embedder = None
    if not args.keyword_only:
        try:
            embedder = TextEmbeddingClient()
        except EmbeddingsError as exc:
            print(f"WARNING: {exc}")
            print("Indexing will be KEYWORD-ONLY (set CLASS_API_KEY in .env to enable semantic search).")

    stats = IndexStats()
    indexer = CorpusIndexer(
        store, embedder=embedder, persist_dir=DEFAULT_PERSIST_DIR
    )
    if args.reindex:
        indexer.vector.clear()
        indexer.keyword = KeywordIndex()

    # optional visual indexing (is a no-op with a clear reason pre-merge)
    if not args.keyword_only:
        from app.visual import production_visual_indexer

        visual, reason = production_visual_indexer(store)
        if visual is None:
            print(f"Visual index DISABLED: {reason}")
        else:
            indexer.visual = visual
            print(f"Visual index enabled (state: {visual.state.path})")

    if args.doc:
        stats = indexer.index_document(args.doc)
    else:
        stats = indexer.index_library()
        stats.vector_chunks = indexer.vector.count

    # explicit visual backfill: an explicit request must never report
    # overall success when the visual index cannot do the work
    if args.visual_backfill:
        if indexer.visual is None:
            print(
                f"ERROR: --visual-backfill requested, but the visual "
                f"index is unavailable ({reason}); text indexing above is "
                f"complete, the backfill was NOT performed.",
                file=sys.stderr,
            )
            return 1
        stats += indexer.visual.reconcile(force=True)

    print(f"Indexed library: {_stats_line(stats)}")
    print(f"Status: {indexer.status()}")
    if stats.errors:
        print("Errors:")
        for e in stats.errors:
            print(f"  ! {e}")
    if stats.skipped:
        print("Skipped (embedding service):")
        for s in stats.skipped:
            print(f"  ~ {s}")
    if stats.visual_failed:
        print("Visual failures (pages NOT verified indexed):")
        for s in stats.visual_failed:
            print(f"  ! {s}")
    if stats.visual_skipped:
        print(f"Visual pages already indexed/skipped: {len(stats.visual_skipped)}")
    return 0 if not (stats.errors or stats.visual_failed) else 1


if __name__ == "__main__":
    sys.exit(main())
