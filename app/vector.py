"""ChromaDB text-vector retrieval over ``app.chunking.Chunk`` records
(Issue #2 / Phase 2).

Uses a persistent ChromaDB collection whose vectors come from an injected
embedder (the real ``TextEmbeddingClient`` in production, a deterministic
fake in offline tests). The collection is named ``TEXT_COLLECTION`` and is
kept strictly separate from the future visual embedding index (reserved
name ``VISUAL_COLLECTION``), per the assignment requirement.

Distance semantics: the collection uses the cosine space, so smaller
distances mean more similar; ``Hit.score = 1 - distance`` and only hits
with a positive similarity are returned.

Mutation semantics mirror ``app.keyword.KeywordIndex``:

- ``add`` upserts by stable chunk id (no duplicates, idempotent).
- ``remove_document`` matches the team's removal-hook contract
  (``Callable[[str], None]``) so ``store.register_removal_hook(index.
  remove_document)`` works once ingestion is wired.
- ``reindex_document`` deletes the document's old vectors first, so a
  re-ingest under different chunking settings cannot leave stale vectors.
- ``clear`` drops the whole collection (persistent path kept).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import chromadb
from chromadb.config import Settings

from app.chunking import Chunk
from app.embeddings import EmbeddingsError
from app.keyword import Hit

TEXT_COLLECTION = "course_text_chunks"
VISUAL_COLLECTION = "course_visual_chunks"  # reserved; created in Phase 2 (visual)


def _metadata(chunk: Chunk) -> dict:
    """ChromaDB metadata values must be primitives (no None)."""
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "filename": chunk.filename,
        "format": chunk.format or "",
        "page_no": int(chunk.page_no),
        "index_in_page": int(chunk.index_in_page),
        "image_path": chunk.image_path or "",
    }


class TextVectorIndex:
    """Persistent ChromaDB semantic index over text chunks."""

    def __init__(
        self,
        embedder,
        persist_dir: Optional[Path | str] = None,
        collection_name: str = TEXT_COLLECTION,
    ) -> None:
        settings = Settings(anonymized_telemetry=False)
        if persist_dir is None:
            self._client = chromadb.EphemeralClient(settings=settings)
        else:
            persist_dir = Path(persist_dir)
            persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(persist_dir), settings=settings
            )
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
        self._embedder = embedder  # any object exposing embed(list[str]) -> list[list[float]]
        self.dimension: Optional[int] = self._stored_dimension()

    def _stored_dimension(self) -> Optional[int]:
        """Dimension the existing collection is locked to (ChromaDB fixes
        the HNSW dimension at first insert), or None if empty."""
        try:
            peek = self._collection.peek(limit=1)
        except Exception:  # pragma: no cover - defensive
            return None
        vectors = peek.get("embeddings")
        if vectors is not None and len(vectors) > 0:  # numpy: no truthiness
            return len(vectors[0])
        return None

    # ---- introspection -------------------------------------------------

    @property
    def count(self) -> int:
        return self._collection.count()

    @property
    def collection_name(self) -> str:
        return self._collection.name

    def collection_names(self) -> List[str]:
        return sorted(c.name for c in self._client.list_collections())

    # ---- mutation ------------------------------------------------------

    def add(self, chunks: Sequence[Chunk]) -> None:
        """Embed and upsert chunk records (idempotent per chunk_id)."""
        chunks = list(chunks)
        if not chunks:
            return
        vectors = self._embedder.embed([c.text for c in chunks])
        if len(vectors) != len(chunks):
            raise RuntimeError(
                f"embedder returned {len(vectors)} vectors for "
                f"{len(chunks)} chunks"
            )
        incoming_dim = len(vectors[0])
        if self.dimension is None:
            self.dimension = incoming_dim
        elif incoming_dim != self.dimension:
            raise EmbeddingsError(
                f"dimension mismatch: this collection is locked to "
                f"{self.dimension}D but the embedder produced "
                f"{incoming_dim}D. Run `python -m app.indexing --reindex` "
                f"(or clear the collection) before switching embedding models."
            )
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[_metadata(c) for c in chunks],
        )

    def remove_document(self, doc_id: str) -> int:
        """Delete every vector for one document; returns count removed.

        ChromaDB 1.x drops ``where`` from ``Collection.count()``, so the
        count comes from ``get(..., include=[])`` and deletion is exact by
        ids (not a where clause).
        """
        where = {"doc_id": doc_id}
        ids = self._collection.get(where=where, include=[]).get("ids", [])
        removed = len(ids)
        if removed:
            self._collection.delete(ids=ids)
        return removed

    def reindex_document(self, doc_id: str, chunks: Sequence[Chunk]) -> None:
        """Replace a document's vectors (old chunking settings dropped)."""
        self.remove_document(doc_id)
        self.add(chunks)

    def clear(self) -> None:
        """Drop all vectors; the collection is recreated in place."""
        name = self._collection.name
        self._client.delete_collection(name)
        self._collection = self._client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )

    # ---- search --------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        doc_ids: Optional[Sequence[str]] = None,
    ) -> List[Hit]:
        """Return the top-k chunks most semantically similar to ``query``.

        ``doc_ids`` restricts retrieval to those documents. Empty queries,
        empty indexes, and `doc_ids=[]` return ``[]``. Hits with non-positive
        similarity are dropped (no meaningful match).
        """
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError(f"top_k must be a positive integer, got {top_k!r}")
        text = (query or "").strip()
        if not text or self.count == 0:
            return []
        allowed = list(doc_ids) if doc_ids is not None else None
        if allowed == []:
            return []

        query_vector = self._embedder.embed([text])[0]
        where = {"doc_id": {"$in": allowed}} if allowed is not None else None
        result = self._collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        ids = (result.get("ids") or [[]])[0] or []
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        hits: List[Hit] = []
        for chunk_id, doc_text, meta, distance in zip(
            ids, documents, metadatas, distances
        ):
            score = 1.0 - float(distance)  # cosine space: 1 - distance
            if score <= 0.0:
                continue
            hits.append(
                Hit(
                    chunk=Chunk(
                        chunk_id=chunk_id,
                        doc_id=meta["doc_id"],
                        filename=meta["filename"],
                        format=meta["format"],
                        page_no=int(meta["page_no"]),
                        index_in_page=int(meta["index_in_page"]),
                        text=doc_text,
                        image_path=meta.get("image_path") or None,
                    ),
                    score=score,
                )
            )
        return hits
