"""BM25 keyword retrieval over text chunks (Issue #2 / Phase 2).

Builds a bm25s index over ``app.chunking.Chunk`` records and searches
them by a student's text query, returning ranked hits that carry the
full chunk (source metadata preserved).

Verified bm25s 0.3.13 behavior this implementation relies on:

- ``bm25s.tokenize()`` returns a ``Tokenized`` object (``ids`` +
  ``vocab``); the empty (stopword-only) query yields no ids.
- ``BM25.get_scores`` accepts a FLAT list of token strings (or ids) and
  returns one score per corpus document; higher = more relevant.
- ``BM25`` has **no delete**: mutation (add/remove/reindex) therefore
  rebuilds the index from the current chunk set. Corpus sizes here are
  small (course materials), rebuilds are fast, and rebuilding keeps
  everything consistent by construction — no stale chunks, no
  id-reuse bugs.
- Indexing an empty corpus raises, so searches on an empty index short
  circuit before touching bm25s.

Integration with the team's document ingestion (PR #10): the removal
hook contract is ``Callable[[doc_id], None]`` — ``remove_document``
matches it, so ``store.register_removal_hook(index.remove_document)``
plugs straight in once that branch merges; nothing here imports or
modifies ``app.documents``/``app.parse``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set

import bm25s
import numpy as np

from app.chunking import Chunk

DEFAULT_TOP_K = 5
_STOPWORDS = "en"


@dataclass(frozen=True)
class Hit:
    """One ranked retrieval result: the full chunk plus its score."""

    chunk: Chunk
    score: float


class KeywordIndex:
    """In-memory BM25 index over chunk records.

    The public surface is deliberately small: ``add``, ``remove_document``,
    ``replace_all``, ``clear``, ``search``, with ``chunk_ids``/``chunks``
    for inspection and ``size`` for counts. All mutations rebuild the
    underlying bm25s index (see module docstring).
    """

    def __init__(self) -> None:
        self._chunks: Dict[str, Chunk] = {}
        self._ids: List[str] = []
        self._bm25: Optional[bm25s.BM25] = None

    # ---- introspection -------------------------------------------------

    @property
    def size(self) -> int:
        return len(self._ids)

    @property
    def chunk_ids(self) -> List[str]:
        return list(self._ids)

    def chunks(self) -> List[Chunk]:
        return [self._chunks[cid] for cid in self._ids]

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        return self._chunks.get(chunk_id)

    # ---- mutation ------------------------------------------------------

    def add(self, chunks: Sequence[Chunk]) -> None:
        """Index new chunks (deduplicated by chunk_id)."""
        changed = False
        for chunk in chunks:
            if chunk.chunk_id in self._chunks:
                continue
            self._chunks[chunk.chunk_id] = chunk
            self._ids.append(chunk.chunk_id)
            changed = True
        if changed:
            self._rebuild()

    def remove_document(self, doc_id: str) -> int:
        """Remove every chunk belonging to one document; returns the
        number of chunks dropped. Matches the team's removal-hook
        signature (Callable[[str], None])."""
        removed = 0
        for chunk_id in list(self._ids):
            if self._chunks[chunk_id].doc_id == doc_id:
                del self._chunks[chunk_id]
                self._ids.remove(chunk_id)
                removed += 1
        if removed:
            self._rebuild()
        return removed

    def replace_all(self, chunks: Sequence[Chunk]) -> None:
        """Drop everything and index the given chunks from scratch.
        Used when documents are re-ingested under different chunking
        settings (new chunk ids) so no stale chunks survive."""
        self.clear()
        self.add(chunks)

    def clear(self) -> None:
        self._chunks = {}
        self._ids = []
        self._bm25 = None

    # ---- search --------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        doc_ids: Optional[Sequence[str]] = None,
    ) -> List[Hit]:
        """Return the top-k chunks most relevant to ``query``, ordered
        best first.

        ``doc_ids`` restricts retrieval to those documents (useful for
        the app's "choose course material" flow). Empty index, empty /
        whitespace / stopword-only queries and queries with no lexical
        match all return ``[]`` without crashing.
        """
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError(f"top_k must be a positive integer, got {top_k!r}")

        text = (query or "").strip()
        if not text or self._bm25 is None:
            return []

        tokens = bm25s.tokenize(text, stopwords=_STOPWORDS)
        # flatten the query to string tokens (verified input for get_scores)
        if not tokens.ids or len(tokens.ids[0]) == 0:
            return []  # stopword-only query: no discriminative terms
        inverse_vocab = {idx: token for token, idx in tokens.vocab.items()}
        query_tokens = [inverse_vocab[int(i)] for i in tokens.ids[0]]

        scores = self._bm25.get_scores(query_tokens)  # shape == (n_corpus,)
        allowed: Optional[Set[str]] = (
            set(doc_ids) if doc_ids is not None else None
        )
        if allowed == set():
            return []  # explicitly no documents selected

        order = np.argsort(-scores)  # descending relevance
        hits: List[Hit] = []
        seen: Set[str] = set()
        for pos in order:
            pos_int = int(pos)
            score = float(scores[pos_int])
            if score <= 0.0:
                break  # everything below is equally non-matching
            chunk_id = self._ids[pos_int]
            chunk = self._chunks[chunk_id]
            if allowed is not None and chunk.doc_id not in allowed:
                continue
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            hits.append(Hit(chunk=chunk, score=score))
            if len(hits) >= top_k:
                break
        return hits

    # ---- internals -----------------------------------------------------

    def _rebuild(self) -> None:
        if not self._ids:
            self._bm25 = None
            return
        corpus = [self._chunks[cid].text for cid in self._ids]
        tokenized = bm25s.tokenize(corpus, stopwords=_STOPWORDS)
        bm25 = bm25s.BM25()
        bm25.index(tokenized)
        self._bm25 = bm25
