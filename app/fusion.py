"""Fusion of independent retrieval rankings (Issue #4 / Phase 3, Step 1).

Reciprocal Rank Fusion (RRF) over the existing per-index retrievers
(``app.keyword.KeywordIndex``, ``app.vector.TextVectorIndex``), with a
schema designed to admit the visual index (PR #11) later.

Scoring: BM25 scores and vector similarities live on different scales, so
they are NEVER combined directly. Only each candidate's *rank* inside a
retriever contributes to the fused score:

    RRF_score(chunk) = SUM over retrievers r of  1 / (k + rank_r(chunk))

where ``rank_r`` is the 1-based position of the chunk in retriever r's
ranking and ``k`` is the RRF constant (default 60, the standard choice;
larger k flattens the influence of high ranks). A chunk found by several
retrievers is boosted by the sum — this is the whole point of hybrid
fusion — and each contribution keeps its raw score for debugging.

Result schema (``FusedHit``) preserves everything the Q&A phase needs:

- ``chunk``: the full ``Chunk`` (doc_id, filename, format, page_no,
  index_in_page, text, image_path) — text evidence with page identity.
- ``methods``: which retrievers found the chunk (e.g. ("keyword",
  "vector"), later "visual").
- ``evidence``: per-retriever rank (1-based) and *unscaled* raw score,
  preserved verbatim for evaluation and the evidence pack.

Visual integration: the engine only knows a ``SearchSource`` protocol
(query text -> ranked hits). ``VisualIndex.query_text`` returns
``VisualHit`` (doc_id, page, image_path, distance) — a small adapter can
satisfy the protocol without changing this module, mapping ``page`` to
``page_no`` and carrying ``image_path``. Nothing here assumes text and
visual embeddings are in the same space: RRF mixes rankings, not vectors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence, runtime_checkable

from app.chunking import Chunk
from app.keyword import Hit

RRF_K_DEFAULT = 60


@runtime_checkable
class SearchSource(Protocol):
    """A retriever: the query shape shared by ``KeywordIndex.search`` and
    ``TextVectorIndex.search`` (and, after an adapter, the visual index).
    Sources are registered by object — the engine calls ``source.search``.
    """

    def search(
        self,
        query: str,
        top_k: int = 5,
        doc_ids: Optional[Sequence[str]] = None,
    ) -> List[Hit]: ...


@dataclass(frozen=True)
class RankedEvidence:
    """One retriever's contribution to a fused candidate: its 1-based rank
    in that retriever's ranking plus the raw, unscaled score."""

    method: str
    rank: int
    score: float


@dataclass(frozen=True)
class FusedHit:
    """A fused candidate: full chunk metadata, provenance, and per-method
    ranks/scores. Later steps (evidence pack) attach matching images by
    doc_id + page."""

    chunk: Chunk
    fused_rank: int
    fused_score: float
    methods: tuple[str, ...]
    evidence: tuple[RankedEvidence, ...]


@dataclass
class _CandidateState:
    """Internal accumulation state while fusing one chunk."""

    chunk: Chunk
    rrf: float = 0.0
    evidence: List[RankedEvidence] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)


class FusionEngine:
    """RRF over a named set of search sources.

    Sources are registered by name (e.g. ``keyword``, ``vector``, later
    ``visual``); ``search`` runs each retriever independently and merges
    the rankings. A source that raises (e.g. the shared embedding service
    is unavailable) is skipped and recorded in ``degraded`` — the other
    retrievers' results are retained.
    """

    def __init__(
        self,
        sources: Optional[Dict[str, SearchSource]] = None,
        rrf_k: int = RRF_K_DEFAULT,
    ) -> None:
        if not isinstance(rrf_k, int) or isinstance(rrf_k, bool) or rrf_k < 1:
            raise ValueError(f"rrf_k must be a positive integer, got {rrf_k!r}")
        self.registered: Dict[str, SearchSource] = dict(sources or {})
        self.rrf_k = rrf_k
        self.degraded: List[str] = []  # warnings from the last search

    # ---- configuration --------------------------------------------------

    def register(self, name: str, source: SearchSource) -> None:
        """Register (or replace) a search source by name."""
        if not name or not isinstance(name, str):
            raise ValueError(f"source name must be a non-empty string, got {name!r}")
        if source is None:
            raise ValueError(f"source {name!r} must not be None")
        self.registered[name] = source

    # ---- search ---------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        doc_ids: Optional[Sequence[str]] = None,
        candidate_k: Optional[int] = None,
    ) -> List[FusedHit]:
        """Fuse all registered retrievers on ``query``.

        ``candidate_k`` is how many candidates each retriever may yield
        before fusion (default ``max(top_k * 3, 20)`` — enough room for
        RRF to re-rank, capped so we never over-fetch from shared
        services). ``doc_ids`` is forwarded to every retriever unchanged.
        """
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError(f"top_k must be a positive integer, got {top_k!r}")
        per_source = candidate_k if candidate_k is not None else max(top_k * 3, 20)
        text = (query or "").strip()
        if not text or not self.registered:
            return []
        if doc_ids is not None and not list(doc_ids):
            return []  # empty selection: nothing can match

        self.degraded = []
        contrib: Dict[str, _CandidateState] = {}  # chunk_id -> fusion state

        for method, source in self.registered.items():
            try:
                hits = source.search(text, top_k=per_source, doc_ids=doc_ids)
            except Exception as exc:  # retriever failure (e.g. service down)
                self.degraded.append(f"{method}: {exc}")
                continue
            for rank, hit in enumerate(hits, start=1):
                cid = hit.chunk.chunk_id
                state = contrib.setdefault(cid, _CandidateState(chunk=hit.chunk))
                state.rrf += 1.0 / (self.rrf_k + rank)
                state.evidence.append(
                    RankedEvidence(method=method, rank=rank, score=hit.score)
                )
                state.methods.append(method)

        candidates = sorted(
            contrib.values(),
            key=lambda s: (-s.rrf, s.chunk.chunk_id),  # deterministic tie-break
        )[:top_k]
        return [
            FusedHit(
                chunk=state.chunk,
                fused_rank=rank,
                fused_score=state.rrf,
                methods=tuple(state.methods),
                evidence=tuple(state.evidence),
            )
            for rank, state in enumerate(candidates, start=1)
        ]
