"""End-to-end retrieval orchestration (Issue #4 / Phase 3, Step 3).

Connects the verified pipeline:

    question -> BM25 + text-embedding retrieval -> RRF fusion
              -> multimodal reranking (class endpoint)
              -> evidence pack (ranked excerpts + validated page images)

Every stage's health is tracked (ok / degraded / skipped) so consumers
know exactly what happened. The pipeline degrades honestly: with no
semantic index it fuses keyword results alone; with no reranker (or a
failed one) it returns the fused ranking with ``rerank_score=None``;
with no candidates it returns an empty pack — never simulated results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from app.evidence import (
    EvidenceItem,
    EvidencePack,
    item_from_fused,
    item_from_page_candidate,
    item_from_visual,
    validate_artifact_image,
)
from app.fusion import FusedHit, FusionEngine
from app.rerank import RerankClient, RerankOutcome, rerank_fused
from app.visual_retrieval import (
    VisualSourceAdapter,
    VisualQueryError,
    fuse_pages,
    page_candidate_to_hit,
    unified_visual_text_embedder,
)


def _page_image_validator():
    """Ownership validator for PAGE-level candidates: the image was vetted
    at fusion time (artifact-authoritative or visual-index claim); here we
    only re-guard existence/decoding before the reranker request."""

    def validator(chunk) -> tuple:
        from pathlib import Path

        raw = chunk.image_path
        if not raw:
            return None, "no image path"
        p = Path(raw)
        if not p.is_file():
            return None, "image file no longer exists"
        from app.evidence import _mime_type

        if _mime_type(p) is None:
            return None, "unreadable/corrupt image"
        return str(p), ""

    return validator


@dataclass
class RetrievalOutcome:
    """Everything the caller needs: rankings, rerank verdict, evidence,
    and an honest account of which stages actually ran."""

    query: str
    fused: List[FusedHit] = field(default_factory=list)
    rerank: Optional[RerankOutcome] = None
    evidence: EvidencePack = field(default_factory=lambda: EvidencePack(""))
    visual_items: List[EvidenceItem] = field(default_factory=list)
    stages: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


def retrieve(
    query: str,
    *,
    keyword,
    vector=None,
    reranker: Optional[RerankClient] = None,
    store=None,
    top_k: int = 5,
    candidate_k: int = 20,
    doc_ids: Optional[Sequence[str]] = None,
    include_images: bool = True,
    visual_source=None,
    visual_k: int = 5,
    text_to_visual_embedder=None,
    max_images: int = 10,
) -> RetrievalOutcome:
    """Run the full text pipeline for one question and build its evidence.

    ``keyword``/``vector`` are SearchSource objects (KeywordIndex /
    TextVectorIndex instances — the FusionEngine protocol). ``store`` is
    a DocumentStore used for manifest-authoritative image validation.

    ``visual_source`` is an OPTIONAL adapter source exposing the PR #11
    VisualIndex protocol (``query_text(query, k)`` / ``query_embedding``).
    When provided and running, visual candidates are FUSED at page level
    (rank-based RRF on ``(doc_id, page)``) and the reranker/evidence run
    over pages — the visual results genuinely participate in ranking.
    ``text_to_visual_embedder`` (callable str -> vector) enables the
    verified unified-space bridge: the query is embedded by the SAME
    visual model (port 9003) and searched via ``query_embedding``.
    If visual retrieval fails or is absent, the pipeline degrades to the
    text-only chunk path unchanged.
    """
    outcome = RetrievalOutcome(query=query)
    text = (query or "").strip()
    if not text:
        outcome.stages = {s: "skipped" for s in
                          ("keyword", "semantic", "fusion", "rerank", "visual")}
        outcome.evidence = EvidencePack(
            query=query, stages=dict(outcome.stages),
            warnings=["empty query"],
        )
        return outcome

    # 1. keyword + 2. semantic retrieval happen INSIDE the fusion engine
    # (one pass, so shared services are called exactly once); their health
    # is derived from FusionEngine.degraded afterwards.
    outcome.stages["keyword"] = "skipped" if keyword is None else "pending"
    outcome.stages["semantic"] = "skipped" if vector is None else "pending"

    sources: Dict[str, object] = {}
    if keyword is not None:
        sources["keyword"] = keyword
    if vector is not None:
        sources["vector"] = vector

    # 3. RRF fusion over whatever retrievers actually produced results
    outcome.stages["fusion"] = "skipped"
    engine = FusionEngine(sources=sources) if sources else None
    if engine is not None:
        outcome.fused = engine.search(text, top_k=top_k + 5,  # slightly wide
                                      candidate_k=candidate_k,
                                      doc_ids=doc_ids)
        degraded = [d for d in engine.degraded]
        outcome.stages["keyword"] = (
            "degraded" if any(d.startswith("keyword:") for d in degraded) else "ok"
        )
        outcome.stages["semantic"] = (
            "degraded" if any(d.startswith("vector:") for d in degraded) else "ok"
        )
        outcome.stages["fusion"] = "ok"
        for d in engine.degraded:
            outcome.warnings.append(f"retrieval degraded: {d}")

    # 5. visual-index retrieval — FUSED at page level, not a side list.
    # Runs INDEPENDENTLY of text retrieval: visual-only pages must be
    # discoverable even when keyword/vector search find nothing.
    visual_candidates = []
    pages: Optional[list] = None
    outcome.stages["visual"] = "skipped"
    if visual_source is not None:
        try:
            adapter = VisualSourceAdapter(visual_source, store=store)
            if text_to_visual_embedder is not None:
                # verified unified-space bridge (9003 Qwen3-VL: images and
                # text live in one 2048-dim space) — embed the query with
                # the visual model itself, then query by embedding
                visual_candidates = adapter.query_embedding(
                    text_to_visual_embedder(text), k=visual_k, doc_ids=doc_ids)
            else:
                # NO silent cross-space fallback: using the source's own
                # text bridge (PR #11's query_text) is flagged explicitly,
                # because the class 9002 bridge is dimension-compatible but
                # NOT in the visual model's space. production_retrieve()
                # always injects the 9003 unified embedder instead.
                outcome.warnings.append(
                    "visual query used the source's own text bridge — for "
                    "class endpoints this is the INCOMPATIBLE 9002 space; "
                    "inject the 9003 unified embedder (production_retrieve "
                    "does this automatically)"
                )
                visual_candidates = adapter.query_text(
                    text, k=visual_k, doc_ids=doc_ids)
            outcome.stages["visual"] = "ok"
            outcome.visual_items = [
                item_from_visual(vc, i, store=store)
                for i, vc in enumerate(visual_candidates, start=1)
            ]
            # page-level fusion only when visual candidates exist; with
            # none, the chunk-level text path is unchanged
            if visual_candidates:
                pages = fuse_pages(outcome.fused, visual_candidates, store=store)
        except Exception as exc:
            outcome.stages["visual"] = "degraded"
            outcome.warnings.append(f"visual retrieval failed: {exc}")

    page_hits = None
    hit_to_page = {}
    if pages is not None:
        page_hits = [page_candidate_to_hit(pc) for pc in pages]
        hit_to_page = {h.chunk.chunk_id: pc for h, pc in zip(page_hits, pages)}

    # 4. multimodal reranking (class endpoint; failure -> fused fallback).
    # Images are ownership-validated BEFORE any request is encoded: page
    # candidates use the image vetted at fusion time; chunk candidates use
    # the manifest-authoritative artifact validation.
    outcome.stages["rerank"] = "skipped"
    rank_inputs = page_hits if page_hits is not None else outcome.fused
    if reranker is not None and rank_inputs:
        if page_hits is not None:
            validate_image = _page_image_validator()
        else:
            validate_image = (
                (lambda c: validate_artifact_image(c, store))
                if store is not None else None
            )
        try:
            outcome.rerank = rerank_fused(
                text, rank_inputs, reranker, top_k=top_k,
                include_images=include_images, max_images=max_images,
                validate_image=validate_image,
            )
            outcome.stages["rerank"] = (
                "ok" if outcome.rerank.applied else "degraded"
            )
            if outcome.rerank.degraded:
                outcome.warnings.extend(outcome.rerank.degraded)
        except Exception as exc:  # defensive: orchestration never crashes
            outcome.stages["rerank"] = "degraded"
            outcome.warnings.append(f"reranking failed: {exc}")

    # 6. evidence pack from the best ranking available
    items = []
    if page_hits is not None:
        if outcome.rerank is not None and outcome.rerank.applied:
            for rank, c in enumerate(outcome.rerank.ranked, start=1):
                pc = hit_to_page.get(c.original.chunk.chunk_id)
                if pc is None:  # pragma: no cover - defensive
                    continue
                items.append(item_from_page_candidate(
                    pc, rank, c.rerank_score, c.image_included, store))
        else:
            for rank, pc in enumerate(pages, start=1):
                items.append(item_from_page_candidate(
                    pc, rank, None, False, store))
    elif outcome.rerank is not None and outcome.rerank.applied:
        for rank, c in enumerate(outcome.rerank.ranked, start=1):
            items.append(item_from_fused(
                c.original, final_rank=rank, rerank_score=c.rerank_score,
                image_included=c.image_included, store=store))
    else:
        for rank, h in enumerate(outcome.fused, start=1):
            items.append(item_from_fused(
                h, final_rank=rank, rerank_score=None,
                image_included=False, store=store))

    outcome.evidence = EvidencePack(
        query=text, items=items, stages=dict(outcome.stages),
        warnings=list(outcome.warnings),
    )

    return outcome


def production_retrieve(
    query: str,
    *,
    keyword,
    vector=None,
    store=None,
    reranker=None,
    visual_index=None,
    visual_k: int = 5,
    top_k: int = 5,
    candidate_k: int = 20,
    doc_ids=None,
    include_images: bool = True,
    max_images: int = 10,
) -> RetrievalOutcome:
    """PRODUCTION wiring for the full pipeline.

    Chooses the CORRECT class services automatically:
      - text-semantic retrieval: 9002 Nemotron (via ``vector``),
      - visual queries: the SAME 9003 model that indexed the visual
        vectors (unified space) — NEVER the incompatible 9002 bridge.
    If the 9003 query embedder cannot be configured, visual retrieval is
    reported degraded (a warning is appended) and the text pipeline
    continues. Fake embedders are never constructed here — offline tests
    use ``retrieve`` directly with injected stubs.
    """
    text_to_visual = None
    pre_warning = None
    if visual_index is not None:
        try:
            text_to_visual = unified_visual_text_embedder()
        except VisualQueryError as exc:
            visual_index = None  # degrade loudly, keep text pipeline
            pre_warning = f"visual retrieval disabled: {exc}"
    outcome = retrieve(
        query, keyword=keyword, vector=vector, store=store,
        reranker=reranker, top_k=top_k, candidate_k=candidate_k,
        doc_ids=doc_ids, include_images=include_images,
        visual_source=visual_index, visual_k=visual_k,
        text_to_visual_embedder=text_to_visual, max_images=max_images,
    )
    if pre_warning:
        outcome.warnings.insert(0, pre_warning)
        outcome.stages["visual"] = "degraded"
    return outcome


# ---------------------------------------------------------------------------
# Application-level retrieval service (Issue #4, Step 7)
# ---------------------------------------------------------------------------

class RetrievalService:
    """Stable retrieval entry-point the Q&A UI (later issue) can call.

    Wires the application's REAL initialized services into the pipeline:
    keyword index, text-vector index, optional visual backend, document
    store, and a lazy multimodal-reranker factory (constructed on first
    use, so application startup makes no network calls and the service
    degrades to text-only evidence if the reranker is unavailable).

    Visual queries ALWAYS use the 9003 unified text embedder (injected by
    production_retrieve); the incompatible 9002 bridge is never used.
    """

    def __init__(
        self,
        keyword,
        vector=None,
        store=None,
        visual_backend=None,
        reranker_factory=None,
    ):
        self.keyword = keyword
        self.vector = vector
        self.store = store
        self.visual_backend = visual_backend
        # reranker_factory: () -> RerankClient; None = reranking disabled
        self._reranker_factory = reranker_factory

    def retrieve(self, query, *, top_k=5, doc_ids=None, include_images=True):
        reranker = None
        pre_warning = None
        if self._reranker_factory is not None:
            try:
                reranker = self._reranker_factory()
            except Exception as exc:
                pre_warning = f"reranker unavailable: {exc}"
        outcome = production_retrieve(
            query,
            keyword=self.keyword,
            vector=self.vector,
            store=self.store,
            reranker=reranker,
            visual_index=self.visual_backend,
            top_k=top_k,
            doc_ids=doc_ids,
            include_images=include_images,
        )
        if pre_warning:
            # the plain pipeline reports 'skipped' for a missing reranker;
            # surface the real reason explicitly
            outcome.stages["rerank"] = "degraded"
            outcome.warnings.append(pre_warning)
        return outcome


def build_retrieval_service(indexer, *, reranker_factory=None, visual_backend=None):
    """Wire a RetrievalService from an initialized CorpusIndexer.

    ``indexer`` may optionally carry a VisualIndexer (indexer.visual); its
    backend becomes the visual retrieval source. ``visual_backend``
    overrides that when given. ``reranker_factory`` defaults to a factory
    for the class RerankClient (lazy; no startup network).

    Pass in injected fakes for offline tests.
    """
    backend = visual_backend
    if backend is None and getattr(indexer, "visual", None) is not None:
        backend = indexer.visual.backend
    if reranker_factory is None:
        from app.rerank import RerankClient

        def default_factory():
            return RerankClient()

        reranker_factory = default_factory
    return RetrievalService(
        keyword=indexer.keyword,
        vector=indexer.vector,
        store=indexer.store,
        visual_backend=backend,
        reranker_factory=reranker_factory,
    )
