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
    item_from_visual,
)
from app.fusion import FusedHit, FusionEngine
from app.rerank import RerankClient, RerankOutcome, rerank_fused


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
    max_images: int = 10,
) -> RetrievalOutcome:
    """Run the full text pipeline for one question and build its evidence.

    ``keyword``/``vector`` are SearchSource objects (KeywordIndex /
    TextVectorIndex instances — the FusionEngine protocol). ``store`` is
    a DocumentStore used for manifest-authoritative image validation.
    ``visual_source`` is an OPTIONAL dependency-injected adapter exposing
    ``query_text(text, k=...) -> [VisualHit-like]`` (PR #11's interface);
    its results are reported as visual_items (image_source="visual_index")
    but NOT merged into the text ranking — visual fusion is a later step
    awaiting PR #11 + port 9003.
    """
    outcome = RetrievalOutcome(query=query)
    text = (query or "").strip()
    if not text:
        outcome.stages = {s: "skipped" for s in
                          ("keyword", "semantic", "fusion", "rerank")}
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

    # 4. multimodal reranking (class endpoint; failure -> fused fallback)
    outcome.stages["rerank"] = "skipped"
    if reranker is not None and outcome.fused:
        try:
            outcome.rerank = rerank_fused(
                text, outcome.fused, reranker, top_k=top_k,
                include_images=include_images, max_images=max_images,
            )
            outcome.stages["rerank"] = (
                "ok" if outcome.rerank.applied else "degraded"
            )
            if outcome.rerank.degraded:
                outcome.warnings.extend(outcome.rerank.degraded)
        except Exception as exc:  # defensive: orchestration never crashes
            outcome.stages["rerank"] = "degraded"
            outcome.warnings.append(f"reranking failed: {exc}")

    # 5. evidence pack from the best ranking available
    if outcome.rerank is not None and outcome.rerank.applied:
        ranked = outcome.rerank.ranked
        build = lambda c, rank: item_from_fused(
            c.original, final_rank=rank,
            rerank_score=c.rerank_score,
            image_included=c.image_included, store=store,
        )
    else:
        ranked = outcome.fused
        build = lambda h, rank: item_from_fused(
            h, final_rank=rank, rerank_score=None,
            image_included=False, store=store,
        )
    items = [build(c, i) for i, c in enumerate(ranked, start=1)] if ranked else []
    outcome.evidence = EvidencePack(
        query=text, items=items, stages=dict(outcome.stages),
        warnings=list(outcome.warnings),
    )

    # 6. optional visual-index retrieval (reported separately, not fused)
    outcome.stages["visual"] = "skipped"
    if visual_source is not None:
        try:
            vhits = visual_source.query_text(text, k=top_k)
            outcome.stages["visual"] = "ok"
            outcome.visual_items = [
                item_from_visual(vh, i, store=store)
                for i, vh in enumerate(vhits, start=1)
            ]
        except Exception as exc:
            outcome.stages["visual"] = "degraded"
            outcome.warnings.append(f"visual retrieval failed: {exc}")

    return outcome
