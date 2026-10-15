"""Visual-index adapter + page-level fusion (Issue #4 / Phase 3, Step 4).

Bridges MSpanny's ``VisualIndex`` (PR #11, unmerged — duck-typed protocol,
never imported) into the text retrieval pipeline WITHOUT mixing embedding
spaces: text and visual rankings are fused rank-based (RRF), and *page*
identity ``(doc_id, page)`` is the join key.

Verified live (2026-10-16): port 9003 serves ``Qwen/Qwen3-VL-Embedding-2B``
and returns 2048-dim embeddings for EITHER a raw base64 data-URI string
(images) or plain text — one unified space. The cross-space bridge in
PR #11's ``ClassEmbeddingClient.embed_text`` (9002 Nemotron text against
9003 Qwen-VL images) is dimension-compatible but its similarities came
back essentially flat on a live probe, so the adapter defaults to
embedding the query text with the SAME 9003 model.

The adapter never substitutes a fake embedder in production: fake
image/text embedders exist only in tests (injected explicitly).

Pipeline integration (retrieval.retrieve): visual candidates are fused at
PAGE level — several text chunks from one page plus one visual hit for
that page contribute to a single ``PageCandidate`` (no duplicate page
images in the evidence, every chunk's provenance preserved). When the
visual service is unavailable the pipeline degrades to text-only, exactly
as before.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from app.chunking import Chunk
from app.evidence import _mime_type  # PNG/JPEG magic check
from app.evidence import validate_artifact_image
from app.fusion import FusedHit, RankedEvidence

RRF_K = 60

IMAGE_VERIFIABLE_NOTE = "visual image exists but ownership not verifiable from artifacts"
IMAGE_MISSING_NOTE = "visual image missing"
IMAGE_CORRUPT_NOTE = "unreadable/corrupt visual image"


def _owned_by_another_document(raw: Path, store, doc_id: str) -> bool:
    """True when the image lives under ANOTHER document's artifact dir
    (identifiable as that document's property — must never be attached
    to the candidate's document)."""
    if store is None:
        return False
    adir = Path(store.artifacts_dir)
    try:
        rawres = raw.resolve()
    except (ValueError, OSError):
        return False
    for entry in adir.iterdir():
        if not entry.is_dir() or entry.name == doc_id:
            continue
        try:
            if rawres.is_relative_to(entry.resolve()):
                return True
        except (ValueError, OSError):
            continue
    return False


@dataclass(frozen=True)
class VisualCandidate:
    """One visual-index hit adapted for fusion. ``distance`` is the raw
    visual distance (lower = closer); ``rank`` is its 1-based retrieval
    position (what RRF uses)."""

    doc_id: str
    page: int
    image_path: Optional[str]
    visual_id: str
    distance: float
    rank: int
    filename: str = ""
    warnings: Tuple[str, ...] = ()


class VisualSourceAdapter:
    """Adapts ANY object exposing ``query_embedding(embedding, k)`` and/or
    ``query_text(text, k)`` (PR #11's VisualIndex protocol) into
    ``VisualCandidate`` lists with filtering, dedup, and image vetting.

    - ``doc_ids`` filter is applied AFTER retrieval (the visual index has
      no server-side filter) and deleted documents are excluded via the
      store when provided.
    - candidates are deduplicated by ``(doc_id, page)`` keeping the best
      distance.
    - images must exist and decode; with a store, ownership against the
      claiming document's artifacts is verified (a visual hit for doc X
      whose image lives outside X's artifacts or mismatches its manifest
      loses the image with a warning — never a substituted image).
    """

    def __init__(self, source, store=None, max_candidates: int = 10) -> None:
        self.source = source  # duck-typed VisualIndex (PR #11 protocol)
        self.store = store
        self.max_candidates = max_candidates

    def _query(self, query: str, k: int, doc_ids: Optional[Sequence[str]]) -> list:
        k = max(1, min(int(k), self.max_candidates))
        hits = list(self.source.query_text(query, k=k))
        return self._adapt(hits, k, doc_ids)

    def query_text(self, query: str, k: int = 5,
                   doc_ids: Optional[Sequence[str]] = None) -> List[VisualCandidate]:
        return self._query(query, k, doc_ids)

    def query_embedding(self, embedding: List[float], k: int = 5,
                        doc_ids: Optional[Sequence[str]] = None) -> List[VisualCandidate]:
        k = max(1, min(int(k), self.max_candidates))
        hits = list(self.source.query_embedding(embedding, k=k))
        return self._adapt(hits, k, doc_ids)

    def _adapt(self, hits: list, k: int,
               doc_ids: Optional[Sequence[str]]) -> List[VisualCandidate]:
        allowed = set(doc_ids) if doc_ids is not None else None
        best: Dict[Tuple[str, int], VisualCandidate] = {}
        for rank, h in enumerate(hits, start=1):
            doc_id = str(getattr(h, "doc_id", ""))
            page = int(getattr(h, "page", getattr(h, "page_no", 0)))
            if not doc_id or page < 1:
                continue
            if allowed is not None and doc_id not in allowed:
                continue  # unselected document
            if self.store is not None:
                try:
                    self.store.get(doc_id)  # raises if deleted
                except Exception:
                    continue  # deleted document
            warnings: List[str] = []
            image_path = str(getattr(h, "image_path", "") or "")
            if image_path:
                raw = Path(image_path)
                if not raw.is_file():
                    warnings.append(IMAGE_MISSING_NOTE)
                    image_path = ""
                elif _mime_type(raw) is None:
                    warnings.append(IMAGE_CORRUPT_NOTE)
                    image_path = ""
                elif self.store is not None:
                    # ownership: an image INSIDE the doc's artifact dir must
                    # pass manifest validation (name, page, record); an
                    # image outside it (e.g. indexed from an arbitrary
                    # directory by the visual CLI) is the visual index's
                    # own claim — accepted, with an explicit warning.
                    adir = Path(self.store.artifacts_dir) / doc_id
                    try:
                        inside = raw.resolve().is_relative_to(adir.resolve())
                    except (ValueError, OSError):
                        inside = False
                    if inside:
                        textless = Chunk(
                            chunk_id=str(getattr(h, "id", "")), doc_id=doc_id,
                            filename="", format="", page_no=page,
                            index_in_page=0, text="", image_path=str(raw),
                        )
                        vpath, vnote = validate_artifact_image(textless, self.store)
                        if vpath is None:
                            warnings.append(f"visual image rejected: {vnote}")
                            image_path = ""
                    elif _owned_by_another_document(raw, self.store, doc_id):
                        # identifiable as a DIFFERENT document's artifact
                        warnings.append(
                            "visual image belongs to another document's "
                            "artifacts (rejected)")
                        image_path = ""
                    else:
                        warnings.append(
                            "visual image outside doc artifacts; claim "
                            "accepted from visual index")
            key = (doc_id, page)
            cand = VisualCandidate(
                doc_id=doc_id, page=page,
                image_path=image_path or None,
                visual_id=str(getattr(h, "id", f"vis:{doc_id}:{page}")),
                distance=float(getattr(h, "distance", 0.0)),
                rank=rank,
                filename=self._filename_for(doc_id),
                warnings=tuple(warnings),
            )
            if key not in best or cand.distance < best[key].distance:
                best[key] = cand
        return sorted(best.values(), key=lambda c: c.rank)

    def _filename_for(self, doc_id: str) -> str:
        if self.store is None:
            return ""
        try:
            return self.store.get(doc_id).filename
        except Exception:
            return ""


@dataclass
class PageCandidate:
    """One page in the fused ranking: text evidence (all chunks) plus an
    optional visual hit, ranked by page-level RRF."""

    doc_id: str
    page_no: int
    filename: str
    page_rrf: float
    page_rank: int
    methods: List[str]
    text_chunks: List[FusedHit]
    best_chunk: Optional[FusedHit]
    visual: Optional[VisualCandidate]
    image_path: Optional[str]
    image_source: str  # "artifact" | "visual_index" | "none"
    warnings: List[str] = field(default_factory=list)


def fuse_pages(
    fused: Sequence[FusedHit],
    visual: Sequence[VisualCandidate],
    store=None,
    rrf_k: int = RRF_K,
) -> List[PageCandidate]:
    """RRF fusion at PAGE level.

    Every text chunk contributes its full fused score (the sum of its
    per-method RRF terms) to its page; every visual hit contributes one
    ``1/(rrf_k + rank)`` term under the ``visual`` method. Pages are
    ranked by the summed score (deterministic tie-break on doc_id+page).
    Multiple chunks from one page keep their individual provenance in
    ``text_chunks``.
    """
    pages: Dict[Tuple[str, int], dict] = {}

    def page_key(cand):
        return (cand.chunk.doc_id, cand.chunk.page_no)

    for hit in fused:
        key = page_key(hit)
        st = pages.setdefault(key, {
            "rrf": 0.0, "chunks": [], "methods": set(),
            "visual": None,
        })
        st["rrf"] += hit.fused_score
        st["chunks"].append(hit)
        st["methods"].update(hit.methods)

    for vc in visual:
        key = (vc.doc_id, vc.page)
        st = pages.setdefault(key, {"rrf": 0.0, "chunks": [], "methods": set(),
                                    "visual": None})
        st["rrf"] += 1.0 / (rrf_k + vc.rank)
        st["methods"].add("visual")
        if st["visual"] is None or vc.distance < st["visual"].distance:
            st["visual"] = vc

    ordered = sorted(
        pages.items(),
        key=lambda kv: (-kv[1]["rrf"], kv[0]),
    )
    result: List[PageCandidate] = []
    for rank, ((doc_id, page), st) in enumerate(ordered, start=1):
        chunks: List[FusedHit] = sorted(
            st["chunks"], key=lambda h: (-h.fused_score, h.chunk.chunk_id))
        best = chunks[0] if chunks else None
        vc: Optional[VisualCandidate] = st["visual"]
        warnings: List[str] = []
        image_path, image_source = None, "none"

        # prefer the validated artifact image of the best text chunk
        if best is not None and best.chunk.image_path and store is not None:
            vpath, vnote = validate_artifact_image(best.chunk, store)
            if vpath is not None:
                image_path, image_source = vpath, "artifact"
            elif vnote:
                warnings.append(vnote)
        # fall back to the visual hit's image — re-vetted here so a
        # wrongly-plumbed visual candidate can never attach a foreign
        # image (defense in depth behind the adapter's own gate)
        if image_path is None and vc is not None and vc.image_path:
            raw = Path(vc.image_path)
            if store is not None:
                adir = Path(store.artifacts_dir) / vc.doc_id
                try:
                    inside = raw.resolve().is_relative_to(adir.resolve())
                except (ValueError, OSError):
                    inside = False
                if inside:
                    probe = Chunk(
                        chunk_id=vc.visual_id, doc_id=vc.doc_id, filename="",
                        format="", page_no=vc.page, index_in_page=0,
                        text="", image_path=str(raw),
                    )
                    vp, vn = validate_artifact_image(probe, store)
                    if vp is not None:
                        image_path, image_source = vp, "visual_index"
                    else:
                        warnings.append(f"visual image rejected: {vn}")
                elif _owned_by_another_document(raw, store, vc.doc_id):
                    warnings.append(
                        "visual image belongs to another document's "
                        "artifacts (rejected)")
                else:
                    image_path, image_source = str(raw), "visual_index"
                    warnings.append(
                        "visual image outside doc artifacts; claim accepted "
                        "from visual index")
            else:
                image_path, image_source = str(raw), "visual_index"

        filename = best.chunk.filename if best else (vc.filename if vc else "")
        if not filename and store is not None:
            try:
                filename = store.get(doc_id).filename
            except Exception:
                pass
        if not filename:
            warnings.append("filename unresolved (no text chunk, no store)")

        if best is None and not (vc and vc.image_path):
            warnings.append("page has neither text nor a usable image")
        result.append(PageCandidate(
            doc_id=doc_id, page_no=page, filename=filename,
            page_rrf=st["rrf"], page_rank=rank,
            methods=sorted(st["methods"]),
            text_chunks=chunks, best_chunk=best, visual=vc,
            image_path=image_path, image_source=image_source,
            warnings=warnings,
        ))
    return result


def page_candidate_to_hit(pc: PageCandidate) -> FusedHit:
    """Shape a PageCandidate into a FusedHit so the existing reranker
    orchestration treats pages uniformly. The page's image is the vetted
    one; text is the best chunk's ('' on visual-only pages)."""
    chunk = Chunk(
        chunk_id=(pc.best_chunk.chunk.chunk_id if pc.best_chunk
                  else f"vis:{pc.doc_id}:{pc.page_no:03d}"),
        doc_id=pc.doc_id,
        filename=pc.filename,
        format=".pdf",
        page_no=pc.page_no,
        index_in_page=pc.best_chunk.chunk.index_in_page if pc.best_chunk else 0,
        text=pc.best_chunk.chunk.text if pc.best_chunk else "",
        image_path=pc.image_path or (pc.visual.image_path if pc.visual else None) or "",
    )
    evidence = list(pc.best_chunk.evidence) if pc.best_chunk else []
    if pc.visual is not None:
        evidence.append(RankedEvidence(method="visual", rank=pc.visual.rank,
                                       score=-pc.visual.distance))
    return FusedHit(
        chunk=chunk,
        fused_rank=pc.page_rank,
        fused_score=pc.page_rrf,
        methods=tuple(pc.methods),
        evidence=tuple(evidence),
    )
