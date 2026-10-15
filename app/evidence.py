"""Evidence pack for Phase 3 (Issue #4, Step 3).

Turns fused+reranked candidates into a self-contained, Q&A-ready
structure: ranked text excerpts + validated original page images + full
source details, while keeping the richer internal provenance for
debugging and evaluation.

Compatibility with teammate PR #14 (app/qna_schema.py, unmerged):

    @dataclass(frozen=True)
    class Source:
        document: str
        page: int
        excerpt: str
        image_path: str | None = None

``EvidencePack.to_sources()`` returns exactly that shape
(``SourceLike(document, page, excerpt, image_path)``). No hard runtime
dependency on the unmerged branch: after PR #14 merges, swap
``SourceLike`` for ``qna_schema.Source`` — the fields are identical.

Visual-only pages (image, no extractable text) cannot be represented as a
PR #14 ``Source`` because its ``excerpt`` is a required non-empty str;
the exported ``EXCERPT_REQUIRED_NOTE`` constant documents this and the
future interface change (``excerpt: str | None``) that would allow
image-only evidence.

Image validation is MANIFEST-AUTHORITATIVE: a filename pattern alone
(e.g. ``page-001.png``) does not prove ownership, so every artifact
image must (1) live under the claiming document's artifact directory,
(2) match the parser's page filename for the claimed page number, and
(3) be recorded for that page in the document's manifest.json (when one
exists). Images that fail are reported as unavailable — never replaced
with a potentially wrong image. Images independently retrieved from the
visual index (future, PR #11) are tagged ``image_source="visual_index"``
and are NOT interchangeable with artifact images.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from app.chunking import Chunk
from app.fusion import FusedHit, RankedEvidence
from app.rerank import RerankedCandidate, _mime_type

# PR #14 compatibility note (see module docstring): its Source.excerpt is a
# required non-empty str, so image-only pages cannot be represented there.
EXCERPT_REQUIRED_NOTE = (
    "PR #14's Source requires a non-empty 'excerpt'; image-only pages keep "
    "their image in the evidence pack but are excluded from Source output "
    "rather than fabricating text. Future PR #14 revision should allow "
    "Source(excerpt=None) for visual-only evidence."
)

__all__ = [
    "EvidencePack", "EvidenceItem", "SourceLike",
    "item_from_fused", "item_from_visual", "item_from_page_candidate",
    "source_position", "validate_artifact_image", "EXCERPT_REQUIRED_NOTE",
]


@dataclass(frozen=True)
class EvidenceItem:
    """One ranked piece of evidence, self-contained for the Q&A phase."""

    doc_id: str
    filename: str
    page_no: int
    excerpt: Optional[str]          # None when the page had no extractable text
    image_path: Optional[str]       # validated path, or None
    chunk_id: str
    final_rank: int
    rerank_score: Optional[float]   # None when reranking did not run
    fused_rank: int
    fused_score: float
    methods: Tuple[str, ...]        # which retrievers found this chunk
    evidence_ranks: Tuple[RankedEvidence, ...]   # per-method rank + raw score
    image_included: bool            # image content was used during reranking
    image_source: str               # "artifact" | "visual_index" | "visual_index_external" | "none"
    image_verified: bool            # safe to cite as an ORIGINAL page image
    warnings: Tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceLike:
    """Field-for-field copy of PR #14's ``qna_schema.Source`` contract."""

    document: str
    page: int
    excerpt: str
    image_path: Optional[str] = None


@dataclass
class EvidencePack:
    """Ranked evidence for one question, plus stage health + warnings."""

    query: str
    items: List[EvidenceItem] = field(default_factory=list)
    stages: Dict[str, str] = field(default_factory=dict)  # ok|degraded|skipped
    warnings: List[str] = field(default_factory=list)

    # ---- PR #14-compatible output ---------------------------------------

    def to_sources(self) -> List[SourceLike]:
        """Q&A prompt/citation sources (exactly PR #14's Source shape).

        Items without an extractable excerpt are excluded here (PR #14
        requires a non-empty ``excerpt``) and stay visible in ``items``
        with an explicit warning instead.
        """
        sources = []
        for item in self.items:
            if not item.excerpt or not item.excerpt.strip():
                continue
            sources.append(
                SourceLike(
                    document=item.filename,
                    page=item.page_no,
                    excerpt=item.excerpt,
                    # ONLY verified original page images are citable;
                    # unverified visual claims stay candidates, never
                    # presented as originals.
                    image_path=item.image_path if item.image_verified else None,
                )
            )
        return sources

    def summary(self) -> Dict[str, object]:
        return {
            "query": self.query,
            "items": len(self.items),
            "stages": dict(self.stages),
            "warnings": list(self.warnings),
        }


# ---- image validation -----------------------------------------------------


def validate_artifact_image(
    chunk: Chunk, store, expected_basename: Optional[str] = None
) -> Tuple[Optional[str], str]:
    """Return (validated image path, note) for a chunk's artifact image.

    The note is '' when the image is usable, otherwise explains why the
    image is NOT attached (missing, corrupt, wrong document, wrong page,
    unrecorded in the manifest). Returns (None, note) on rejection.
    """
    if not chunk.image_path:
        return None, "no image path"

    raw = Path(chunk.image_path)
    image_dir = store.artifacts_dir / chunk.doc_id if store is not None else None

    if store is not None:
        # 1. the image must live under the claiming document's artifacts
        try:
            image_dir = Path(store.artifacts_dir) / chunk.doc_id
            raw.resolve().relative_to(image_dir.resolve())
        except (ValueError, OSError):
            return None, (
                f"image {raw} is outside the artifact directory of "
                f"document {chunk.doc_id} (wrong-document image rejected)"
            )

        # 2. the parser's page filename must match the claimed page
        expected = expected_basename or f"page-{chunk.page_no:03d}.png"
        if raw.name != expected:
            return None, (
                f"image {raw.name!r} does not match the parser's page "
                f"filename {expected!r}"
            )

        # 3. the manifest must record this image for this page
        manifest = image_dir / "manifest.json"
        if manifest.is_file():
            import json

            try:
                pages = json.loads(manifest.read_text(encoding="utf-8")).get("pages", [])
                recorded = [
                    str(p.get("image_path", "")) for p in pages
                    if int(p.get("number", -1)) == chunk.page_no
                ]
            except Exception:
                recorded = []
            if recorded and chunk.image_path not in recorded:
                return None, (
                    f"manifest for {chunk.doc_id} does not record "
                    f"{chunk.image_path} for page {chunk.page_no}"
                )

    # 4. file must exist and decode as PNG/JPEG
    if not raw.is_file():
        return None, "image file no longer exists"
    if _mime_type(raw) is None:
        return None, "unreadable/corrupt image"

    return str(raw), ""


def item_from_fused(
    hit: FusedHit,
    final_rank: int,
    rerank_score: Optional[float],
    image_included: bool,
    store=None,
) -> EvidenceItem:
    """Build one evidence item from a fused (or reranked) candidate."""
    path, warn = validate_artifact_image(hit.chunk, store)
    warnings = [warn] if warn else []
    if not hit.chunk.text.strip():
        warnings.append("page has no extractable text (visual-only page)")
    return EvidenceItem(
        doc_id=hit.chunk.doc_id,
        filename=hit.chunk.filename,
        page_no=hit.chunk.page_no,
        excerpt=hit.chunk.text if hit.chunk.text.strip() else None,
        image_path=path,
        chunk_id=hit.chunk.chunk_id,
        final_rank=final_rank,
        rerank_score=rerank_score,
        fused_rank=hit.fused_rank,
        fused_score=hit.fused_score,
        methods=hit.methods,
        evidence_ranks=hit.evidence,
        image_included=image_included,
        image_source="artifact" if path else "none",
        image_verified=path is not None,
        warnings=tuple(warnings),
    )


def item_from_visual(
    visual_hit,
    final_rank: int = 0,
    store=None,
) -> EvidenceItem:
    """Adapt a VisualIndex hit (PR #11 interface: doc_id, page, image_path,
    distance, id) into an evidence item tagged VISUAL-INDEX origin.

    These are independently retrieved images — NOT the same as artifact
    images attached from text chunks. Nothing here imports PR #11; the
    adapter is duck-typed so it works against any conforming object.
    """
    page = int(getattr(visual_hit, "page", getattr(visual_hit, "page_no", 0)))
    image_path = str(getattr(visual_hit, "image_path", "") or "")
    warnings: List[str] = []
    if not image_path:
        warnings.append("visual hit has no image path")
    else:
        raw = Path(image_path)
        if not raw.is_file():
            warnings.append("visual image file no longer exists")
        elif _mime_type(raw) is None:
            warnings.append("visual image unreadable/corrupt")
    return EvidenceItem(
        doc_id=str(getattr(visual_hit, "doc_id", "")),
        filename=str(getattr(visual_hit, "filename", "")),
        page_no=page,
        excerpt=None,  # visual hits carry no text — never invented
        image_path=image_path if image_path else None,
        chunk_id=str(getattr(visual_hit, "id", f"vis:{image_path}")),
        final_rank=final_rank,
        rerank_score=None,
        fused_rank=0,
        fused_score=0.0,
        methods=(),
        evidence_ranks=(),
        image_included=False,
        image_source="visual_index" if image_path else "none",
        image_verified=False,  # raw visual claim: not proven original
        warnings=tuple(warnings),
    )


def item_from_page_candidate(
    pc,
    final_rank: int,
    rerank_score: Optional[float],
    image_included: bool,
    store=None,
) -> EvidenceItem:
    """Evidence item from a fused PAGE (text chunks + optional visual hit).

    The excerpt is the best text chunk's (None on visual-only pages, with
    an explicit warning — never invented); the image is the vetted page
    image (artifact first, then visual-index); chunk_id stays stable and
    provenance keeps every original method/rank/score."""
    warnings = list(pc.warnings)
    excerpt = None
    if pc.best_chunk is not None and pc.best_chunk.chunk.text.strip():
        excerpt = pc.best_chunk.chunk.text
    else:
        warnings.append("page has no extractable text (visual-only page)")
    if pc.visual is not None and pc.image_source == "visual_index":
        for w in pc.visual.warnings:
            if w not in warnings:
                warnings.append(w)
    evidence_ranks = tuple(pc.best_chunk.evidence) if pc.best_chunk else ()
    if pc.visual is not None:
        # preserve the visual method's own rank + distance in provenance
        evidence_ranks = evidence_ranks + (
            RankedEvidence(method="visual", rank=pc.visual.rank,
                           score=-pc.visual.distance),
        )
    return EvidenceItem(
        doc_id=pc.doc_id,
        filename=pc.filename,
        page_no=pc.page_no,
        excerpt=excerpt,
        image_path=pc.image_path,
        chunk_id=(pc.best_chunk.chunk.chunk_id if pc.best_chunk
                  else f"vis:{pc.doc_id}:{pc.page_no:03d}"),
        final_rank=final_rank,
        rerank_score=rerank_score,
        fused_rank=pc.page_rank,
        fused_score=pc.page_rrf,
        methods=tuple(pc.methods),
        evidence_ranks=evidence_ranks,
        image_included=image_included,
        image_source=pc.image_source,
        image_verified=pc.image_verified,
        warnings=tuple(warnings),
    )


# ---- evaluation helpers -----------------------------------------------------


def source_position(
    ranked: Sequence[object], expected_doc: str, expected_page: Optional[int] = None
) -> Optional[int]:
    """1-based position of the expected source in ``ranked``, or None.

    ``ranked`` items expose either ``.chunk.filename`` (FusedHit) or
    ``.filename``/``.original.chunk.filename`` (EvidenceItem/RerankedCandidate).
    """
    for i, item in enumerate(ranked, start=1):
        if isinstance(item, FusedHit):
            name, page = item.chunk.filename, item.chunk.page_no
        elif hasattr(item, "original"):  # RerankedCandidate
            name, page = item.original.chunk.filename, item.original.chunk.page_no
        else:  # EvidenceItem
            name, page = item.filename, item.page_no
        if name == expected_doc and (expected_page is None or page == expected_page):
            return i
    return None


def rank_metrics(
    ranked: Sequence[object], expected_doc: str, k: int = 3,
    expected_page: Optional[int] = None,
) -> Dict[str, object]:
    """top-1 accuracy + recall@k for one expected source."""
    pos = source_position(ranked, expected_doc, expected_page)
    return {
        "expected": expected_doc,
        "position": pos,
        "top1": pos == 1,
        f"recall@{k}": pos is not None and pos <= k,
    }
