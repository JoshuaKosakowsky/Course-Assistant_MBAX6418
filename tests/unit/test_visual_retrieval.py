"""Unit tests for app.visual_retrieval (Issue #4 / Phase 3, Step 4).

Adapter + page-level fusion mechanics with deterministic stubs. The
REAL VisualIndex class (PR #11, unmerged) is exercised separately in
tests/smoke/test_real_visual_index.py via a read-only worktree checkout
with injected fake embedders — never mixed into this branch's code.
"""

import json
from pathlib import Path

import pytest

import pymupdf

from app.chunking import Chunk, PageText, chunk_pages
from app.documents import DocumentStore
from app.fusion import FusedHit, RankedEvidence
from app.keyword import KeywordIndex
from app.vector import TextVectorIndex
from app.visual_retrieval import (
    VisualSourceAdapter,
    fuse_pages,
    page_candidate_to_hit,
)
from tests.fake_embeddings import FakeEmbedder

pytestmark = pytest.mark.unit


def mk_chunk(chunk_id, doc_id="doc-a", page_no=1, text="cash flow text",
             filename="a.pdf", image_path=""):
    return Chunk(chunk_id=chunk_id, doc_id=doc_id, filename=filename,
                 format=".pdf", page_no=page_no, index_in_page=0, text=text,
                 image_path=image_path)


def mk_hit(chunk, fused_rank=1, methods=("keyword", "vector")):
    return FusedHit(
        chunk=chunk, fused_rank=fused_rank, fused_score=0.05,
        methods=methods,
        evidence=tuple(RankedEvidence(m, rank, 0.5)
                       for rank, m in enumerate(methods, start=1)),
    )


def make_visual(doc_id="doc-a", page=1, image_path="", distance=0.5, rank=1,
                vid="v1"):
    from app.visual_retrieval import VisualCandidate
    return VisualCandidate(doc_id=doc_id, page=page, image_path=image_path,
                           visual_id=vid, distance=distance, rank=rank,
                           filename="", warnings=())


def make_png(path):
    d = pymupdf.open(); pg = d.new_page(); pg.insert_text((72, 72), "chart")
    pg.get_pixmap(dpi=72).save(str(path)); d.close()


def make_store_with_doc(tmp_path, doc_id="doc-a", filename="a.pdf"):
    """A DocumentStore with ONE registered content-addressed document."""
    store = DocumentStore(root=tmp_path / "data")
    src = tmp_path / f"{filename}"
    d = pymupdf.open(); pg = d.new_page(); pg.insert_text((72, 72), "content")
    d.save(str(src)); d.close()
    rec = store.add(src)
    return store, rec


class StubSource:
    """A VisualIndex-protocol stub: query_text/query_embedding -> hits."""

    def __init__(self, hits=None, error=None):
        self._hits = hits or []
        self._error = error
        self.calls = []

    def query_text(self, text, k=5):
        self.calls.append(("text", text, k))
        if self._error is not None:
            raise self._error
        return self._hits[:k]

    def query_embedding(self, embedding, k=5):
        self.calls.append(("embedding", len(embedding), k))
        if self._error is not None:
            raise self._error
        return self._hits[:k]


# 1-2. adapter: retrieval through the protocol + hit adaptation ----------

def test_adapter_adapts_visual_hits():
    hits = [type("VH", (), {"doc_id": "doc-x", "page": 4,
                            "image_path": "", "id": "i1", "distance": 0.3})()]
    src = StubSource(hits=hits)
    out = VisualSourceAdapter(src).query_text("q", k=5)
    assert len(out) == 1
    c = out[0]
    assert (c.doc_id, c.page) == ("doc-x", 4)
    assert c.visual_id == "i1" and c.distance == 0.3 and c.rank == 1
    assert c.image_path is None  # no image -> dropped


def test_adapter_uses_query_embedding_path():
    src = StubSource()
    VisualSourceAdapter(src).query_embedding([0.1, 0.2], k=4)
    assert src.calls[0][0] == "embedding" and src.calls[0][1] == 2


# 3. combining keyword/text/visual rankings (page RRF) ---------------------

def test_page_fusion_combines_rankings():
    fused = [
        mk_hit(mk_chunk("c1", page_no=1, text="cash flow"),
               fused_rank=1, methods=("keyword", "vector")),
        mk_hit(mk_chunk("c2", doc_id="doc-b", page_no=1), fused_rank=2),
    ]
    visual = [make_visual(doc_id="doc-a", page=1, rank=1, distance=0.1)]
    pages = fuse_pages(fused, visual)
    assert pages[0].doc_id == "doc-a" and pages[0].page_no == 1
    assert "visual" in pages[0].methods
    # page rrf = text fused score (2 methods) + 1/(60+1)
    assert pages[0].page_rrf > 0.05
    assert pages[1].doc_id == "doc-b"


# 4. document/page matching --------------------------------------------------

def test_page_identity_joins_text_and_visual():
    fused = [mk_hit(mk_chunk("c1", doc_id="doc-a", page_no=3))]
    visual = [make_visual(doc_id="doc-a", page=3, rank=1)]
    pages = fuse_pages(fused, visual)
    assert len(pages) == 1
    assert pages[0].text_chunks[0].chunk.chunk_id == "c1"
    assert pages[0].visual.visual_id == "v1"
    # a visual hit for a DIFFERENT page stays separate
    pages2 = fuse_pages(fused, [make_visual(doc_id="doc-a", page=4, rank=1)])
    assert len(pages2) == 2


# 5. multiple text chunks from one page --------------------------------------

def test_multiple_chunks_one_page_kept():
    fused = [mk_hit(mk_chunk("c1", page_no=1), fused_rank=1),
             mk_hit(mk_chunk("c2", page_no=1), fused_rank=3)]
    pages = fuse_pages(fused, [])
    assert len(pages) == 1
    assert len(pages[0].text_chunks) == 2  # provenance preserved


# 6. no duplicate page evidence ------------------------------------------------

def test_visual_dedupe_by_doc_page():
    v1 = make_visual(doc_id="doc-a", page=1, distance=0.8, rank=1, vid="bad")
    v2 = make_visual(doc_id="doc-a", page=1, distance=0.2, rank=2, vid="good")
    src = StubSource(hits=[type("VH", (), {"doc_id": d.doc_id, "page": d.page,
                                           "image_path": "", "id": d.visual_id,
                                           "distance": d.distance})()
                           for d in (v1, v2)])
    out = VisualSourceAdapter(src).query_text("q", k=5)
    assert len(out) == 1  # deduplicated
    assert out[0].distance == 0.2  # best kept


# 7. visual-only pages -----------------------------------------------------------

def test_visual_only_page_survives_fusion():
    visual = [make_visual(doc_id="doc-a", page=7, rank=1)]
    pages = fuse_pages([], visual)  # no text chunks at all
    assert len(pages) == 1
    assert pages[0].best_chunk is None
    # page_candidate_to_hit must still produce a rerankable candidate
    hit = page_candidate_to_hit(pages[0])
    assert hit.chunk.page_no == 7 and hit.chunk.text == ""  # never invented
    assert "visual" in hit.methods


# 8. selected-document filtering ---------------------------------------------------

def test_doc_filter_excludes_unselected():
    hits = [type("VH", (), {"doc_id": d, "page": 1, "image_path": "",
                            "id": f"i{i}", "distance": 0.1})()
            for i, d in enumerate(("doc-a", "doc-b"))]
    out = VisualSourceAdapter(StubSource(hits=hits)).query_text(
        "q", k=5, doc_ids=["doc-a"])
    assert [c.doc_id for c in out] == ["doc-a"]


# 9. deleted-document filtering -----------------------------------------------------

def test_deleted_document_excluded(tmp_path):
    store = DocumentStore(root=tmp_path / "data")
    src = tmp_path / "a.pdf"
    d = pymupdf.open(); pg = d.new_page(); pg.insert_text((72, 72), "x")
    d.save(str(src)); d.close()
    rec = store.add(src)
    hits = [type("VH", (), {"doc_id": rec.doc_id, "page": 1,
                            "image_path": "", "id": "i1", "distance": 0.1})()]
    out = VisualSourceAdapter(StubSource(hits=hits), store=store).query_text("q")
    assert len(out) == 1
    store.remove(rec.doc_id)  # now deleted
    out2 = VisualSourceAdapter(StubSource(hits=hits), store=store).query_text("q")
    assert out2 == []


# 10-11. image ownership rejection + validation -----------------------------------------

def test_wrong_document_artifact_image_rejected(tmp_path):
    """A visual hit inside the WRONG document's artifact dir loses its image."""
    store = DocumentStore(root=tmp_path / "data")
    adir_a = store.artifacts_dir / "doc-a"
    adir_b = store.artifacts_dir / "doc-b"
    (adir_b / "pages").mkdir(parents=True, exist_ok=True)
    foreign = adir_b / "pages" / "page-001.png"
    make_png(foreign)
    # chunk claims doc-a but image lives under doc-b's artifacts
    chunk = mk_chunk("c0", doc_id="doc-a", page_no=1, image_path=str(foreign))
    fused = [mk_hit(chunk, fused_rank=1)]
    pages = fuse_pages(fused, [make_visual("doc-a", 1, str(foreign))], store=store)
    assert pages[0].image_path is None  # never substitute the wrong image
    assert any("wrong-document" in w for w in pages[0].warnings)


def test_external_visual_image_accepted_with_warning(tmp_path):
    """An image the visual CLI indexed from an arbitrary directory is the
    index's own claim: accepted (with a warning), not hard-rejected."""
    store, rec = make_store_with_doc(tmp_path)
    img = tmp_path / "frontpage.png"
    make_png(img)
    src = StubSource(hits=[type("VH", (), {"doc_id": rec.doc_id, "page": 1,
                                           "image_path": str(img), "id": "i1",
                                           "distance": 0.1})()])
    out = VisualSourceAdapter(src, store=store).query_text("q")
    assert out[0].image_path == str(img)
    assert any("outside doc artifacts" in w for w in out[0].warnings)


# 12-13. missing/corrupt images + failure fallback --------------------------------------

def test_missing_and_corrupt_visual_images():
    hits = [type("VH", (), {"doc_id": "doc-a", "page": 1, "image_path": p,
                            "id": f"i{i}", "distance": 0.1})()
            for i, p in ((1, "/no/such/page.png"), (2, ""))]
    out = VisualSourceAdapter(StubSource(hits=hits)).query_text("q")
    assert all(c.image_path is None for c in out)
    assert len(out) == 1  # pathless + missing -> merged onto one page


def test_visual_source_error_propagates_for_caller_handling():
    src = StubSource(error=RuntimeError("index down"))
    with pytest.raises(RuntimeError, match="index down"):
        VisualSourceAdapter(src).query_text("q")


# 14. incompatible vector dimensions do not matter (rank-based only) -----------

def test_fusion_is_rank_based_not_vector_based():
    """RRF uses positions only: text and visual scores live on different
    scales/dimensions and are never compared directly."""
    fused = [mk_hit(mk_chunk("c1", page_no=1), fused_rank=1)]
    # absurd fake distances/dims are irrelevant to RRF
    visual = [make_visual(doc_id="doc-a", page=1, rank=3, distance=12345.0)]
    pages = fuse_pages(fused, visual)
    expected = 0.05 + 1.0 / (60 + 3)
    assert pages[0].page_rrf == pytest.approx(expected)


# 15. provenance preserved -----------------------------------------------------------

def test_provenance_preserved_in_page_candidate():
    fused = [mk_hit(mk_chunk("c1", page_no=1, text="cash"),
                    fused_rank=1, methods=("keyword", "vector"))]
    visual = [make_visual("doc-a", 1, rank=2, distance=0.3)]
    pages = fuse_pages(fused, visual)
    pc = pages[0]
    assert pc.text_chunks[0].fused_rank == 1
    hit = page_candidate_to_hit(pc)
    assert hit.methods == ("keyword", "vector", "visual")
    assert any(e.method == "visual" and e.rank == 2 for e in hit.evidence)


# 16. genuine visual influence on ranking order ----------------------------------------

def test_visual_can_reorder_pages():
    """A page that text ranks SECOND can be pulled FIRST by a strong
    visual hit (its RRF contribution)."""
    text_p1 = mk_hit(mk_chunk("c1", doc_id="doc-a", page_no=1), fused_rank=1)
    text_p2 = mk_hit(mk_chunk("c2", doc_id="doc-a", page_no=2),
                     fused_rank=2)
    # p2 gets a strong visual contribution (rank 1)
    visual = [make_visual("doc-a", 2, rank=1, distance=0.05)]
    pages = fuse_pages([text_p1, text_p2], visual)
    assert pages[0].doc_id == "doc-a" and pages[0].page_no == 2
    assert "visual" in pages[0].methods


# 17. reranker receives the page's correct image -------------------------------------

def test_page_image_flows_to_rerank_payload(tmp_path):
    from app.rerank import rerank_fused

    img = tmp_path / "p2.png"
    make_png(img)

    class T:
        payloads = []

        def rank(self, query, documents, top_n=None):
            self.payloads.append(documents)
            return {i: (len(documents) - i) / len(documents)
                    for i in range(len(documents))}

    t = T()
    fused = [mk_hit(mk_chunk("c1", doc_id="doc-a", page_no=2,
                             image_path=str(img)), fused_rank=1)]
    visual = [make_visual("doc-a", 2, str(img), rank=1)]
    pages = fuse_pages(fused, visual)
    hit = page_candidate_to_hit(pages[0])

    def page_validator(chunk):
        # mirrors retrieve()'s page-level validator: the page image was
        # vetted at fusion time; guard existence/decoding only (the name
        # need not follow the parser's page-NNN convention)
        if not chunk.image_path:
            return None, "no image path"
        p = Path(chunk.image_path)
        return (str(p), "") if p.is_file() else (None, "image missing")

    outcome = rerank_fused("q", [hit], t, include_images=True,
                           validate_image=page_validator)
    docs = t.payloads[0]
    assert isinstance(docs[0], dict)  # multimodal: text + image
    assert "image_url" in str(docs[0])
    assert outcome.ranked[0].image_included is True


# 18. Q&A evidence format via page candidates ------------------------------------------

def test_page_evidence_compatible_with_source_contract(tmp_path):
    from app.evidence import item_from_page_candidate, rank_metrics

    fused = [mk_hit(mk_chunk("c1", doc_id="doc-a", page_no=2, text="real text"),
                    fused_rank=1)]
    pages = fuse_pages(fused, [])
    pc = pages[0]
    item = item_from_page_candidate(pc, final_rank=1, rerank_score=0.9,
                                    image_included=False)
    assert (item.filename, item.page_no, item.excerpt) == ("a.pdf", 2, "real text")
    s = item_from_page_candidate.__module__  # sanity: same module family
    assert s == "app.evidence"
    # rank_metrics works on the page-evidence item type
    m = rank_metrics([item], "a.pdf", k=3, expected_page=2)
    assert m["top1"] is True
