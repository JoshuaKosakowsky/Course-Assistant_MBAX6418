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
    # one RRF term per method: keyword(1) + vector(2) + visual(1)
    assert pages[0].page_rrf == pytest.approx(1 / 61 + 1 / 62 + 1 / 61)
    assert pages[0].page_rrf > pages[1].page_rrf  # doc-b: keyword+vector only
    assert pages[1].page_rrf == pytest.approx(1 / 61 + 1 / 62)


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
    # keyword r1 + vector r2 (from mk_hit evidence) + visual r3
    assert pages[0].page_rrf == pytest.approx(1 / 61 + 1 / 62 + 1 / 63)


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


# 19. page-length bias: one contribution per method (Sec.4) ------------------------

def test_many_chunks_do_not_inflate_page_score():
    """A page with TEN relevant chunks must NOT outrank a page with ONE
    equally-relevant chunk — each retrieval method counts once per page."""
    ten_chunks = [mk_hit(mk_chunk(f"c{i}", doc_id="doc-a", page_no=1,
                                  text="same relevance"), fused_rank=1)
                  for i in range(10)]
    one_chunk = [mk_hit(mk_chunk("x", doc_id="doc-b", page_no=1,
                                 text="same relevance"), fused_rank=1)]
    pages = fuse_pages(ten_chunks + one_chunk, [])
    by_doc = {p.doc_id: p for p in pages}
    assert by_doc["doc-a"].page_rrf == pytest.approx(
        by_doc["doc-b"].page_rrf)  # both: keyword r1 + vector r1
    assert len(by_doc["doc-a"].text_chunks) == 10  # provenance preserved


def test_multimethod_boost():
    """A page found by keyword AND vector AND visual gets three terms;
    a page found by one method gets one."""
    multi = [mk_hit(mk_chunk("c1", doc_id="doc-a", page_no=1, text="t"),
                    fused_rank=1, methods=("keyword", "vector"))]
    single = [mk_hit(mk_chunk("c2", doc_id="doc-b", page_no=1, text="t"),
                     fused_rank=1, methods=("keyword",))]
    visual = [make_visual("doc-a", 1, rank=1)]
    pages = fuse_pages(multi + single, visual)
    a = next(p for p in pages if p.doc_id == "doc-a")
    b = next(p for p in pages if p.doc_id == "doc-b")
    assert a.page_rrf == pytest.approx(1 / 61 + 1 / 62 + 1 / 61)
    assert b.page_rrf == pytest.approx(1 / 61)   # keyword only
    assert a.page_rrf > b.page_rrf


def test_ties_deterministic():
    p1 = [mk_hit(mk_chunk("a", doc_id="doc-a", page_no=1), fused_rank=1)]
    p2 = [mk_hit(mk_chunk("b", doc_id="doc-b", page_no=1), fused_rank=1)]
    r1 = fuse_pages(p1 + p2, [])
    r2 = fuse_pages(p1 + p2, [])
    assert [p.doc_id for p in r1] == [p.doc_id for p in r2]
    assert r1[0].page_rrf == r1[1].page_rrf


def test_duplicate_visual_hits_single_contribution():
    """Two visual hits for the SAME page contribute ONE method term."""
    fused = [mk_hit(mk_chunk("c1", doc_id="doc-a", page_no=1), fused_rank=1)]
    dup = [make_visual("doc-a", 1, rank=1, distance=0.9, vid="d1"),
           make_visual("doc-a", 1, rank=2, distance=0.1, vid="d2")]
    pages = fuse_pages(fused, dup)
    assert pages[0].page_rrf == pytest.approx(
        1 / 61 + 1 / 62 + 1 / 61)  # keyword r1 + vector r2 + ONE visual r2
    assert pages[0].visual.visual_id == "d2"  # best distance kept


# 20. provenance tiers: verified vs candidate-only images (Sec.5) ------------------

def test_external_image_is_candidate_only_not_verified(tmp_path):
    store, rec = make_store_with_doc(tmp_path)
    img = tmp_path / "external.png"   # NOT under artifacts; arbitrary name
    make_png(img)
    src = StubSource(hits=[type("VH", (), {"doc_id": rec.doc_id, "page": 1,
                                           "image_path": str(img), "id": "i1",
                                           "distance": 0.1})()])
    cand = VisualSourceAdapter(src, store=store).query_text("q")[0]
    assert cand.image_path == str(img)  # still a retrieval candidate

    fused = [mk_hit(mk_chunk("c1", doc_id=rec.doc_id, page_no=1, text="text"))]
    pages = fuse_pages(fused, [cand], store=store)
    pc = pages[0]
    assert pc.image_source == "visual_index_external"
    assert pc.image_verified is False
    from app.evidence import EvidencePack, item_from_page_candidate
    item = item_from_page_candidate(pc, 1, 0.9, True)
    assert item.image_verified is False
    assert item.image_path is not None          # candidate retains the image
    sources = EvidencePack(query="q", items=[item]).to_sources()
    if sources:  # excerpt exists -> source emitted, but unverified image dropped
        assert sources[0].image_path is None
    assert any("unverified" in w for w in pc.warnings)


def test_artifact_image_is_verified(tmp_path):
    """A manifest-validated artifact image is a VERIFIED original."""
    store, rec = make_store_with_doc(tmp_path)
    adir = store.artifacts_dir / rec.doc_id
    (adir / "pages").mkdir(parents=True)
    img = adir / "pages" / "page-001.png"
    make_png(img)
    import json
    (adir / "manifest.json").write_text(json.dumps({
        "pages": [{"number": 1, "text": "t", "image_path": str(img),
                   "embedded_images": 0}]}), encoding="utf-8")
    fused = [mk_hit(mk_chunk("c1", doc_id=rec.doc_id, page_no=1,
                             text="t", image_path=str(img)), fused_rank=1)]
    pages = fuse_pages(fused, [], store=store)
    assert pages[0].image_source == "artifact"
    assert pages[0].image_verified is True


# 21. production visual query model selection (Sec.3) --------------------------------

def test_unified_visual_embedder_uses_9003_model(monkeypatch):
    import app.config as cfg
    monkeypatch.setattr(cfg, "VISUAL_EMBED_BASE_URL", "http://x.test:9003/v1")
    monkeypatch.setattr(cfg, "VISUAL_EMBED_API_KEY", "test-key")
    calls = {}

    class FakeResp:
        status_code = 200

        def json(self):
            return {"data": [{"embedding": [0.1] * 4}]}

    class FakeSession:
        def post(self, url, headers=None, json=None, timeout=None):
            calls.update(url=url, headers=headers, json=json)
            return FakeResp()

    from app.visual_retrieval import unified_visual_text_embedder
    emb = unified_visual_text_embedder(session=FakeSession())
    vec = emb("what is this chart?")
    assert len(vec) == 4
    assert calls["url"] == "http://x.test:9003/v1/embeddings"
    assert calls["json"]["model"] == "Qwen/Qwen3-VL-Embedding-2B"
    assert calls["json"]["input"] == ["what is this chart?"]
    assert calls["headers"]["Authorization"].startswith("Bearer ")


def test_unified_visual_embedder_rejects_port_9000(monkeypatch):
    import app.config as cfg
    monkeypatch.setattr(cfg, "VISUAL_EMBED_BASE_URL", "http://dobolyi.com:9000/v1")
    monkeypatch.setattr(cfg, "VISUAL_EMBED_API_KEY", "test-key")
    from app.visual_retrieval import unified_visual_text_embedder
    with pytest.raises(ValueError):
        unified_visual_text_embedder()


def test_unified_visual_embedder_requires_key(monkeypatch):
    import app.config as cfg
    monkeypatch.setattr(cfg, "VISUAL_EMBED_BASE_URL", "http://x.test:9003/v1")
    monkeypatch.setattr(cfg, "VISUAL_EMBED_API_KEY", "")
    from app.visual_retrieval import unified_visual_text_embedder, VisualQueryError
    with pytest.raises(VisualQueryError):
        unified_visual_text_embedder()


def test_production_retrieve_injects_unified_embedder(monkeypatch):
    """production_retrieve must select the 9003 unified embedder — never
    silently fall through to the source's own (9002) text bridge."""
    import app.retrieval as retrieval_mod

    injected = []

    def fake_factory(session=None):
        injected.append(True)

        def emb(text):
            return [0.1] * 4
        return emb

    monkeypatch.setattr(retrieval_mod, "unified_visual_text_embedder", fake_factory)
    monkeypatch.setenv("CLASS_API_KEY", "k")

    class FakeKw:
        def search(self, query, top_k=5, doc_ids=None):
            return []

    class FakeVisual:
        def query_embedding(self, embedding, k=5):
            return [type("VH", (), {"doc_id": "doc-1", "page": 3,
                                    "image_path": "", "id": "i1",
                                    "distance": 0.1})()]

    outcome = retrieval_mod.production_retrieve(
        "chart page", keyword=FakeKw(), visual_index=FakeVisual())
    assert injected == [True]
    assert outcome.stages["visual"] == "ok"
    assert any(i.doc_id == "doc-1" for i in outcome.visual_items)


def test_production_retrieve_degrades_when_visual_unconfigured(monkeypatch):
    import app.retrieval as retrieval_mod
    from app.visual_retrieval import VisualQueryError

    def failing(session=None):
        raise VisualQueryError("no key")

    monkeypatch.setattr(retrieval_mod, "unified_visual_text_embedder", failing)

    class FakeKw:
        def search(self, query, top_k=5, doc_ids=None):
            return []

    outcome = retrieval_mod.production_retrieve(
        "q", keyword=FakeKw(), visual_index=object())
    assert outcome.stages["visual"] == "degraded"
    assert any("visual" in w for w in outcome.warnings)


# ---------------------------------------------------------------------------
# F1 (review finding 1): a rejected foreign image must NEVER reach the
# reranker — page_candidate_to_hit may only use pc.image_path (the vetted
# one), never fall back to pc.visual.image_path (which can still hold the
# raw un-vetted path when the adapter had no store to verify ownership)
# ---------------------------------------------------------------------------

def _foreign_store(tmp_path):
    """doc-a registered; doc-b's artifacts hold a real page image that a
    visual hit for doc-a might wrongly point into."""
    store, rec = make_store_with_doc(tmp_path, doc_id="doc-a")
    bdir = store.artifacts_dir / "doc-b" / "pages"
    bdir.mkdir(parents=True, exist_ok=True)
    foreign = bdir / "page-001.png"
    make_png(foreign)
    (store.artifacts_dir / "doc-b" / "manifest.json").write_text(
        json.dumps({"pages": [{"number": 1, "text": "",
                               "image_path": str(foreign),
                               "embedded_images": 0}]}),
        encoding="utf-8")
    return store, rec, foreign


def test_foreign_doc_image_rejected_and_absent_from_rerank_payload(tmp_path):
    from app.rerank import rerank_fused

    store, rec, foreign = _foreign_store(tmp_path)
    # an adapter WITHOUT a store cannot verify ownership, so the raw
    # foreign path survives adaptation (this is the pre-fix danger); the
    # store-aware fusion + page_candidate_to_hit must then reject it
    adapter = VisualSourceAdapter(StubSource(hits=[type(
        "VH", (), {"doc_id": "doc-a", "page": 1,
                   "image_path": str(foreign), "id": "v1",
                   "distance": 0.1})()]), store=None)
    cands = adapter.query_text("q", k=5)
    assert cands and cands[0].image_path == str(foreign)

    fused = [mk_hit(mk_chunk("c1", doc_id="doc-a", page_no=1,
                             text="cash flow"), fused_rank=1)]
    pages = fuse_pages(fused, cands, store=store)
    pc = pages[0]
    assert pc.image_path is None                     # foreign image rejected
    assert any("another document" in w for w in pc.warnings)
    hit = page_candidate_to_hit(pc)
    assert hit.chunk.image_path == ""   # REGRESSION: old code re-attached
                                        # pc.visual.image_path (foreign)

    # reranker payload contains NO image for this hit
    class T:
        payloads = []

        def rank(self, query, documents, top_n=None):
            self.payloads.append(documents)
            return {i: (len(documents) - i) for i in range(len(documents))}

    t = T()
    rerank_fused("q", [hit], t, include_images=True)
    docs = t.payloads[0]
    assert not isinstance(docs[0], dict) or "image_url" not in str(docs[0])
    assert "base64" not in str(docs[0])          # foreign bytes never sent


def test_visual_only_foreign_candidate_never_leaks_raw_path(tmp_path):
    store, rec, foreign = _foreign_store(tmp_path)
    cand = make_visual(doc_id="doc-a", page=1, image_path=str(foreign),
                       rank=1, distance=0.1)
    pages = fuse_pages([], [cand], store=store)   # visual-only page
    assert pages and pages[0].visual is not None
    assert pages[0].image_path is None
    hit = page_candidate_to_hit(pages[0])
    assert hit.chunk.image_path == ""   # NO fallback to pc.visual.image_path


# ---------------------------------------------------------------------------
# F9 (review finding 9): filtering visual retrieval must not starve allowed
# documents ranked below the global top-k — bounded overfetch, then
# filter/dedupe/limit, preserving global rank as provenance
# ---------------------------------------------------------------------------

class CountingSource(StubSource):
    def count(self):
        return len(self._hits)


def _ranked_hits(n, doc_at_rank=None, doc_id="doc-a"):
    hits = []
    for i in range(n):
        doc = doc_id if (doc_at_rank is not None and i == doc_at_rank - 1) \
            else "doc-b"
        hits.append(type("VH", (), {
            "doc_id": doc, "page": (i % 2) + 1, "image_path": "",
            "id": f"v{i}", "distance": 0.01 * (i + 1)})())
    return hits


def test_overfetch_text_route_finds_allowed_hit_below_global_top_k():
    hits = _ranked_hits(14, doc_at_rank=12)   # doc-a only at global rank 12
    src = CountingSource(hits)
    out = VisualSourceAdapter(src).query_text("q", k=2, doc_ids={"doc-a"})
    assert len(out) == 1 and out[0].doc_id == "doc-a"
    assert src.calls and src.calls[0][2] > 2   # overfetched beyond k
    assert out[0].rank == 12                   # global rank preserved


def test_overfetch_embedding_route_finds_allowed_hit_below_global_top_k():
    hits = _ranked_hits(14, doc_at_rank=12)
    src = CountingSource(hits)
    out = VisualSourceAdapter(src).query_embedding([0.1, 0.2], k=2,
                                                   doc_ids={"doc-a"})
    assert len(out) == 1 and out[0].doc_id == "doc-a"


def test_overfetch_dedupes_pages_and_honors_doc_filter():
    hits = _ranked_hits(6, doc_at_rank=None)  # all doc-b
    # interleave a doc-a page twice (duplicate page at ranks 1 and 5)
    hits[0] = type("VH", (), {"doc_id": "doc-a", "page": 1, "image_path": "",
                              "id": "v0", "distance": 0.01})()
    hits[4] = type("VH", (), {"doc_id": "doc-a", "page": 1, "image_path": "",
                              "id": "v4", "distance": 0.05})()
    src = CountingSource(hits)
    out = VisualSourceAdapter(src).query_text("q", k=5, doc_ids={"doc-a"})
    pages = {(c.doc_id, c.page) for c in out}
    assert pages == {("doc-a", 1)}     # duplicate page deduped to one
    assert out[0].rank == 1            # best (global) rank kept


def test_overfetch_deleted_documents_excluded():
    # deleted docs reach the adapter only through doc_ids (retrieve() passes
    # the live set); a deleted doc's hits must be excluded with the filter
    hits = _ranked_hits(6)
    hits[2] = type("VH", (), {"doc_id": "vanished-doc", "page": 1,
                              "image_path": "", "id": "vz",
                              "distance": 0.02})()
    src = CountingSource(hits)
    out = VisualSourceAdapter(src).query_text(
        "q", k=5, doc_ids={"doc-b"})          # vanished-doc NOT in the set
    assert all(c.doc_id != "vanished-doc" for c in out)
    assert {c.doc_id for c in out} == {"doc-b"}
