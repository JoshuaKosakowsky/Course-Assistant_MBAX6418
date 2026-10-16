"""Unit tests for app.retrieval (Issue #4 / Phase 3, Step 3).

Offline orchestration: real KeywordIndex/TextVectorIndex (FakeEmbedder)
for the happy path, controlled stubs for failure modes, and a mock
reranker transport. Live endpoints are covered by opt-in e2e tests.
"""

import json

import pytest

import pymupdf

from app.chunking import PageText, Chunk, chunk_pages
from app.documents import DocumentStore
from app.evidence import EvidencePack
from app.keyword import Hit, KeywordIndex
from app.retrieval import retrieve
from app.vector import TextVectorIndex
from tests.fake_embeddings import FakeEmbedder

pytestmark = pytest.mark.unit

FIN = ("Free cash flow equals operating cash flow minus capital expenditures. "
       "Discounted cash flow valuation estimates enterprise value.")
MKT = ("Brand positioning defines how customers perceive a product. "
       "Customer segmentation groups buyers by behavior and needs.")


def mk_chunk(chunk_id, doc_id="doc-a", page_no=1, text="evidence text",
             filename="a.pdf", image_path=""):
    return Chunk(chunk_id=chunk_id, doc_id=doc_id, filename=filename,
                 format=".pdf", page_no=page_no, index_in_page=0, text=text,
                 image_path=image_path)


class StubSource:
    """Controllable SearchSource (or the raising variant)."""

    def __init__(self, hits=None, error=None):
        self._hits = hits or []
        self._error = error
        self.calls = 0

    def search(self, query, top_k=5, doc_ids=None):
        self.calls += 1
        if self._error is not None:
            raise self._error
        return self._hits[:top_k]


class StubReranker:
    """Stand-in RerankClient; records the request payload."""

    def __init__(self, error=None, scores_fn=None):
        self._error = error
        self._scores_fn = scores_fn or (lambda n: {i: (n - i) / n for i in range(n)})
        self.payloads = []

    def rank(self, query, documents, top_n=None):
        if self._error is not None:
            raise self._error
        self.payloads.append({"query": query, "documents": documents})
        return self._scores_fn(len(documents))


def make_env(tmp_path):
    """Real KeywordIndex + TextVectorIndex over a 2-doc corpus."""
    store = DocumentStore(root=tmp_path / "data")
    kw = KeywordIndex()
    vec = TextVectorIndex(FakeEmbedder(), persist_dir=tmp_path / "db")
    for doc_id, filename, text in (("doc-a", "a.pdf", FIN),
                                   ("doc-b", "b.pdf", MKT)):
        chunks = chunk_pages(doc_id, filename, [PageText(page_no=1, text=text)],
                             chunk_size=80, chunk_overlap=10, format=".pdf")
        kw.add(chunks)
        vec.add(chunks)
    return store, kw, vec


# 1. full retrieval orchestration ---------------------------------------------

def test_full_pipeline(tmp_path):
    store, kw, vec = make_env(tmp_path)
    reranker = StubReranker()
    outcome = retrieve("free cash flow", keyword=kw, vector=vec,
                       reranker=reranker, store=store, top_k=3)
    assert outcome.fused and outcome.evidence.items
    assert outcome.rerank is not None and outcome.rerank.applied
    assert outcome.evidence.items
    assert outcome.stages["keyword"] == "ok"
    assert outcome.stages["semantic"] == "ok"
    assert outcome.stages["fusion"] == "ok"
    assert outcome.stages["rerank"] == "ok"
    assert len(reranker.payloads) == 1  # exactly one rerank request


# 2+3. evidence pack + metadata ------------------------------------------------

def test_evidence_from_pipeline(tmp_path):
    store, kw, vec = make_env(tmp_path)
    outcome = retrieve("free cash flow", keyword=kw, vector=vec,
                       reranker=StubReranker(), store=store, top_k=2)
    item = outcome.evidence.items[0]
    assert item.filename == "a.pdf"
    assert item.page_no == 1
    assert item.excerpt and "cash flow" in item.excerpt
    assert item.fused_rank >= 1 and item.rerank_score is not None
    assert item.methods == ("keyword", "vector")
    assert isinstance(outcome.evidence, EvidencePack)
    assert outcome.evidence.to_sources()  # PR #14-compatible


# 4. source metadata -----------------------------------------------------------

def test_doc_ids_in_evidence(tmp_path):
    store, kw, vec = make_env(tmp_path)
    outcome = retrieve("segmentation", keyword=kw, vector=vec,
                       reranker=StubReranker(), store=store, top_k=5)
    names = {i.filename for i in outcome.evidence.items}
    assert "b.pdf" in names and all(i.doc_id in ("doc-a", "doc-b")
                                    for i in outcome.evidence.items)


# 5. correct image association --------------------------------------------------

def test_image_attached_from_artifacts(tmp_path):
    store, kw, vec = make_env(tmp_path)
    # give doc-a an artifact image + manifest, then ask about cash flow
    adir = store.artifacts_dir / "doc-a"
    (adir / "pages").mkdir(parents=True, exist_ok=True)
    img = adir / "pages" / "page-001.png"
    d = pymupdf.open(); pg = d.new_page(); pg.insert_text((72, 72), "cash flow")
    pg.get_pixmap(dpi=72).save(str(img)); d.close()
    (adir / "manifest.json").write_text(json.dumps({
        "pages": [{"number": 1, "text": FIN, "image_path": str(img),
                   "embedded_images": 0}]}), encoding="utf-8")
    # re-chunk doc-a with the image path
    kw = KeywordIndex(); vec = TextVectorIndex(FakeEmbedder(), persist_dir=tmp_path / "db2")
    chunks = chunk_pages("doc-a", "a.pdf",
                         [PageText(page_no=1, text=FIN, image_path=str(img))],
                         chunk_size=80, chunk_overlap=10, format=".pdf")
    kw.add(chunks); vec.add(chunks)
    outcome = retrieve("free cash flow", keyword=kw, vector=vec,
                       reranker=StubReranker(), store=store, top_k=3)
    top = outcome.evidence.items[0]
    assert top.image_path == str(img)
    assert top.image_source == "artifact"


# 6. wrong-document image rejected ----------------------------------------------

def test_wrong_document_image_rejected_in_pipeline(tmp_path):
    store, kw, vec = make_env(tmp_path)
    # doc-b's image path planted on doc-a's chunk (valid filename)
    bdir = store.artifacts_dir / "doc-b"
    (bdir / "pages").mkdir(parents=True, exist_ok=True)
    foreign = bdir / "pages" / "page-001.png"
    d = pymupdf.open(); pg = d.new_page(); pg.insert_text((72, 72), "x")
    pg.get_pixmap(dpi=72).save(str(foreign)); d.close()
    (bdir / "manifest.json").write_text(json.dumps({
        "pages": [{"number": 1, "text": MKT, "image_path": str(foreign),
                   "embedded_images": 0}]}), encoding="utf-8")

    kw = KeywordIndex(); vec = TextVectorIndex(FakeEmbedder(), persist_dir=tmp_path / "db2")
    chunks = chunk_pages("doc-a", "a.pdf",
                         [PageText(page_no=1, text=FIN, image_path=str(foreign))],
                         chunk_size=80, chunk_overlap=10, format=".pdf")
    kw.add(chunks); vec.add(chunks)
    outcome = retrieve("free cash flow", keyword=kw, vector=vec,
                       reranker=StubReranker(), store=store, top_k=3)
    top = outcome.evidence.items[0]
    assert top.image_path is None
    assert any("wrong-document" in w for w in top.warnings)


# 7. missing/corrupt image handling ----------------------------------------------

def test_corrupt_image_flagged(tmp_path):
    store, kw, vec = make_env(tmp_path)
    adir = store.artifacts_dir / "doc-a"
    (adir / "pages").mkdir(parents=True, exist_ok=True)
    img = adir / "pages" / "page-001.png"
    img.write_bytes(b"\x00\x01 corrupt not png")
    (adir / "manifest.json").write_text(json.dumps({
        "pages": [{"number": 1, "text": FIN, "image_path": str(img),
                   "embedded_images": 0}]}), encoding="utf-8")
    chunks = chunk_pages("doc-a", "a.pdf",
                         [PageText(page_no=1, text=FIN, image_path=str(img))],
                         chunk_size=80, chunk_overlap=10, format=".pdf")
    kw = KeywordIndex(); vec = TextVectorIndex(FakeEmbedder(), persist_dir=tmp_path / "db2")
    kw.add(chunks); vec.add(chunks)
    outcome = retrieve("free cash flow", keyword=kw, vector=vec,
                       reranker=StubReranker(), store=store, top_k=3)
    assert outcome.evidence.items[0].image_path is None
    assert any("corrupt" in w for w in outcome.evidence.items[0].warnings)


# 8. selected-document filters ---------------------------------------------------

def test_doc_filter_passthrough(tmp_path):
    store, kw, vec = make_env(tmp_path)
    outcome = retrieve("free cash flow", keyword=kw, vector=vec,
                       reranker=StubReranker(), store=store,
                       doc_ids=["doc-a"], top_k=3)
    assert all(i.doc_id == "doc-a" for i in outcome.evidence.items)


# 10. correct final ranking (rerank decides) --------------------------------------

def test_reranked_order_wins(tmp_path):
    store, kw, vec = make_env(tmp_path)

    def scores(n):  # reverse the fused order
        return {i: (i + 1) / n for i in range(n)}

    outcome = retrieve("free cash flow", keyword=kw, vector=vec,
                       reranker=StubReranker(scores_fn=scores), store=store, top_k=4)
    ranks = [i.final_rank for i in outcome.evidence.items]
    assert ranks == sorted(ranks)


# 11. semantic retrieval failure ------------------------------------------------

def test_semantic_failure_degrades_to_keyword(tmp_path):
    store, kw, _ = make_env(tmp_path)
    broken = StubSource(error=RuntimeError("embedding service down"))
    outcome = retrieve("free cash flow", keyword=kw, vector=broken,
                       reranker=StubReranker(), store=store, top_k=3)
    assert outcome.stages["semantic"] == "degraded"
    assert outcome.evidence.items  # keyword evidence still usable
    assert any("vector" in w and "embedding service down" in w
               for w in outcome.warnings)


# 12. reranker failure and fallback -----------------------------------------------

def test_reranker_failure_fallback(tmp_path):
    store, kw, vec = make_env(tmp_path)
    outcome = retrieve("free cash flow", keyword=kw, vector=vec,
                       reranker=StubReranker(error=RuntimeError("boom")),
                       store=store, top_k=3)
    assert outcome.stages["rerank"] == "degraded"
    assert outcome.evidence.items
    assert all(i.rerank_score is None for i in outcome.evidence.items)
    assert any("reranker" in w for w in outcome.warnings)


def test_no_reranker_provided(tmp_path):
    store, kw, vec = make_env(tmp_path)
    outcome = retrieve("free cash flow", keyword=kw, vector=vec,
                       reranker=None, store=store, top_k=3)
    assert outcome.stages["rerank"] == "skipped"
    assert outcome.evidence.items and all(i.rerank_score is None
                                          for i in outcome.evidence.items)


# 13. empty query / index ----------------------------------------------------------

def test_empty_query():
    outcome = retrieve("", keyword=StubSource(hits=[Hit(mk_chunk("c"), 0.5)]))
    assert outcome.evidence.items == []
    assert outcome.stages["keyword"] == "skipped"
    assert any("empty query" in w for w in outcome.evidence.warnings)


def test_empty_keyword_index(tmp_path):
    store, _, _ = make_env(tmp_path)
    kw = KeywordIndex()  # empty
    outcome = retrieve("free cash flow", keyword=kw, vector=None, store=store)
    assert outcome.evidence.items == []
    assert outcome.stages["fusion"] == "ok"  # fusion ran; zero candidates is a verdict


# 14. missing supporting excerpts ------------------------------------------------

def test_textless_page_excerpt_none(tmp_path):
    """A candidate whose page had no extractable text yields excerpt=None
    with an explicit warning — never an invented excerpt."""
    store, _, _ = make_env(tmp_path)
    textless = mk_chunk("c0", text="   ", page_no=3)

    class TextlessSource:
        def search(self, query, top_k=5, doc_ids=None):
            return [Hit(chunk=textless, score=0.9)]

    outcome = retrieve("visual chart question", keyword=TextlessSource(),
                       vector=None, store=store, top_k=1)
    assert outcome.evidence.items
    item = outcome.evidence.items[0]
    assert item.excerpt is None
    assert item.page_no == 3
    assert any("no extractable text" in w for w in item.warnings)
    assert outcome.evidence.to_sources() == []  # PR #14 needs non-empty excerpt


# 15. deleted documents ------------------------------------------------------------

def test_deleted_document_flagged(tmp_path):
    """Deleting a document's artifacts must be reported instead of
    attaching a path that no longer exists."""
    import shutil

    store, kw, vec = make_env(tmp_path)
    adir = store.artifacts_dir / "doc-a"
    (adir / "pages").mkdir(parents=True, exist_ok=True)
    img = adir / "pages" / "page-001.png"
    d = pymupdf.open(); pg = d.new_page(); pg.insert_text((72, 72), "cash flow")
    pg.get_pixmap(dpi=72).save(str(img)); d.close()
    (adir / "manifest.json").write_text(json.dumps({
        "pages": [{"number": 1, "text": FIN, "image_path": str(img),
                   "embedded_images": 0}]}), encoding="utf-8")

    kw2 = KeywordIndex(); vec2 = TextVectorIndex(FakeEmbedder(), persist_dir=tmp_path / "db2")
    chunks = chunk_pages("doc-a", "a.pdf",
                         [PageText(page_no=1, text=FIN, image_path=str(img))],
                         chunk_size=80, chunk_overlap=10, format=".pdf")
    kw2.add(chunks); vec2.add(chunks)

    shutil.rmtree(store.artifacts_dir / "doc-a")  # document removed
    outcome = retrieve("free cash flow", keyword=kw2, vector=vec2,
                       reranker=StubReranker(), store=store, top_k=3)
    assert outcome.evidence.items
    top = outcome.evidence.items[0]
    assert top.filename == "a.pdf"
    assert top.image_path is None  # never attach a deleted image
    # rejected at validation (manifest gone with the artifacts, or the
    # file itself gone) — every path ends in "no image" for this doc
    assert any(("missing" in w or "no longer exists" in w) for w in top.warnings)


# 16. consistent top-k --------------------------------------------------------------

def test_top_k_behavior(tmp_path):
    store, kw, vec = make_env(tmp_path)
    o3 = retrieve("free cash flow", keyword=kw, vector=vec,
                  reranker=StubReranker(), store=store, top_k=3)
    o5 = retrieve("free cash flow", keyword=kw, vector=vec,
                  reranker=StubReranker(), store=store, top_k=5)
    assert len(o3.evidence.items) <= 3
    assert len(o5.evidence.items) <= 5
    assert len(o5.evidence.items) >= len(o3.evidence.items)


# 17. evaluation metrics ------------------------------------------------------------

def test_retrieval_metrics_offline(tmp_path):
    from app.evidence import rank_metrics

    store, kw, vec = make_env(tmp_path)
    outcome = retrieve("customer segmentation", keyword=kw, vector=vec,
                       reranker=StubReranker(), store=store, top_k=5)
    m = rank_metrics(outcome.evidence.items, "b.pdf", k=3)
    assert m["top1"] is True or m["position"] in (1, 2, 3)


# 18. retrieved-visual candidates are FUSED at page level -----------------------

def test_visual_candidates_fused_at_page_level(tmp_path):
    """Step-4 semantics: when the visual source succeeds, its candidates
    genuinely participate in ranking (page-level RRF) instead of being
    kept in a separate list."""
    store = DocumentStore(root=tmp_path / "data")
    src = tmp_path / "a.pdf"
    d = pymupdf.open(); pg = d.new_page()
    pg.insert_text((72, 72), "Free cash flow equals operating cash flow minus capex.")
    d.save(str(src)); d.close()
    rec = store.add(src)  # real record; content-addressed id
    kw = KeywordIndex()
    vec = TextVectorIndex(FakeEmbedder(), persist_dir=tmp_path / "db")
    chunks = chunk_pages(rec.doc_id, "a.pdf",
                         [PageText(page_no=1, text=FIN)],
                         chunk_size=80, chunk_overlap=10, format=".pdf")
    kw.add(chunks); vec.add(chunks)

    img = tmp_path / "vis.png"
    d = pymupdf.open(); pg = d.new_page(); pg.insert_text((72, 72), "cash flow chart")
    pg.get_pixmap(dpi=72).save(str(img)); d.close()

    class FakeVisualSource:
        def query_text(self, text, k=5):
            return [type("VH", (), {"doc_id": rec.doc_id, "page": 1,
                                    "image_path": str(img), "id": "v1",
                                    "distance": 0.1})()]

    outcome = retrieve("free cash flow", keyword=kw, vector=vec,
                       reranker=StubReranker(), store=store,
                       visual_source=FakeVisualSource(), top_k=3)
    assert outcome.stages["visual"] == "ok"
    assert outcome.visual_items[0].image_source == "visual_index"
    # the page-1 evidence carries the visual method AND the page image
    top = outcome.evidence.items[0]
    assert top.filename == "a.pdf" and top.page_no == 1
    assert "visual" in top.methods
    assert top.image_path is not None  # visual image became the page image
    assert top.image_source == "visual_index_external"  # unverifiable claim
    assert top.image_verified is False
    assert any(e.method == "visual" for e in top.evidence_ranks)


def test_visual_failure_degrades_to_text_only(tmp_path):
    store, kw, vec = make_env(tmp_path)

    class BrokenVisual:
        def query_text(self, text, k=5):
            raise RuntimeError("visual service down")

    outcome = retrieve("free cash flow", keyword=kw, vector=vec,
                       reranker=StubReranker(), store=store,
                       visual_source=BrokenVisual(), top_k=3)
    assert outcome.stages["visual"] == "degraded"
    assert outcome.evidence.items  # text evidence survives
    assert any("visual" in w for w in outcome.warnings)



# 19. visual-only retrieval runs even when text finds nothing (Sec.2) ------------

def test_visual_only_page_retrieved_without_text(tmp_path):
    """REGRESSION: keyword+semantic return NOTHING, but the visual index
    finds the image-only page — it must reach the reranker payload and
    the evidence pack, with no invented excerpt."""
    store = DocumentStore(root=tmp_path / "data")
    src = tmp_path / "a.pdf"
    d = pymupdf.open(); pg = d.new_page()
    pg.insert_text((72, 72), "free cash flow overview")
    d.save(str(src)); d.close()
    rec = store.add(src)
    kw = KeywordIndex()                # empty: no matches for anything
    vec = TextVectorIndex(FakeEmbedder(), persist_dir=tmp_path / "db")
    # empty vector index too

    img = tmp_path / "chart-page.png"  # visual-only page image
    d = pymupdf.open(); pg = d.new_page()
    pg.draw_rect(pymupdf.Rect(50, 50, 200, 150), color=(0.2, 0.4, 0.9),
                 fill=(0.2, 0.4, 0.9))
    pg.get_pixmap(dpi=72).save(str(img)); d.close()

    class VisualOnlySource:
        def query_embedding(self, embedding, k=5):
            return [type("VH", (), {"doc_id": rec.doc_id, "page": 1,
                                    "image_path": str(img), "id": "v1",
                                    "distance": 0.05})()]

    reranker = StubReranker()
    outcome = retrieve("which page shows a bar chart?",
                       keyword=kw, vector=vec, reranker=reranker,
                       store=store, top_k=3,
                       visual_source=VisualOnlySource(),
                       text_to_visual_embedder=lambda t: [0.1] * 8)
    assert outcome.fused == []                 # text found nothing
    assert outcome.stages["visual"] == "ok"
    assert outcome.stages["rerank"] == "ok"    # page-level rerank ran
    assert outcome.evidence.items, "visual-only evidence must exist"
    top = outcome.evidence.items[0]
    assert top.excerpt is None                 # never invented
    assert top.image_path == str(img)
    assert any("no extractable text" in w for w in top.warnings)
    # the reranker really received the image (multimodal payload)
    docs = reranker.payloads[0]["documents"]
    assert any(isinstance(x, dict) and "image_url" in str(x) for x in docs)


def test_visual_retrieval_respects_doc_filters_without_text(tmp_path):
    store = DocumentStore(root=tmp_path / "data")
    src = tmp_path / "a.pdf"
    d = pymupdf.open(); pg = d.new_page(); pg.insert_text((72, 72), "x")
    d.save(str(src)); d.close()
    rec = store.add(src)

    class VisualSource:
        def query_embedding(self, embedding, k=5):
            return [type("VH", (), {"doc_id": rec.doc_id, "page": 1,
                                    "image_path": "", "id": "v1",
                                    "distance": 0.05})()]

    outcome = retrieve("chart", keyword=KeywordIndex(), vector=None,
                       store=store, visual_source=VisualSource(),
                       text_to_visual_embedder=lambda t: [0.1] * 8,
                       doc_ids=["some-other-doc"])
    assert outcome.visual_items == []  # filtered out client-side


# 16b. fallback evidence bounded by top_k (review finding 6) -----------------

def _seven_chunks():
    """Seven DISTINCT chunks that ALL match the query 'common' (identical
    text, unique ids), so fusion sees 7 candidates for the top_k bound."""
    return [Chunk(chunk_id="c%02d" % i, doc_id="doc-a", filename="a.pdf",
                  format=".pdf", page_no=1, index_in_page=0,
                  text="common", image_path="") for i in range(7)]


class BoomReranker:
    """Transport failure: rerank_fused must fall back to fused order."""

    def rank(self, query, documents, top_n=None):
        raise RuntimeError("connection refused")


def test_no_reranker_fallback_chunk_path_bounded_by_top_k(tmp_path):
    store = DocumentStore(root=tmp_path / "data")
    kw, vec = KeywordIndex(), TextVectorIndex(FakeEmbedder(),
                                              persist_dir=tmp_path / "db")
    chunks = _seven_chunks()
    kw.add(chunks)
    vec.add(chunks)
    o = retrieve("common", keyword=kw, vector=vec, reranker=None,
                 store=store, top_k=2)
    assert len(o.evidence.items) == 2


def test_failing_reranker_chunk_path_bounded_by_top_k(tmp_path):
    store = DocumentStore(root=tmp_path / "data")
    kw, vec = KeywordIndex(), TextVectorIndex(FakeEmbedder(),
                                              persist_dir=tmp_path / "db")
    chunks = _seven_chunks()
    kw.add(chunks)
    vec.add(chunks)
    o = retrieve("common", keyword=kw, vector=vec, reranker=BoomReranker(),
                 store=store, top_k=2)
    assert o.evidence.items and len(o.evidence.items) == 2


def test_no_reranker_visual_page_path_bounded_by_top_k(tmp_path):
    store = DocumentStore(root=tmp_path / "data")
    kw = KeywordIndex()
    chunks = _seven_chunks()
    kw.add(chunks)

    class RankedSource:
        def __init__(self, hits):
            self._hits = hits

        def count(self):
            return len(self._hits)

        def query_text(self, text, k=5):
            return self._hits[:k]

        def query_embedding(self, embedding, k=5):
            return self._hits[:k]

    n = 7
    visual_hits = [type("VH", (), {"doc_id": "doc-a", "page": i + 1,
                                   "image_path": "", "id": "v%d" % i,
                                   "distance": 0.01 * (i + 1)})()
                   for i in range(n)]
    from app.visual_retrieval import VisualSourceAdapter
    o = retrieve("common", keyword=kw, vector=None,
                 visual_source=RankedSource(visual_hits), store=store,
                 top_k=2, reranker=None)
    assert len(o.evidence.items) == 2          # 7 page candidates -> top 2


def test_invalid_top_k_rejected(tmp_path):
    store, kw, vec = make_env(tmp_path)
    for bad in (0, -1, True, "3"):
        with pytest.raises(ValueError):
            retrieve("free cash flow", keyword=kw, vector=vec,
                     store=store, top_k=bad)


# 16c. semantic stage health (review finding 10) ----------------------------

def test_keyword_only_stage_reports_semantic_skipped(tmp_path):
    store, kw, vec = make_env(tmp_path)
    o = retrieve("free cash flow", keyword=kw, vector=None,
                 store=store, top_k=3)
    assert o.stages["keyword"] == "ok"
    assert o.stages["semantic"] == "skipped"   # vector never ran, keyword ran


def test_vector_ok_stage_reports_ok(tmp_path):
    store, kw, vec = make_env(tmp_path)
    o = retrieve("free cash flow", keyword=kw, vector=vec,
                 store=store, top_k=3)
    assert o.stages["keyword"] == "ok"
    assert o.stages["semantic"] == "ok"


def test_failing_vector_stage_reports_degraded(tmp_path):
    class BoomVec:
        def search(self, query, top_k=None):
            raise RuntimeError("embedding service down")

    store, kw, _ = make_env(tmp_path)
    o = retrieve("free cash flow", keyword=kw, vector=BoomVec(),
                 store=store, top_k=3)
    assert o.stages["keyword"] == "ok"
    assert o.stages["semantic"] == "degraded"


def test_keyword_absent_stage_stays_skipped(tmp_path):
    store, kw, vec = make_env(tmp_path)
    o = retrieve("free cash flow", keyword=None, vector=vec,
                 store=store, top_k=3)
    assert o.stages["keyword"] == "skipped"
    assert o.stages["semantic"] == "ok"
