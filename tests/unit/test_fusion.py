"""Unit tests for app.fusion.FusionEngine (Issue #4 / Phase 3, Step 1).

Deterministic and offline: stub search sources produce controlled
rankings, and the existing KeywordIndex/TextVectorIndex (with the
FakeEmbedder) are exercised to prove compatibility with the real Hit /
Chunk structures. No network, no class endpoints, no visual index.
"""

import pytest

from app.chunking import PageText, Chunk, chunk_pages
from app.fusion import (
    FusedHit,
    FusionEngine,
    RankedEvidence,
    SearchSource,
)
from app.keyword import Hit, KeywordIndex
from app.vector import TextVectorIndex
from tests.fake_embeddings import FakeEmbedder

pytestmark = pytest.mark.unit

DOC_FIN = "doc-finance1111"
DOC_MKT = "doc-marketing22"


def mk_chunk(chunk_id, doc_id=DOC_FIN, page_no=1, text="some evidence text",
             filename="finance.pdf", image_path=""):
    return Chunk(
        chunk_id=chunk_id, doc_id=doc_id, filename=filename, format=".pdf",
        page_no=page_no, index_in_page=0, text=text, image_path=image_path,
    )


def stub_source(*chunk_id_orders):
    """A search source returning a fixed ranking (chunk ids in order) for
    every query. Records the last doc_ids filter it received."""
    ranking = chunk_id_orders[0] if chunk_id_orders else ()

    class _Stub:
        def __init__(self):
            self.seen_doc_ids = None

        def search(self, query, top_k=5, doc_ids=None):
            self.seen_doc_ids = list(doc_ids) if doc_ids is not None else None
            return [Hit(chunk=mk_chunk(cid), score=(10.0 - i) / 10.0)
                    for i, cid in enumerate(ranking)][:top_k]

    return _Stub()


class RaisingSource:
    """A search source whose service is down (e.g. embedding endpoint)."""

    def search(self, query, top_k=5, doc_ids=None):
        raise RuntimeError("service unavailable (simulated)")


# 1. RRF formula correctness ----------------------------------------------

def test_rrf_formula():
    kw = stub_source(("a", "b"))  # a@1, b@2
    vec = stub_source(("c", "a"))  # c@1, a@2
    engine = FusionEngine(sources={"keyword": kw, "vector": vec}, rrf_k=60)
    result = engine.search("q", top_k=10)

    by_id = {h.chunk.chunk_id: h for h in result}
    assert len(by_id) == 3
    # a: 1/(60+1) + 1/(60+2) = 1/61 + 1/62
    assert by_id["a"].fused_score == pytest.approx(1 / 61 + 1 / 62)
    # b: 1/(60+2); c: 1/(60+1)
    assert by_id["b"].fused_score == pytest.approx(1 / 62)
    assert by_id["c"].fused_score == pytest.approx(1 / 61)
    # a ranks above c (1/61 + 1/62 > 1/61); b ranks last
    assert [h.chunk.chunk_id for h in result] == ["a", "c", "b"]


def test_rrf_k_is_configurable():
    kw = stub_source(("a",))
    vec = stub_source(("a",))
    engine = FusionEngine(sources={"keyword": kw, "vector": vec}, rrf_k=10)
    hit = engine.search("q")[0]
    assert hit.fused_score == pytest.approx(1 / 11 + 1 / 11)


def test_invalid_rrf_k_rejected():
    with pytest.raises(ValueError, match="rrf_k"):
        FusionEngine(rrf_k=0)


# 2. BM25-only retrieval ---------------------------------------------------

def test_keyword_only():
    kw = stub_source(("a", "b"))
    engine = FusionEngine(sources={"keyword": kw})
    result = engine.search("q")
    assert [h.chunk.chunk_id for h in result] == ["a", "b"]
    assert result[0].methods == ("keyword",)
    assert result[0].evidence[0].method == "keyword"
    assert result[0].evidence[0].rank == 1


# 3. semantic-only retrieval ------------------------------------------------

def test_vector_only():
    vec = stub_source(("x",))
    engine = FusionEngine(sources={"vector": vec})
    result = engine.search("q")
    assert [h.chunk.chunk_id for h in result] == ["x"]
    assert result[0].methods == ("vector",)


# 4. candidates found by both methods ---------------------------------------

def test_both_methods_boosted():
    kw = stub_source(("a", "b"))
    vec = stub_source(("c", "a"))
    engine = FusionEngine(sources={"keyword": kw, "vector": vec})
    result = engine.search("q", top_k=10)
    by_id = {h.chunk.chunk_id: h for h in result}
    # a appears once even though both methods found it
    assert by_id["a"].fused_score == pytest.approx(1 / 61 + 1 / 62)
    assert by_id["a"].methods == ("keyword", "vector")
    assert len(by_id["a"].evidence) == 2


# 5. duplicate removal -------------------------------------------------------

def test_no_duplicate_chunks():
    kw = stub_source(("a", "b"))
    vec = stub_source(("b", "a"))
    engine = FusionEngine(sources={"keyword": kw, "vector": vec})
    result = engine.search("q", top_k=10)
    ids = [h.chunk.chunk_id for h in result]
    assert len(ids) == len(set(ids)) == 2
    assert set(ids) == {"a", "b"}


# 6. deterministic ranking and tie-breaking ----------------------------------

def test_deterministic_tie_break():
    # b (keyword rank 1) and c (vector rank 1) get identical RRF scores
    # (1/61 each); chunk_id decides the order deterministically.
    kw = stub_source(("b",))
    vec = stub_source(("c",))
    engine = FusionEngine(sources={"keyword": kw, "vector": vec})
    r1 = engine.search("q", top_k=10)
    r2 = engine.search("q", top_k=10)
    assert r1[0].fused_score == r1[1].fused_score  # tied
    assert r1[0].fused_score == pytest.approx(1 / 61)
    assert [h.chunk.chunk_id for h in r1] == [h.chunk.chunk_id for h in r2]
    assert [h.chunk.chunk_id for h in r1] == sorted(
        h.chunk.chunk_id for h in r1)  # deterministic tie-break by chunk_id


# 7. document and page metadata preservation --------------------------------

def test_metadata_preserved():
    chunk = mk_chunk("a", doc_id=DOC_FIN, page_no=3, text="cash flow text",
                     filename="finance.pdf", image_path="img/p3.png")

    class _CustomStub:
        def search(self, query, top_k=5, doc_ids=None):
            return [Hit(chunk=chunk, score=0.9)]

    engine = FusionEngine(sources={"keyword": _CustomStub()})
    hit = engine.search("q")[0]
    assert hit.chunk.doc_id == DOC_FIN
    assert hit.chunk.page_no == 3
    assert hit.chunk.image_path == "img/p3.png"
    assert hit.chunk.filename == "finance.pdf"
    assert hit.chunk.format == ".pdf"


# 8. per-index ranks and raw scores preserved -------------------------------

def test_evidence_ranks_and_scores():
    kw = stub_source(("a", "b"))
    vec = stub_source(("b", "a"))
    engine = FusionEngine(sources={"keyword": kw, "vector": vec})
    hit = engine.search("q", top_k=10)
    by_id = {h.chunk.chunk_id: h for h in hit}
    a_ev = by_id["a"].evidence
    assert {(e.method, e.rank) for e in a_ev} == {("keyword", 1), ("vector", 2)}
    assert all(isinstance(e.score, float) for e in a_ev)  # raw scores kept


# 9. document filtering ------------------------------------------------------

def test_document_filtering_forwarded():
    kw = stub_source(("a",))
    vec = stub_source(("a",))
    engine = FusionEngine(sources={"keyword": kw, "vector": vec})
    engine.search("q", doc_ids=[DOC_FIN])
    assert kw.seen_doc_ids == [DOC_FIN]
    assert vec.seen_doc_ids == [DOC_FIN]


def test_empty_doc_ids_yields_nothing():
    kw = stub_source(("a",))
    engine = FusionEngine(sources={"keyword": kw})
    assert engine.search("q", doc_ids=[]) == []


# 10. empty queries and empty indexes -----------------------------------------

def test_empty_and_whitespace_query():
    kw = stub_source(("a",))
    engine = FusionEngine(sources={"keyword": kw})
    assert engine.search("") == []
    assert engine.search("   ") == []


def test_no_sources():
    engine = FusionEngine()
    assert engine.search("q") == []


def test_empty_indexes():
    kw = stub_source(())  # retriever with no candidates at all
    vec = stub_source(())
    engine = FusionEngine(sources={"keyword": kw, "vector": vec})
    assert engine.search("q") == []


# 11. semantic-service failure fallback --------------------------------------

def test_vector_failure_degrades_to_keyword():
    kw = stub_source(("a", "b"))
    vec = RaisingSource()
    engine = FusionEngine(sources={"keyword": kw, "vector": vec})
    result = engine.search("q")
    assert [h.chunk.chunk_id for h in result] == ["a", "b"]
    assert engine.degraded and "vector" in engine.degraded[0]
    # fused score reflects only the surviving retriever
    assert result[0].fused_score == pytest.approx(1 / 61)


def test_all_sources_failing():
    engine = FusionEngine(sources={"keyword": RaisingSource(),
                                   "vector": RaisingSource()})
    assert engine.search("q") == []
    assert len(engine.degraded) == 2


# 12. configurable top-k ------------------------------------------------------

def test_top_k():
    kw = stub_source(("a", "b", "c", "d"))
    engine = FusionEngine(sources={"keyword": kw})
    assert len(engine.search("q", top_k=2)) == 2
    assert len(engine.search("q", top_k=10)) == 4
    with pytest.raises(ValueError, match="top_k"):
        engine.search("q", top_k=0)


def test_candidate_k_caps_fetch():
    kw = stub_source(("a", "b", "c", "d", "e", "f"))
    engine = FusionEngine(sources={"keyword": kw})
    result = engine.search("q", top_k=2, candidate_k=3)
    # only the first 3 candidates were fetched, then top_k applied
    assert len(result) == 2


# 13. compatibility with existing Hit/Chunk structures -------------------------

def test_compatible_with_real_indexes(tmp_path):
    """The real KeywordIndex and TextVectorIndex (offline FakeEmbedder)
    plug into the engine unchanged — proving we reuse existing interfaces."""
    text = ("Free cash flow equals operating cash flow minus capital "
            "expenditures. Discounted cash flow valuation estimates value.")
    chunks = chunk_pages(DOC_FIN, "finance.pdf",
                         [PageText(page_no=1, text=text)],
                         chunk_size=80, chunk_overlap=10, format=".pdf")

    kw = KeywordIndex()
    kw.add(chunks)
    vec = TextVectorIndex(FakeEmbedder(), persist_dir=tmp_path / "db")
    vec.add(chunks)

    engine = FusionEngine(sources={"keyword": kw, "vector": vec})
    result = engine.search("free cash flow", top_k=5)
    assert result  # both retrievers found candidates and fused them
    assert all(isinstance(h, FusedHit) and isinstance(h.chunk, Chunk)
               for h in result)
    assert all(h.chunk.doc_id == DOC_FIN for h in result)
    assert all(any(e.method == "keyword" for e in h.evidence) for h in result)
    assert all(any(e.method == "vector" for e in h.evidence) for h in result)


# 14. isolation from the visual-index implementation ---------------------------

def test_engine_agnostic_to_source_identity():
    """The engine speaks only the SearchSource protocol: a third source
    (in the future MSpanny's VisualIndex via a thin adapter) registers by
    name without any code change. VisualHit has ``page`` while chunks use
    ``page_no`` — an adapter maps it; this test only proves the engine
    treats every source identically."""
    class FakeVisualAdapter:
        """MECHANICAL TEST STUB ONLY — never used in production. Maps a
        VisualHit-like ranking onto Hit objects carrying the image path."""

        def search(self, query, top_k=5, doc_ids=None):
            return [Hit(
                chunk=mk_chunk("vis", doc_id=DOC_MKT, page_no=7,
                               text="slide text for the image",
                               filename="marketing.pdf",
                               image_path="artifacts/x/p7.png"),
                score=0.87,
            )]

    engine = FusionEngine(sources={"keyword": stub_source(()),
                                   "visual": FakeVisualAdapter()})
    result = engine.search("q")
    assert result[0].methods == ("visual",)
    assert result[0].chunk.image_path == "artifacts/x/p7.png"
    assert result[0].chunk.page_no == 7  # adapter normalized page -> page_no
    # just to be explicit about the integration surface
    assert result[0].methods[0] == "visual"


def test_fusion_module_has_no_visual_dependency():
    """The engine must not import or reference the unmerged visual-index
    module — integration happens later through the SearchSource protocol."""
    import app.fusion as fusion
    assert not hasattr(fusion, "VisualIndex")
    assert not hasattr(fusion, "VisualHit")
    assert "visual_index" not in getattr(fusion, "__imports__", "")
