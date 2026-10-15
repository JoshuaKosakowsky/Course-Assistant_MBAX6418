"""Unit tests for app.keyword (Issue #2 / Phase 2 change 2) — BM25.

Synthetic corpus with known contents; offline, no network, no class
endpoints. Marked unit so it runs in the default `pytest` tier.
"""

import pytest

from app.chunking import PageText, chunk_pages
from app.keyword import DEFAULT_TOP_K, Hit, KeywordIndex

pytestmark = pytest.mark.unit

DOC_FIN = "doc-finance1111"
DOC_MKT = "doc-marketing22"
DOC_OPS = "doc-operations33"

FIN_TEXT = (
    "Free cash flow equals operating cash flow minus capital "
    "expenditures. Analysts use discounted cash flow valuation to "
    "estimate enterprise value."
)
MKT_TEXT = (
    "Brand positioning defines how customers perceive a product. "
    "Customer segmentation groups buyers by behavior and needs."
)
OPS_TEXT = (
    "Inventory management balances holding costs and stockouts. "
    "Supply chain logistics optimizes delivery and warehouse flow."
)

FIN_FILE, MKT_FILE, OPS_FILE = "finance.pdf", "marketing.pdf", "operations.pdf"


def make_doc_chunks(doc_id, filename, text, chunk_size=80):
    pages = [PageText(page_no=1, text=text)]
    return chunk_pages(doc_id, filename, pages, chunk_size=chunk_size,
                       chunk_overlap=10, format=".pdf")


@pytest.fixture
def corpus():
    idx = KeywordIndex()
    idx.add(make_doc_chunks(DOC_FIN, FIN_FILE, FIN_TEXT))
    idx.add(make_doc_chunks(DOC_MKT, MKT_FILE, MKT_TEXT))
    idx.add(make_doc_chunks(DOC_OPS, OPS_FILE, OPS_TEXT))
    return idx


# 1. creating an index from valid chunks ----------------------------------

def test_index_created_from_chunks(corpus):
    assert corpus.size >= 3
    assert len(corpus.chunk_ids) == corpus.size
    assert len({c.chunk_id for c in corpus.chunks()}) == corpus.size
    for c in corpus.chunks():
        assert c.doc_id in (DOC_FIN, DOC_MKT, DOC_OPS)


# 2. retrieving a relevant chunk from a simple query ----------------------

def test_retrieves_relevant_chunk(corpus):
    hits = corpus.search("free cash flow")
    assert hits, "finance query should hit at least one chunk"
    assert hits[0].chunk.doc_id == DOC_FIN
    assert "free cash flow" in hits[0].chunk.text.lower()


# 3. relevance order ------------------------------------------------------

def test_relevance_order(corpus):
    hits = corpus.search("brand positioning")
    assert hits[0].chunk.doc_id == DOC_MKT
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True), "results must be ordered best-first"
    assert hits[0].score >= hits[-1].score


def test_more_matches_outrank_single_match(corpus):
    # finance doc mentions free cash flow multiple times; operations
    # doc only shares "flow" which is stopword-free but single
    hits = corpus.search("free cash flow valuation")
    assert hits[0].chunk.doc_id == DOC_FIN


# 4. metadata preserved ---------------------------------------------------

def test_metadata_preserved(corpus):
    hits = corpus.search("inventory management")
    hit = hits[0]
    assert hit.chunk.doc_id == DOC_OPS
    assert hit.chunk.filename == OPS_FILE
    assert hit.chunk.format == ".pdf"
    assert hit.chunk.page_no == 1
    assert hit.chunk.chunk_id.startswith(DOC_OPS + "::")
    assert isinstance(hit.chunk.text, str) and hit.chunk.text


# 5. multiple documents / multiple chunks per document --------------------

def test_multiple_documents_and_chunks():
    idx = KeywordIndex()
    # one doc with three pages, each long enough to split into 2+ chunks
    pages = [
        PageText(page_no=i + 1,
                 text=(" ".join([f"term{i}"] * 40)) + " finance")
        for i in range(3)
    ]
    chunks = chunk_pages(DOC_FIN, FIN_FILE, pages, chunk_size=60, chunk_overlap=5)
    assert chunks[0].doc_id == chunks[-1].doc_id
    idx.add(chunks)
    hits = idx.search("term1 term2", top_k=20)  # cover both pages
    assert len({h.chunk.chunk_id for h in hits}) == len(hits)
    assert len({h.chunk.page_no for h in hits}) >= 2  # hits across pages


# 6. filtering by document id ---------------------------------------------

def test_filter_by_doc_id(corpus):
    hits = corpus.search("free cash flow", doc_ids=[DOC_FIN])
    assert hits and all(h.chunk.doc_id == DOC_FIN for h in hits)

    hits2 = corpus.search("free cash flow", doc_ids=[DOC_MKT])
    assert all(h.chunk.doc_id == DOC_MKT for h in hits2)
    assert "free cash flow" not in (MKT_TEXT.lower())  # sanity: no true match


def test_filter_by_doc_id_empty_set_returns_nothing(corpus):
    assert corpus.search("free cash flow", doc_ids=[]) == []


def test_filter_unknown_doc_id_returns_nothing(corpus):
    assert corpus.search("free cash flow", doc_ids=["nope"]) == []


# 7. empty index ----------------------------------------------------------

def test_empty_index():
    idx = KeywordIndex()
    assert idx.search("anything") == []
    assert idx.size == 0
    assert idx.remove_document("whatever") == 0


# 8. empty / whitespace-only query ----------------------------------------

def test_empty_and_whitespace_query(corpus):
    assert corpus.search("") == []
    assert corpus.search("   \n\t ") == []
    assert corpus.search(None) == []


def test_stopword_only_query(corpus):
    assert corpus.search("the and of for") == []


# 9. no relevant matches --------------------------------------------------

def test_no_matches(corpus):
    hits = corpus.search("quantum entanglement neutrino")
    assert hits == []


# 10. no duplicate chunk ids ----------------------------------------------

def test_no_duplicate_results(corpus):
    hits = corpus.search("cash flow")
    ids = [h.chunk.chunk_id for h in hits]
    assert len(ids) == len(set(ids))


# 11. removing a document -------------------------------------------------

def test_remove_document(corpus):
    removed = corpus.remove_document(DOC_FIN)
    assert removed >= 1
    hits = corpus.search("free cash flow")
    assert all(h.chunk.doc_id != DOC_FIN for h in hits)
    # other documents unaffected
    assert corpus.search("brand positioning")[0].chunk.doc_id == DOC_MKT
    assert DOC_FIN not in {c.doc_id for c in corpus.chunks()}


def test_removal_hook_signature(corpus):
    """Matches DocumentStore.register_removal_hook(Callable[[str], None])"""
    hook = corpus.remove_document
    removed = hook(DOC_OPS)  # called exactly like the future store hook
    assert removed >= 1
    assert corpus.search("inventory") == []


# 12. rebuilding after changes --------------------------------------------

def test_rebuild_after_add_remove(tmp_path):
    idx = KeywordIndex()
    idx.add(make_doc_chunks(DOC_FIN, FIN_FILE, FIN_TEXT))
    assert idx.search("valuation")
    idx.remove_document(DOC_FIN)
    assert idx.search("valuation") == []
    idx.add(make_doc_chunks(DOC_MKT, MKT_FILE, MKT_TEXT))
    assert idx.search("brand positioning")[0].chunk.doc_id == DOC_MKT


# 13. reindexing with different chunk settings ----------------------------

def test_reindex_with_new_chunk_settings(corpus):
    # NOTE: positional chunk ids like `::c000` may coincide across
    # settings, so id-disjointness is NOT the correctness criterion —
    # replace_all() (clear, then index) is, because it guarantees no
    # stale chunks survive a settings change.
    old_texts = {c.chunk_id: c.text for c in corpus.chunks()}
    new_chunks = make_doc_chunks(DOC_FIN, FIN_FILE, FIN_TEXT, chunk_size=400)
    corp = make_doc_chunks(DOC_MKT, MKT_FILE, MKT_TEXT, chunk_size=400)
    ops = make_doc_chunks(DOC_OPS, OPS_FILE, OPS_TEXT, chunk_size=400)
    corpus.replace_all(new_chunks + corp + ops)
    assert corpus.size == len(new_chunks) + len(corp) + len(ops)
    # every surviving chunk matches the NEW settings (no stale text)
    for c in corpus.chunks():
        assert len(c.text) <= 400 + 10  # new chunk_size + overlap
    hits = corpus.search("discounted cash flow valuation")
    assert hits and hits[0].chunk.doc_id == DOC_FIN
    # any old chunk text is gone
    assert all(c.text != old_texts.get(c.chunk_id) for c in corpus.chunks())


def test_replace_all_drops_old_stale_chunks(corpus):
    corpus.replace_all([])
    assert corpus.size == 0
    assert corpus.search("anything") == []


# 14. configurable top-k --------------------------------------------------

def test_top_k(corpus):
    one = corpus.search("cash", top_k=1)
    three = corpus.search("cash", top_k=3)
    assert len(one) == 1
    assert len(three) <= 3
    assert len(three) >= len(one)


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "3"])
def test_invalid_top_k_rejected(corpus, bad):
    with pytest.raises(ValueError, match="top_k"):
        corpus.search("cash", top_k=bad)


def test_default_top_k_is_five():
    assert DEFAULT_TOP_K == 5


# extra: hit shape --------------------------------------------------------

def test_hit_carries_chunk_and_score(corpus):
    hit = corpus.search("inventory management")[0]
    assert isinstance(hit, Hit)
    assert isinstance(hit.score, float) and hit.score > 0.0
    assert hit.chunk.doc_id == DOC_OPS
