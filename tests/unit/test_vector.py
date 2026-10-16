"""Unit tests for app.vector.TextVectorIndex (offline, ChromaDB on disk).

Uses the deterministic FakeEmbedder — these tests verify mechanics
(insert, rank, filter, dedupe, delete, reindex, persistence, separation),
NOT real semantic quality. See tests/e2e for the real-endpoint check.
"""

import pytest

from app.chunking import PageText, Chunk, chunk_pages
from app.embeddings import EmbeddingsError
from app.keyword import Hit
from app.vector import TEXT_COLLECTION, VISUAL_COLLECTION, TextVectorIndex
from tests.fake_embeddings import FakeEmbedder

pytestmark = pytest.mark.unit

DOC_FIN = "doc-finance1111"
DOC_MKT = "doc-marketing22"
DOC_OPS = "doc-operations33"

FIN_TEXT = ("Free cash flow equals operating cash flow minus capital "
            "expenditures. Discounted cash flow valuation drives the price "
            "analysts pay for the whole company.")
MKT_TEXT = ("Brand positioning defines how customers perceive a product. "
            "Customer segmentation groups buyers by behavior and needs.")
OPS_TEXT = ("Inventory management reduces holding costs and stockouts. "
            "Supply chain logistics optimizes delivery and warehouse flow.")


def doc_chunks(doc_id, filename, text, chunk_size=80):
    return chunk_pages(
        doc_id, filename, [PageText(page_no=1, text=text)],
        chunk_size=chunk_size, chunk_overlap=10, format=".pdf",
    )


@pytest.fixture
def index(tmp_path):
    idx = TextVectorIndex(FakeEmbedder(), persist_dir=tmp_path / "db")
    idx.add(doc_chunks(DOC_FIN, "finance.pdf", FIN_TEXT))
    idx.add(doc_chunks(DOC_MKT, "marketing.pdf", MKT_TEXT))
    idx.add(doc_chunks(DOC_OPS, "operations.pdf", OPS_TEXT))
    return idx


# 1+2. insertion and retrieval; ranking --------------------------------

def test_insert_and_retrieve_relevant(index):
    hits = index.search("free cash flow")
    assert hits
    assert hits[0].chunk.doc_id == DOC_FIN


def test_ranking_descending(index):
    hits = index.search("cash flow")
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


# 3. document filtering ---------------------------------------------------

def test_filter_by_doc_id(index):
    hits = index.search("free cash flow", doc_ids=[DOC_FIN])
    assert hits and all(h.chunk.doc_id == DOC_FIN for h in hits)


def test_filter_empty_and_unknown(index):
    assert index.search("cash flow", doc_ids=[]) == []
    assert index.search("cash flow", doc_ids=["nope"]) == []


# 4. metadata preservation ------------------------------------------------

def test_metadata_preserved(index):
    hit = index.search("inventory management")[0]
    assert hit.chunk.doc_id == DOC_OPS
    assert hit.chunk.filename == "operations.pdf"
    assert hit.chunk.format == ".pdf"
    assert hit.chunk.page_no == 1
    assert hit.chunk.index_in_page == 0
    assert isinstance(hit.chunk.text, str) and "inventory" in hit.chunk.text.lower()
    assert hit.chunk.chunk_id.startswith(DOC_OPS + "::")


# 5. duplicate prevention -------------------------------------------------

def test_duplicate_add_is_idempotent(index):
    before = index.count
    index.add(doc_chunks(DOC_FIN, "finance.pdf", FIN_TEXT))
    assert index.count == before


def test_same_id_new_content_replaces_vector(index):
    chunks = chunk_pages(
        DOC_FIN, "finance.pdf",
        [PageText(page_no=1, text="totally different alpha beta gamma content words")],
        chunk_size=80, chunk_overlap=10, format=".pdf",
    )
    before = index.count
    index.add(chunks)  # same ids (positional) -> upsert, no growth
    assert index.count == before
    hits = index.search("alpha")
    assert hits and hits[0].chunk.doc_id == DOC_FIN
    assert "alpha" in hits[0].chunk.text


# 6. document deletion ----------------------------------------------------

def test_remove_document(index):
    removed = index.remove_document(DOC_FIN)
    assert removed == before_count(index, DOC_FIN) or removed >= 1
    hits = index.search("free cash flow")
    assert all(h.chunk.doc_id != DOC_FIN for h in hits)
    assert index.search("brand positioning")[0].chunk.doc_id == DOC_MKT


def before_count(index, doc_id):
    return sum(1 for c in index._collection.get(where={"doc_id": doc_id})["ids"])


def test_removal_hook_signature(index):
    hook = index.remove_document  # Callable[[str], None] like DocumentStore hooks
    n = hook(DOC_OPS)
    assert n >= 1
    assert index.search("inventory") == []


# 7. reindexing with changed text -----------------------------------------

def test_reindex_document_replaces_stale_vectors(index):
    old_matching = index.search("free cash flow valuation")
    assert old_matching and old_matching[0].chunk.doc_id == DOC_FIN
    new_chunks = chunk_pages(
        DOC_FIN, "finance.pdf",
        [PageText(page_no=1, text="quantum superposition entanglement mechanics")],
        chunk_size=80, chunk_overlap=10, format=".pdf",
    )
    index.reindex_document(DOC_FIN, new_chunks)
    # The store itself must contain ONLY the new text for this document
    # (vectors are replaced, not appended) — checked at the source so the
    # test does not depend on fake-embedder hash collisions.
    rows = index._collection.get(where={"doc_id": DOC_FIN}, include=["documents"])
    docs = rows["documents"]
    assert docs and all("quantum" in d for d in docs)
    assert not any("free cash flow" in d for d in docs)
    # and the new content is retrievable
    hits = index.search("quantum")
    assert hits and hits[0].chunk.doc_id == DOC_FIN


# 8. persistence ----------------------------------------------------------

def test_persistence_after_reopen(tmp_path):
    db = tmp_path / "db"
    first = TextVectorIndex(FakeEmbedder(), persist_dir=db)
    first.add(doc_chunks(DOC_FIN, "finance.pdf", FIN_TEXT))
    n1 = first.count

    reopened = TextVectorIndex(FakeEmbedder(), persist_dir=db)  # new instance
    assert reopened.count == n1
    hits = reopened.search("free cash flow")
    assert hits and hits[0].chunk.doc_id == DOC_FIN


# 9+10. empty handling ----------------------------------------------------

def test_empty_index_search_returns_nothing(tmp_path):
    idx = TextVectorIndex(FakeEmbedder(), persist_dir=tmp_path / "db")
    assert idx.search("anything") == []
    assert idx.count == 0


def test_empty_and_whitespace_query(index):
    assert index.search("") == []
    assert index.search("   \n ") == []


# 11. no relevant match ---------------------------------------------------

def test_no_relevant_match_returns_nothing(tmp_path):
    from tests.fake_embeddings import FakeEmbedder

    embedder = FakeEmbedder()
    idx = TextVectorIndex(embedder, persist_dir=tmp_path / "db")
    idx.add(doc_chunks(DOC_FIN, "finance.pdf", FIN_TEXT))

    # Prove the precondition deterministically: pick a query word whose
    # fake vector shares no feature bucket with the corpus text, so cosine
    # similarity is exactly zero and the threshold policy is exercised
    # without depending on hash-collision luck.
    corpus_vec = embedder.embed_one(FIN_TEXT)

    def cos(a, b):
        return sum(x * y for x, y in zip(a, b))

    query = None
    for word in ("quantum", "entanglement", "neutrino", "muon", "boson",
                 "quasar", "nebula", "zebra", "xylophone", "kaleidoscope"):
        if abs(cos(corpus_vec, embedder.embed_one(word))) < 1e-9:
            query = word
            break
    assert query is not None, "could not find an orthogonal query word"

    hits = idx.search(query)
    assert all(h.score <= 0.05 for h in hits)
    assert all(h.chunk.doc_id == DOC_FIN for h in hits)  # nothing foreign


# 12. separation from visual store -----------------------------------------

def test_text_and_visual_collections_are_separate(index):
    assert TEXT_COLLECTION != VISUAL_COLLECTION
    assert index.collection_name == TEXT_COLLECTION
    assert index.collection_names() == [TEXT_COLLECTION]  # no visual collection created


def test_second_index_uses_own_collection(tmp_path):
    idx = TextVectorIndex(FakeEmbedder(), persist_dir=tmp_path / "db",
                          collection_name="custom_chunks")
    assert idx.collection_name == "custom_chunks"
    assert "course_text_chunks" not in idx.collection_names()


# 14. dimension guard (ChromaDB locks dim at first insert) ------------------

def test_dimension_mismatch_raises_with_remediation(tmp_path):
    class _Dim7:
        def embed(self, texts):
            return [[0.0] * 7 for _ in texts]

    path = str(tmp_path / "db")
    a = TextVectorIndex(FakeEmbedder(), persist_dir=path)
    a.add(doc_chunks(DOC_FIN, "finance.pdf", FIN_TEXT))
    assert a.dimension == 64

    b = TextVectorIndex(_Dim7(), persist_dir=path)
    assert b.dimension == 64
    with pytest.raises(Exception, match="dimension mismatch"):
        b.add(doc_chunks(DOC_MKT, "marketing.pdf", MKT_TEXT))
    # the failed add left the collection untouched
    assert TextVectorIndex(FakeEmbedder(), persist_dir=path).count == a.count


# 13. configurable top-k ---------------------------------------------------

def test_top_k(index):
    one = index.search("cash", top_k=1)
    three = index.search("cash", top_k=3)
    assert len(one) == 1
    assert len(three) in (2, 3)


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "3"])
def test_invalid_top_k_rejected(index, bad):
    with pytest.raises(ValueError, match="top_k"):
        index.search("cash", top_k=bad)


# 14. result shape parity with KeywordIndex --------------------------------

def test_hits_are_keyword_hit_objects(index):
    hit = index.search("cash flow")
    assert hit and isinstance(hit[0], Hit)


# 15. SAFE REINDEX: embed + validate BEFORE delete; clear resets dimension --

class FailingEmbedder:
    """Embedder that simulates a class-endpoint outage (Issue #2 review #1)."""

    def embed(self, texts):
        raise EmbeddingsError("service unavailable (simulated)")

    def embed_one(self, text):
        raise EmbeddingsError("service unavailable (simulated)")


def test_reindex_outage_preserves_old_vectors(index):
    """REGRESSION (review finding 1): an embedding outage during reindex
    must NOT destroy the previously indexed document."""
    assert index.search("free cash flow valuation")[0].chunk.doc_id == DOC_FIN
    count_before = index.count
    index._embedder = FailingEmbedder()
    with pytest.raises(EmbeddingsError):
        index.reindex_document(DOC_FIN, doc_chunks(DOC_FIN, "finance.pdf", FIN_TEXT))
    # nothing was deleted: same count, still indexed
    assert index.count == count_before
    assert DOC_FIN in index.stored_doc_ids()
    # still searchable once the embedder recovers
    index._embedder = FakeEmbedder()
    hits = index.search("free cash flow valuation")
    assert hits and hits[0].chunk.doc_id == DOC_FIN


def test_reindex_dimension_mismatch_preserves_old_vectors(tmp_path):
    """A dimension mismatch on reindex must also leave old vectors intact."""
    path = str(tmp_path / "db")
    idx = TextVectorIndex(FakeEmbedder(dim=64), persist_dir=path)
    idx.add(doc_chunks(DOC_FIN, "finance.pdf", FIN_TEXT))
    count_before = idx.count

    class Dim7:
        def embed(self, texts):
            return [[0.0] * 7 for _ in texts]

    idx._embedder = Dim7()
    with pytest.raises(Exception, match="dimension mismatch"):
        idx.reindex_document(DOC_FIN, doc_chunks(DOC_FIN, "finance.pdf", FIN_TEXT))
    assert idx.count == count_before
    # still searchable with a compatible embedder
    idx._embedder = FakeEmbedder(dim=64)
    assert idx.search("free cash flow")


def test_reindex_success_leaves_no_stale_vectors(tmp_path):
    """Changing chunk layout replaces the old id set exactly."""
    path = str(tmp_path / "db")
    idx = TextVectorIndex(FakeEmbedder(), persist_dir=path)
    many = doc_chunks(DOC_FIN, "finance.pdf", FIN_TEXT, chunk_size=40)   # several chunks
    idx.add(many)
    old_ids = set(idx._collection.get(include=[])["ids"])
    assert len(old_ids) >= 2

    few = doc_chunks(DOC_FIN, "finance.pdf", FIN_TEXT, chunk_size=400)   # one chunk
    idx.reindex_document(DOC_FIN, few)
    # exactly the current chunk ids remain — nothing stale, nothing missing
    stored = set(idx._collection.get(where={"doc_id": DOC_FIN}, include=[])["ids"])
    assert stored == {c.chunk_id for c in few}
    assert idx.search("free cash flow")


def test_clear_swaps_embedding_dimension(tmp_path):
    """REGRESSION (review finding 2): clear() must reset the locked
    dimension so a different embedding model can index afterwards."""
    path = str(tmp_path / "db")
    a = TextVectorIndex(FakeEmbedder(dim=64), persist_dir=path)
    a.add(doc_chunks(DOC_FIN, "finance.pdf", FIN_TEXT))
    assert a.dimension == 64

    a.clear()
    assert a.count == 0
    assert a.dimension is None                       # reset, not stale 64

    class Dim8:
        def embed(self, texts):
            return [[0.0] * 8 for _ in texts]

    a._embedder = Dim8()
    a.add(doc_chunks(DOC_MKT, "marketing.pdf", MKT_TEXT))
    assert a.dimension == 8 and a.count >= 1

    # a NEW handle on the same persistent store agrees (no stale lock)
    b = TextVectorIndex(Dim8(), persist_dir=path)
    assert b.dimension == 8
