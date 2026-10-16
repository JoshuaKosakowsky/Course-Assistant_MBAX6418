"""Smoke-tier integration tests: DocumentStore -> parse -> chunk ->
BM25 + ChromaDB (Issue #2 / Phase 2 completion).

Offline and deterministic (FakeEmbedder). Uses generated multi-page PDF
fixtures; no real course materials, no network, no class endpoints.
"""

import subprocess
import sys
from pathlib import Path

import pytest

import pymupdf

from app.documents import DocumentStore
from app.embeddings import EmbeddingsError
from app.indexing import CorpusIndexer, IndexStats
from tests.fake_embeddings import FakeEmbedder

pytestmark = pytest.mark.smoke


def make_pdf(tmp_path: Path, name: str, pages: list[str]) -> Path:
    """Generate a multi-page PDF with the given per-page text."""
    path = tmp_path / name
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()
    return path


FIN_PAGES = [
    "Free cash flow equals operating cash flow minus capital expenditures.",
    "Discounted cash flow valuation estimates enterprise value.",
]
MKT_PAGES = [
    "Brand positioning defines how customers perceive a product.",
    "Customer segmentation groups buyers by behavior and needs.",
]
OPS_PAGES = [
    "Inventory management balances holding costs against stockouts.",
    "Supply chain logistics optimizes delivery and warehouse flow.",
]


@pytest.fixture
def library(tmp_path):
    """Store + indexer wired the way the real app wires them."""
    store = DocumentStore(root=tmp_path / "data")
    indexer = CorpusIndexer(
        store, embedder=FakeEmbedder(), persist_dir=tmp_path / "data" / "indexes"
    )
    indexer.register_hooks()
    return store, indexer


def add_pdf(store, indexer, tmp_path, name, pages):
    path = make_pdf(tmp_path, name, pages)
    doc = store.add(path)
    stats = indexer.index_document(doc.doc_id)
    assert stats.errors == [], stats.errors
    return doc


# 1. multiple-document indexing ------------------------------------------

def test_multiple_document_indexing(library, tmp_path):
    store, indexer = library
    add_pdf(store, indexer, tmp_path, "finance.pdf", FIN_PAGES)
    add_pdf(store, indexer, tmp_path, "marketing.pdf", MKT_PAGES)
    add_pdf(store, indexer, tmp_path, "operations.pdf", OPS_PAGES)
    assert indexer.status()["library_docs"] == 3
    assert indexer.keyword.size >= 2
    assert indexer.vector.count >= 2
    assert indexer.keyword.search("free cash flow")[0].chunk.filename == "finance.pdf"
    assert indexer.vector.search("free cash flow")[0].chunk.filename == "finance.pdf"


# 2. multi-page source metadata ------------------------------------------

def test_multi_page_metadata(library, tmp_path):
    store, indexer = library
    doc = add_pdf(store, indexer, tmp_path, "two.pdf",
                  ["PAGE-ONE-MARKER alpha beta gamma", "PAGE-TWO-MARKER delta epsilon"])
    kw = indexer.keyword.search("PAGE-TWO-MARKER")
    assert kw and kw[0].chunk.page_no == 2 and kw[0].chunk.doc_id == doc.doc_id
    vec = indexer.vector.search("PAGE-TWO-MARKER")
    assert vec and vec[0].chunk.page_no == 2 and vec[0].chunk.doc_id == doc.doc_id
    kw1 = indexer.keyword.search("PAGE-ONE-MARKER")
    assert kw1 and kw1[0].chunk.page_no == 1


# 3. repeated indexing, no duplicates ------------------------------------

def test_repeated_indexing_no_duplicates(library, tmp_path):
    store, indexer = library
    add_pdf(store, indexer, tmp_path, "finance.pdf", FIN_PAGES)
    k1, v1 = indexer.keyword.size, indexer.vector.count
    stats = indexer.index_library()
    assert stats.errors == []
    assert indexer.keyword.size == k1
    assert indexer.vector.count == v1


# 4. removal purges BOTH indexes (the assignment-critical path) -----------

def test_removal_purges_both_indexes(library, tmp_path):
    store, indexer = library
    doc = add_pdf(store, indexer, tmp_path, "finance.pdf", FIN_PAGES)
    assert indexer.keyword.search("free cash flow")
    assert indexer.vector.search("free cash flow")
    k_before = indexer.keyword.size
    store.remove(doc.doc_id)  # app flow: hooks fire here
    assert indexer.keyword.search("free cash flow") == []
    assert indexer.vector.search("free cash flow") == []
    assert indexer.keyword.size < k_before


# 5. re-upload after deletion ---------------------------------------------

def test_reupload_after_deletion(library, tmp_path):
    store, indexer = library
    doc = add_pdf(store, indexer, tmp_path, "finance.pdf", FIN_PAGES)
    store.remove(doc.doc_id)
    assert indexer.keyword.search("free cash flow") == []
    again = store.add(make_pdf(tmp_path, "finance.pdf", FIN_PAGES))
    indexer.index_document(again.doc_id)  # UI/CLI explicit indexing step
    assert indexer.keyword.search("free cash flow")[0].chunk.doc_id == again.doc_id
    assert indexer.vector.search("free cash flow")[0].chunk.doc_id == again.doc_id


# 6. ChromaDB persistence across restarts ---------------------------------

def test_chroma_persistence_across_restart(tmp_path):
    store = DocumentStore(root=tmp_path / "data")
    i1 = CorpusIndexer(store, embedder=FakeEmbedder(), persist_dir=tmp_path / "idx")
    add_pdf(store, i1, tmp_path, "finance.pdf", FIN_PAGES)
    count1 = i1.vector.count

    i2 = CorpusIndexer(store, embedder=FakeEmbedder(), persist_dir=tmp_path / "idx")
    assert i2.vector.count == count1
    assert i2.vector.search("free cash flow")[0].chunk.filename == "finance.pdf"


# 7. BM25 reconstruction after restart ------------------------------------

def test_bm25_reconstruction_after_restart(tmp_path):
    store = DocumentStore(root=tmp_path / "data")
    i1 = CorpusIndexer(store, embedder=FakeEmbedder(), persist_dir=tmp_path / "idx")
    add_pdf(store, i1, tmp_path, "marketing.pdf", MKT_PAGES)

    i2 = CorpusIndexer(store, embedder=FakeEmbedder(), persist_dir=tmp_path / "idx")
    assert i2.keyword.size == 0  # BM25 is in-memory: rebuilt on reconcile
    stats = i2.reconcile()
    assert stats.errors == []
    hits = i2.keyword.search("brand positioning")
    assert hits and hits[0].chunk.filename == "marketing.pdf"


def test_restart_does_not_resurrect_deleted_documents(tmp_path):
    store = DocumentStore(root=tmp_path / "data")
    i1 = CorpusIndexer(store, embedder=FakeEmbedder(), persist_dir=tmp_path / "idx")
    doc = add_pdf(store, i1, tmp_path, "finance.pdf", FIN_PAGES)
    store.remove(doc.doc_id)

    i2 = CorpusIndexer(store, embedder=FakeEmbedder(), persist_dir=tmp_path / "idx")
    i2.reconcile()
    assert i2.keyword.search("free cash flow") == []
    assert i2.vector.search("free cash flow") == []


# 8. unparseable documents -------------------------------------------------

def test_unparseable_document_handled(library, tmp_path):
    store, indexer = library
    bad = tmp_path / "corrupt.pdf"
    bad.write_bytes(b"this is not a pdf at all")
    doc = store.add(bad)
    stats = indexer.index_document(doc.doc_id)
    assert stats.errors, "unparseable document must be reported"
    assert "corrupt.pdf" in store.list()[0].filename  # library intact
    assert indexer.keyword.size == 0
    store.remove(doc.doc_id)  # removal still works
    assert store.list() == []


# 9. embedding service unavailable ------------------------------------------

class BrokenEmbedder:
    def embed(self, texts):
        raise EmbeddingsError("service unavailable (simulated)")

    def embed_one(self, text):
        raise EmbeddingsError("service unavailable (simulated)")


def test_embedding_service_unavailable(library, tmp_path):
    store, _ = library
    broken = CorpusIndexer(store, embedder=BrokenEmbedder(),
                           persist_dir=tmp_path / "brokendb")
    broken.register_hooks()
    doc = add_pdf(store, broken, tmp_path, "finance.pdf", FIN_PAGES)
    stats = broken.index_document(doc.doc_id)
    assert stats.skipped and "vector" in stats.skipped[0]
    assert broken.keyword.search("free cash flow"), "keyword must still work"
    assert store.list(), "library must not be corrupted"

    # retriable: a healthy embedder indexes the same document afterwards
    healthy = CorpusIndexer(store, embedder=FakeEmbedder(),
                            persist_dir=tmp_path / "brokendb")
    stats2 = healthy.index_document(doc.doc_id)
    assert stats2.errors == [] and stats2.vector_chunks >= 1
    assert healthy.vector.search("free cash flow")


def test_removal_works_while_embedding_down(library, tmp_path):
    store, indexer = library
    doc = add_pdf(store, indexer, tmp_path, "marketing.pdf", MKT_PAGES)
    # embedder still healthy here; simulate service dying afterwards
    indexer.embedder = BrokenEmbedder()
    store.remove(doc.doc_id)
    assert indexer.keyword.search("brand positioning") == []
    assert indexer.vector.search("brand positioning") == []


# 10. reindexing after chunking-settings changes ----------------------------

def test_reindex_after_chunking_settings_change(tmp_path):
    store = DocumentStore(root=tmp_path / "data")
    i1 = CorpusIndexer(store, embedder=FakeEmbedder(),
                       persist_dir=tmp_path / "idx",
                       chunk_size=60, chunk_overlap=5)
    add_pdf(store, i1, tmp_path, "finance.pdf", FIN_PAGES)
    small_count = i1.keyword.size

    i2 = CorpusIndexer(store, embedder=FakeEmbedder(),
                       persist_dir=tmp_path / "idx", chunk_size=400)
    stats = i2.index_library()
    assert stats.errors == []
    assert 0 < i2.keyword.size < small_count  # fewer, bigger chunks
    rows = i2.vector._collection.get(include=["documents"])["documents"]
    assert rows and all(len(d) <= 400 + 20 for d in rows)  # no stale big/small mix


# 11. consistent retrieval from the sample corpus ----------------------------

def test_consistent_queries_across_indexes(library, tmp_path):
    store, indexer = library
    add_pdf(store, indexer, tmp_path, "finance.pdf", FIN_PAGES)
    add_pdf(store, indexer, tmp_path, "marketing.pdf", MKT_PAGES)
    add_pdf(store, indexer, tmp_path, "operations.pdf", OPS_PAGES)
    pairs = [
        ("free cash flow", "finance.pdf"),
        ("brand positioning", "marketing.pdf"),
        ("inventory management", "operations.pdf"),
    ]
    for query, expected in pairs:
        kw = indexer.keyword.search(query)
        vec = indexer.vector.search(query)
        assert kw and kw[0].chunk.filename == expected, (query, "keyword")
        assert vec and vec[0].chunk.filename == expected, (query, "vector")


# 12. accurate indexing counts ------------------------------------------------

def test_accurate_counts(library, tmp_path):
    store, indexer = library
    d1 = add_pdf(store, indexer, tmp_path, "finance.pdf", FIN_PAGES)
    d2 = add_pdf(store, indexer, tmp_path, "marketing.pdf", MKT_PAGES)
    stats = indexer.index_library()
    assert stats.doc_ids == 2
    assert stats.pages == 4  # 2 pages per doc, all non-empty
    assert stats.chunks == indexer.keyword.size
    assert indexer.vector.count == indexer.keyword.size
    assert stats.keyword_chunks == indexer.keyword.size
    assert stats.vector_chunks == indexer.vector.count
    # repeated indexing reports the same totals
    stats2 = indexer.index_library()
    assert stats2.chunks == stats.chunks


# CLI entry point -------------------------------------------------------------

def test_cli_help_exits_zero():
    result = subprocess.run(
        [sys.executable, "-m", "app.indexing", "--help"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0
    assert "Index the document library" in result.stdout


def test_stats_object_merges():
    a = IndexStats(doc_ids=1, pages=2, chunks=3, keyword_chunks=3, vector_chunks=3)
    b = IndexStats(doc_ids=1, pages=2, chunks=3, keyword_chunks=3, vector_chunks=3,
                   skipped=["vector:x"])
    c = a + b
    assert (c.doc_ids, c.pages, c.chunks) == (2, 4, 6)
    assert c.skipped == ["vector:x"]


# 9b. failed reindex through CorpusIndexer preserves old vectors ------------

def test_failed_reindex_preserves_old_vectors(library, tmp_path):
    """REGRESSION (review finding 1, pipeline level): re-indexing a
    document while the embedding service is down must keep its previous
    vectors searchable — the fix lives in TextVectorIndex.reindex_document
    but must be proven through the CorpusIndexer path used by the app."""
    store, indexer = library
    doc = add_pdf(store, indexer, tmp_path, "finance.pdf", FIN_PAGES)
    count_before = indexer.vector.count
    assert indexer.vector.search("free cash flow")

    indexer.vector._embedder = BrokenEmbedder()      # service dies mid-flight
    stats = indexer.index_document(doc.doc_id)       # triggers reindex path
    assert stats.skipped and "vector" in stats.skipped[0]
    assert indexer.vector.count == count_before      # nothing destroyed
    # still searchable once the service recovers
    indexer.vector._embedder = FakeEmbedder()
    assert indexer.vector.search("free cash flow")

    # recovery: a healthy embedder replaces cleanly (no stale mix)
    indexer.vector._embedder = FakeEmbedder()
    stats2 = indexer.index_document(doc.doc_id)
    assert stats2.vector_chunks >= 1


# 10c. --reindex workflow: clearing allows an embedding-model switch --------

def test_reindex_workflow_swaps_embedding_dimension(tmp_path):
    """REGRESSION (review finding 2, pipeline level): the documented
    ``python -m app.indexing --reindex`` model-switch workflow — clear the
    collection, then index with a different embedding dimension — must
    work end to end through the CorpusIndexer."""
    store = DocumentStore(root=tmp_path / "data")
    i64 = CorpusIndexer(store, embedder=FakeEmbedder(dim=64),
                        persist_dir=tmp_path / "idx")
    doc = add_pdf(store, i64, tmp_path, "finance.pdf", FIN_PAGES)
    assert i64.vector.dimension == 64 and i64.vector.count >= 1

    i8 = CorpusIndexer(store, embedder=FakeEmbedder(dim=8),
                       persist_dir=tmp_path / "idx")
    i8.vector.clear()                                     # the --reindex step
    assert i8.vector.dimension is None and i8.vector.count == 0
    stats = i8.index_library()
    assert stats.errors == []
    assert i8.vector.dimension == 8 and i8.vector.count >= 1
    assert i8.vector.search("free cash flow")
