"""Smoke tests for the REAL application startup path (``python -m app.ui``):
``build_ui(store, indexer)`` wiring, upload -> index, removal -> purge,
restart recovery, keyword-only fallback, and no-redundant-embedding
guarantees.

Offline and deterministic (FakeEmbedder plus counting/broken variants).
The production entry point never uses a fake embedder — ``_production_indexer``
builds the real class client or degrades to keyword-only mode.
"""

import pymupdf
import pytest

import app.ingest as ingest
import app.ui as ui
from app.documents import DocumentStore
from app.embeddings import EmbeddingsError
from app.indexing import CorpusIndexer
from app.parse import ConversionError
from tests.fake_embeddings import FakeEmbedder

pytestmark = pytest.mark.smoke

FIN = [
    "Free cash flow equals operating cash flow minus capital expenditures.",
    "Discounted cash flow valuation estimates enterprise value.",
]


def make_pdf(tmp_path, name, pages):
    path = tmp_path / name
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()
    return path


class CountingEmbedder:
    """FakeEmbedder wrapper that records how many embed() calls happen."""

    def __init__(self):
        self._inner = FakeEmbedder()
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return self._inner.embed(texts)

    def embed_one(self, text):
        return self._inner.embed_one(text)


class BrokenEmbedder:
    """Embedder whose service is down (used for outage simulation)."""

    def embed(self, texts):
        raise EmbeddingsError("service unavailable (simulated)")

    def embed_one(self, text):
        raise EmbeddingsError("service unavailable (simulated)")


@pytest.fixture
def app_env(tmp_path):
    store = DocumentStore(root=tmp_path / "data")
    indexer = CorpusIndexer(store, embedder=FakeEmbedder(),
                            persist_dir=tmp_path / "data" / "indexes")
    ui.build_ui(store, indexer)  # the real startup call: hooks + reconcile
    return store, indexer, tmp_path


# 1. startup initializes the indexer -------------------------------------

def test_app_startup_initializes_indexer(app_env):
    store, indexer, _ = app_env
    assert indexer._wired is True  # noqa: SLF001 - test hooks registration
    assert len(store._removal_hooks) == 1  # noqa: SLF001
    # bound methods compare by (self, func), not identity
    assert store._removal_hooks[0] == indexer.remove_document  # noqa: SLF001


# 2. a successful upload creates searchable text --------------------------

def test_upload_creates_searchable_text(app_env):
    store, indexer, tmp = app_env
    _df, _choices, note = ui.add_file(store, str(make_pdf(tmp, "a.pdf", FIN)), indexer)
    assert "indexed 2 chunk(s)" in note
    assert indexer.keyword.search("free cash flow")[0].chunk.filename == "a.pdf"
    assert indexer.vector.search("free cash flow")[0].chunk.filename == "a.pdf"


# 3. a failed parse never reports successful indexing ----------------------

def test_failed_parse_does_not_report_indexing(app_env):
    store, indexer, tmp = app_env
    bad = tmp / "corrupt.pdf"
    bad.write_bytes(b"this is not a pdf at all")
    _df, _choices, note = ui.add_file(store, str(bad), indexer)
    assert "parsing failed" in note
    assert "indexed" not in note
    assert indexer.keyword.size == 0
    assert indexer.vector.count == 0


def test_libreoffice_missing_does_not_report_indexing(app_env):
    """ConversionError path: message warns about the missing converter but
    must not trigger indexing (regression for the old startswith('Added')
    check)."""
    store, indexer, tmp = app_env
    # Monkey-patch the parse_file name app.ingest imported at module load
    # (the seam add_and_parse() calls) to raise the conversion error a
    # PPTX upload hits without LibreOffice.
    real = ingest.parse_file

    def fake_convert(source, doc_id, out_dir):
        raise ConversionError("soffice not found")

    ingest.parse_file = fake_convert
    try:
        _df, _choices, note = ui.add_file(store, str(make_pdf(tmp, "b.pdf", FIN)), indexer)
    finally:
        ingest.parse_file = real
    assert "soffice not found" in note
    assert "indexed" not in note
    assert indexer.keyword.size == 0


# 4. removing a document purges both indexes (app handler path) ------------

def test_remove_purges_both_indexes(app_env):
    store, indexer, tmp = app_env
    _df, _choices, note = ui.add_file(store, str(make_pdf(tmp, "a.pdf", FIN)), indexer)
    assert indexer.keyword.search("free cash flow")
    doc_id = store.list()[0].doc_id
    _df, _choices, msg = ui.remove_selected(store, doc_id)
    assert "Removed" in msg
    assert indexer.keyword.search("free cash flow") == []
    assert indexer.vector.search("free cash flow") == []


# 5. restart restores keyword search --------------------------------------

def test_restart_restores_keyword_search(tmp_path):
    store = DocumentStore(root=tmp_path / "data")
    i1 = CorpusIndexer(store, embedder=FakeEmbedder(),
                       persist_dir=tmp_path / "data" / "indexes")
    ui.add_file(store, str(make_pdf(tmp_path, "a.pdf", FIN)), i1)

    # simulated restart: fresh indexer, same store + chroma dir
    i2 = CorpusIndexer(store, embedder=FakeEmbedder(),
                       persist_dir=tmp_path / "data" / "indexes")
    ui.build_ui(store, i2)
    hits = i2.keyword.search("free cash flow")
    assert hits and hits[0].chunk.filename == "a.pdf"


# 6. ChromaDB data persists across restart ---------------------------------

def test_chroma_persists_across_restart(tmp_path):
    store = DocumentStore(root=tmp_path / "data")
    i1 = CorpusIndexer(store, embedder=FakeEmbedder(),
                       persist_dir=tmp_path / "data" / "indexes")
    ui.add_file(store, str(make_pdf(tmp_path, "a.pdf", FIN)), i1)
    count1 = i1.vector.count

    i2 = CorpusIndexer(store, embedder=FakeEmbedder(),
                       persist_dir=tmp_path / "data" / "indexes")
    ui.build_ui(store, i2)
    assert i2.vector.count == count1
    assert i2.vector.search("free cash flow")[0].chunk.filename == "a.pdf"


# 7. startup does not unnecessarily re-embed --------------------------------

def test_startup_does_not_reembed(tmp_path):
    counter = CountingEmbedder()
    store = DocumentStore(root=tmp_path / "data")
    i1 = CorpusIndexer(store, embedder=counter,
                       persist_dir=tmp_path / "data" / "indexes")
    ui.add_file(store, str(make_pdf(tmp_path, "a.pdf", FIN)), i1)
    calls_after_upload = counter.calls
    assert calls_after_upload == 1  # one batch for the document's chunks

    # restart with the same counting embedder: reconcile must embed nothing
    i2 = CorpusIndexer(store, embedder=counter,
                       persist_dir=tmp_path / "data" / "indexes")
    ui.build_ui(store, i2)
    assert counter.calls == calls_after_upload  # no re-embedding after restart


# 8. missing vectors are repaired (only the missing ones) -------------------

def test_missing_vectors_repaired_on_startup(tmp_path):
    store = DocumentStore(root=tmp_path / "data")
    i1 = CorpusIndexer(store, embedder=BrokenEmbedder(),
                       persist_dir=tmp_path / "data" / "indexes")
    _df, _choices, note = ui.add_file(store, str(make_pdf(tmp_path, "a.pdf", FIN)), i1)
    assert "vectors skipped" in note
    assert i1.vector.count == 0

    counter = CountingEmbedder()
    i2 = CorpusIndexer(store, embedder=counter,
                       persist_dir=tmp_path / "data" / "indexes")
    ui.build_ui(store, i2)  # repair embeds ONLY the missing document
    assert counter.calls == 1, "repair must not re-embed documents that have vectors"
    assert i2.vector.count >= 1
    assert i2.vector.search("free cash flow")[0].chunk.filename == "a.pdf"


# 9. keyword-only mode when credentials are missing --------------------------

def test_keyword_only_mode_without_credentials(tmp_path):
    store = DocumentStore(root=tmp_path / "data")
    indexer = CorpusIndexer(store, embedder=None,  # exactly what production does
                            persist_dir=tmp_path / "data" / "indexes")
    ui.build_ui(store, indexer)
    _df, _choices, note = ui.add_file(store, str(make_pdf(tmp_path, "a.pdf", FIN)), indexer)
    assert "indexed 2 chunk(s)" in note
    assert "keyword-only mode" in note
    assert indexer.vector.count == 0
    assert indexer.keyword.search("free cash flow")[0].chunk.filename == "a.pdf"
    assert indexer.vector.search("free cash flow") == []  # guarded, no crash
    # startup reconcile with embedder=None must also not raise
    i2 = CorpusIndexer(store, embedder=None,
                       persist_dir=tmp_path / "data" / "indexes")
    ui.build_ui(store, i2)
    assert i2.keyword.search("free cash flow")  # still searchable after restart
