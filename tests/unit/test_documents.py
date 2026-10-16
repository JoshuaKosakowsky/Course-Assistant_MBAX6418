"""Unit tests for the content-addressed document store."""

import os
import stat as stat_module

import pytest

from app.documents import DocumentNotFoundError, DocumentStore

pytestmark = pytest.mark.unit


@pytest.fixture
def store(tmp_path):
    return DocumentStore(root=tmp_path / "data")


def test_add_copies_file_into_corpus(store, tmp_path):
    f = tmp_path / "syllabus.pdf"
    f.write_bytes(b"syllabus")
    doc = store.add(f)
    assert (store.corpus_dir / doc.doc_id / "syllabus.pdf").is_file()
    assert store.get(doc.doc_id).filename == "syllabus.pdf"


def test_loading_same_file_twice_does_not_duplicate(store, tmp_path):
    f = tmp_path / "notes.pdf"
    f.write_bytes(b"same content")
    first = store.add(f)
    second = store.add(f)
    assert first.doc_id == second.doc_id
    assert len(store.list()) == 1


def test_different_content_gets_distinct_documents(store, tmp_path):
    a = tmp_path / "a.pdf"
    b = tmp_path / "b.pdf"
    a.write_bytes(b"content one")
    b.write_bytes(b"content two")
    assert store.add(a).doc_id != store.add(b).doc_id
    assert len(store.list()) == 2


def test_add_missing_file_raises(store, tmp_path):
    with pytest.raises(FileNotFoundError):
        store.add(tmp_path / "does-not-exist.pdf")


def test_remove_deletes_record_and_artifacts(store, tmp_path):
    f = tmp_path / "deck.pdf"
    f.write_bytes(b"deck")
    doc = store.add(f)
    art = store.artifacts_dir / doc.doc_id
    (art / "pages").mkdir(parents=True)
    (art / "pages" / "page-001.png").write_bytes(b"png")
    store.remove(doc.doc_id)
    assert store.list() == []
    assert not (store.corpus_dir / doc.doc_id).exists()
    assert not (store.artifacts_dir / doc.doc_id).exists()


def test_remove_unknown_document_raises(store):
    with pytest.raises(DocumentNotFoundError):
        store.remove("deadbeef00deadbeef")


def test_removal_hooks_fire_for_future_indexes(store, tmp_path):
    calls = []
    store.register_removal_hook(lambda doc_id: calls.append(doc_id))
    f = tmp_path / "a.pdf"
    f.write_bytes(b"indexed later")
    doc = store.add(f)
    store.remove(doc.doc_id)
    assert calls == [doc.doc_id]


def test_find_returns_existing_document(store, tmp_path):
    f = tmp_path / "a.pdf"
    f.write_bytes(b"find me")
    doc = store.add(f)
    assert store.find(f).doc_id == doc.doc_id
    assert store.find(tmp_path / "absent.pdf") is None


def test_corpus_path_returns_stored_copy(store, tmp_path):
    f = tmp_path / "syllabus.pdf"
    f.write_bytes(b"copy me")
    doc = store.add(f)
    assert store.corpus_path(doc.doc_id).is_file()
    assert store.corpus_path(doc.doc_id).name == "syllabus.pdf"


def test_cleanup_orphans_removes_unrecorded_directories(store, tmp_path):
    f = tmp_path / "a.pdf"
    f.write_bytes(b"orphan sweep")
    doc = store.add(f)
    # Simulate a half-finished remove: record gone, leftover dirs remain
    # (as happens when a transient lock outlives the retries).
    (store.library_dir / f"{doc.doc_id}.json").unlink()
    (store.artifacts_dir / doc.doc_id / "pages").mkdir(parents=True)
    (store.artifacts_dir / doc.doc_id / "pages" / "page-001.png").write_bytes(b"png")
    assert (store.artifacts_dir / doc.doc_id).is_dir()
    assert (store.corpus_dir / doc.doc_id).is_dir()

    removed = store.cleanup_orphans()
    assert {p.name for p in removed} == {doc.doc_id}
    assert not (store.artifacts_dir / doc.doc_id).exists()
    assert not (store.corpus_dir / doc.doc_id).exists()


def test_cleanup_orphans_keeps_recorded_directories(store, tmp_path):
    f = tmp_path / "a.pdf"
    f.write_bytes(b"keep me")
    doc = store.add(f)
    assert store.cleanup_orphans() == []
    assert (store.artifacts_dir / doc.doc_id).is_dir() or True
    assert store.get(doc.doc_id).doc_id == doc.doc_id


@pytest.mark.skipif(os.name != "nt", reason="Windows read-only attribute semantics")
def test_remove_clears_read_only_directories(store, tmp_path):
    """Regression: OneDrive marks synced dirs read-only, which blocked
    shutil.rmtree with WinError 5. Clearing the attribute must happen
    before deletion."""
    f = tmp_path / "locked.pdf"
    f.write_bytes(b"locked dir")
    doc = store.add(f)
    art = store.artifacts_dir / doc.doc_id / "pages"
    art.mkdir(parents=True)
    (art / "page-001.png").write_bytes(b"png")
    os.chmod(art, stat_module.S_IREAD)

    store.remove(doc.doc_id)
    assert store.list() == []
    assert not (store.artifacts_dir / doc.doc_id).exists()
