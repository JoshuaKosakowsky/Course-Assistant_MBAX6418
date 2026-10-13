"""Unit tests for the content-addressed document store."""

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
