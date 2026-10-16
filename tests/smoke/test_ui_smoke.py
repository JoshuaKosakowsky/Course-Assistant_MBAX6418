"""Smoke tests for the Gradio management UI (Task 1).

Builds the Blocks app and exercises the add/remove handlers against a
temporary store. Offline — no class endpoints, no real course files.
"""

import pytest

import app.parse as parse
from app.documents import DocumentStore
from app.ui import add_file, build_ui, remove_selected

pytestmark = pytest.mark.smoke


@pytest.fixture
def store(tmp_path):
    return DocumentStore(root=tmp_path / "data")


def test_ui_builds(store):
    demo = build_ui(store)
    assert demo is not None


def test_add_parses_and_lists(store, tiny_pdf):
    df, _, status = add_file(store, str(tiny_pdf))
    assert len(df) == 1
    assert df.iloc[0]["filename"] == tiny_pdf.name
    assert df.iloc[0]["pages"] == 2
    assert "parsed 2 page(s)" in status


def test_readd_same_file_does_not_duplicate(store, tiny_pdf):
    add_file(store, str(tiny_pdf))
    df, _, status = add_file(store, str(tiny_pdf))
    assert len(df) == 1
    assert "already in the library" in status


def test_remove_deletes_and_refreshes(store, tiny_pdf):
    add_file(store, str(tiny_pdf))
    doc = store.list()[0]
    df, _, status = remove_selected(store, doc.doc_id)
    assert len(df) == 0
    assert "Removed" in status
    assert store.list() == []


def test_remove_without_selection_is_helpful(store):
    df, _, status = remove_selected(store, None)
    assert "Select a document" in status


def test_pptx_without_libreoffice_rolls_back_and_warns(store, tmp_path, monkeypatch):
    monkeypatch.setattr(parse, "find_soffice", lambda: None)
    fake_pptx = tmp_path / "deck.pptx"
    fake_pptx.write_bytes(b"not a real pptx")
    df, _, status = add_file(store, str(fake_pptx))
    assert len(df) == 0
    assert store.list() == []  # rolled back so a retry works after installing LibreOffice
    assert "LibreOffice" in status
