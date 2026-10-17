"""Smoke tests for the Gradio management UI (Task 1).

Builds the Blocks app and exercises the add/remove handlers against a
temporary store. Offline — no class endpoints, no real course files.
"""

from threading import Event, Thread
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


def test_quiz_material_refresh_waits_for_document_mutations(store):
    demo = build_ui(store)
    dependencies = demo.config["dependencies"]

    for action in ("add_file", "remove_selected"):
        matches = [
            (index, event)
            for index, event in enumerate(dependencies)
            if event.get("api_name") == action
        ]
        assert len(matches) == 1
        event_index, _ = matches[0]

        refreshes = [
            event
            for event in dependencies
            if event.get("trigger_after") == event_index
        ]

        assert len(refreshes) == 2

        # The quiz selector must be updated after the mutation.
        quiz_selector = next(
            component["id"]
            for component in demo.config["components"]
            if component.get("props", {}).get("label") == "Course materials"
        )
        assert any(refresh["outputs"] == [quiz_selector] for refresh in refreshes)
        question_selector = next(
            component["id"] for component in demo.config["components"]
            if component.get("props", {}).get("label") == "Materials for your question"
        )
        assert any(refresh["outputs"] == [question_selector] for refresh in refreshes)
        assert all(refresh["trigger_only_on_success"] is False for refresh in refreshes)


def test_delayed_document_mutations_refresh_choices_after_completion(
    store, tiny_pdf, monkeypatch
):
    import app.ui as ui

    original_add = ui.add_file
    original_remove = ui.remove_selected

    for action in ("add", "remove"):
        if action == "remove":
            original_add(store, str(tiny_pdf))
            doc_id = store.list()[0].doc_id

        started = Event()
        release = Event()
        results = []

        def delayed_operation():
            started.set()
            if not release.wait(timeout=5):
                raise TimeoutError("Delayed operation was not released")
            if action == "add":
                return original_add(store, str(tiny_pdf))
            return original_remove(store, doc_id)

        def worker():
            results.append(delayed_operation())
            results.append(ui._choices(store))

        thread = Thread(target=worker)
        thread.start()

        try:
            assert started.wait(timeout=5)
            assert results == []

            release.set()
            thread.join(timeout=10)

            assert not thread.is_alive()
            assert len(results) == 2

            choices = results[1]
            if action == "add":
                assert len(choices) == 1
            else:
                assert choices == []
        finally:
            release.set()
            thread.join(timeout=10)
