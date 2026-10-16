"""Smoke test: the Task 1 management flow end to end, offline.

add (dedup) -> parse -> list/get -> remove (cascade). Uses a generated
2-page PDF fixture and a temp data root; no network, no class endpoints.
"""

import pytest

from app.documents import DocumentStore
from app.parse import parse_file

pytestmark = pytest.mark.smoke


def test_manage_document_lifecycle(tmp_path, tiny_pdf):
    store = DocumentStore(root=tmp_path / "data")

    doc = store.add(tiny_pdf)
    assert len(store.list()) == 1
    assert store.add(tiny_pdf).doc_id == doc.doc_id  # no duplicate

    res = parse_file(tiny_pdf, doc.doc_id, store.artifacts_dir / doc.doc_id)
    assert len(res.pages) == 2
    assert any(page.text for page in res.pages)
    assert (store.artifacts_dir / doc.doc_id / "pages" / "page-001.png").is_file()

    store.remove(doc.doc_id)
    assert store.list() == []
    assert not (store.artifacts_dir / doc.doc_id).exists()
    assert not (store.corpus_dir / doc.doc_id).exists()
