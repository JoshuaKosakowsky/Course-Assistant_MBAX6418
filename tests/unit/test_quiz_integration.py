"""Contract probes against the shape of PR #15, without importing its branch."""
from dataclasses import dataclass

import pytest

from app.quiz_integration import evidence_from_pack

pytestmark = pytest.mark.unit


@dataclass
class Item:
    filename: str = "Week 5"
    page_no: int = 9
    excerpt: str | None = "BM25 matches exact terms"
    image_path: str | None = "artifacts/week5/page-009.png"
    image_verified: bool = True
    doc_id: str = "doc-five"


@dataclass
class Pack:
    items: list


def test_pack_converts_text_and_verified_image():
    ref, = evidence_from_pack(Pack([Item()]))
    assert (ref.document, ref.page, ref.excerpt, ref.image_path, ref.doc_id) == (
        "Week 5", 9, "BM25 matches exact terms", "artifacts/week5/page-009.png", "doc-five")


def test_doc_id_is_preserved_and_missing_doc_id_normalizes_to_none():
    ref, = evidence_from_pack(Pack([Item()]))
    assert ref.doc_id == "doc-five"
    legacy, = evidence_from_pack(Pack([Item(doc_id="  ")]))
    assert legacy.doc_id is None
    assert legacy.document == "Week 5"


def test_unverified_image_is_never_exposed_as_quiz_evidence():
    ref, = evidence_from_pack(Pack([Item(image_verified=False)]))
    assert ref.excerpt == "BM25 matches exact terms"
    assert ref.image_path is None


def test_visual_only_requires_verified_image():
    ref, = evidence_from_pack(Pack([Item(excerpt=None)]))
    assert ref.excerpt is None and ref.image_path == "artifacts/week5/page-009.png"
    assert evidence_from_pack(Pack([Item(excerpt=None, image_verified=False)])) == ()


def test_missing_identity_and_forged_types_are_rejected():
    assert evidence_from_pack(Pack([Item(filename="")])) == ()
    assert evidence_from_pack(Pack([Item(page_no=True)])) == ()
    assert evidence_from_pack(Pack([Item(excerpt=None, image_path=None)])) == ()


def test_adapter_does_not_mutate_pack():
    item = Item(image_verified=False)
    pack = Pack([item])
    evidence_from_pack(pack)
    assert item.image_path == "artifacts/week5/page-009.png"


def test_adapter_fails_closed_without_items():
    with pytest.raises(ValueError, match="items"):
        evidence_from_pack(object())
