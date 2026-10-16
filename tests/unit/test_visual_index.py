"""Unit tests for the visual index — synthetic embeddings, fully offline."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from app.embeddings import fake_embedder
from app.visual_index import COLLECTION_NAME, VisualIndex

pytestmark = pytest.mark.unit

W, H = 48, 48


def make_image(path: Path, color: tuple[int, int, int]) -> Path:
    """Deterministic tiny test image."""
    img = Image.new("RGB", (W, H), color)
    ImageDraw.Draw(img).rectangle((4, 4, 20, 20), fill=(255, 255, 255))
    img.save(path)
    return path


@pytest.fixture
def index_factory(tmp_path):
    def make(root: Path | None = None, **kwargs):
        return VisualIndex(root=root or tmp_path / "idx", embedder=fake_embedder(), **kwargs)

    return make


def test_fake_embedder_is_deterministic_and_distinct(tmp_path):
    p1 = make_image(tmp_path / "a.png", (200, 30, 30))
    p2 = make_image(tmp_path / "b.png", (30, 200, 30))
    embed = fake_embedder()
    v1a, v1b, v2 = embed(Image.open(p1)), embed(Image.open(p1)), embed(Image.open(p2))
    assert v1a == v1b
    assert v1a != v2
    assert abs(sum(v * v for v in v1a) - 1.0) < 1e-6


def test_add_and_query_roundtrip(index_factory, tmp_path):
    idx = index_factory()
    images = [make_image(tmp_path / f"p{i}.png", c) for i, c in [(1, (200, 30, 30)), (2, (30, 200, 30)), (3, (30, 30, 200))]]
    for i, path in enumerate(images):
        idx.add_image("docA", i + 1, path)
    assert idx.count() == 3

    hits = idx.query_image(images[1])
    assert hits
    top = hits[0]
    assert top.doc_id == "docA"
    assert top.page == 2
    assert Path(top.image_path).name == "p2.png"
    assert top.distance < 1e-6  # identical image bytes -> identical vector


def test_reindex_same_image_is_idempotent(index_factory, tmp_path):
    idx = index_factory()
    path = make_image(tmp_path / "p1.png", (10, 10, 10))
    first = idx.add_image("docA", 1, path)
    second = idx.add_image("docA", 1, path)
    assert first == second
    assert idx.count() == 1


def test_collection_is_visual_separate_from_text(index_factory):
    idx = index_factory()
    assert idx._collection.name == COLLECTION_NAME
    # the persistent store is a dedicated directory, not shared with a text index


def test_remove_doc_purges_only_that_document(index_factory, tmp_path):
    idx = index_factory()
    for doc in ("A", "B"):
        for page in (1, 2):
            idx.add_image(doc, page, make_image(tmp_path / f"{doc}{page}.png", (50, 60, 70)))
    assert idx.count() == 4
    assert idx.remove_doc("A") == 2
    assert idx.count() == 2
    hits = idx.query_image(make_image(tmp_path / "query.png", (50, 60, 70)))
    assert all(h.doc_id == "B" for h in hits)
    assert idx.remove_doc("absent") == 0


def test_query_text_bridge_with_injected_text_embedder(index_factory, tmp_path):
    vec_note = [1.0, 0.0, 0.0, 0.0]
    vec_gift = [0.0, 1.0, 0.0, 0.0]

    def text_embed(text: str) -> list[float]:
        return vec_note if "retrieval" in text.lower() else vec_gift

    idx = index_factory(text_embedder=text_embed)
    idx.add_image("docA", 1, make_image(tmp_path / "p1.png", (10, 10, 10)), embedding=vec_note)
    idx.add_image("docA", 2, make_image(tmp_path / "p2.png", (20, 20, 20)), embedding=vec_gift)

    hits = idx.query_text("slides about retrieval")
    assert hits and hits[0].page == 1
    hits2 = idx.query_text("gift card design")
    assert hits2 and hits2[0].page == 2


def test_query_text_defaults_to_visual_space_embedder(monkeypatch, tmp_path):
    """Regression: the text-to-image bridge must embed text with the SAME
    visual model/endpoint as images (embed_text_visual), never the separate
    port-9002 text-embedding space."""

    import app.visual_index as vi_mod

    calls = {}

    class FakeClient:
        def embed_text_visual(self, text):
            calls["method"] = "embed_text_visual"
            return [0.1, 0.2]

        def embed_text(self, text):
            calls["method"] = "embed_text"  # must never be reached
            return [0.9, 0.9]

    monkeypatch.setattr(vi_mod, "ClassEmbeddingClient", lambda: FakeClient())

    idx = VisualIndex(root=tmp_path / "idx")  # no embedder injected: real default path
    img = make_image(tmp_path / "p1.png", (10, 10, 10))
    idx.add_image("docA", 1, img, embedding=[0.1, 0.2])

    hits = idx.query_text("some diagram topic")
    assert calls.get("method") == "embed_text_visual"
    assert hits and hits[0].doc_id == "docA"


def test_query_empty_index_returns_empty(index_factory):
    idx = index_factory()
    assert idx.query_embedding([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) == []


def test_index_directory_logs_counts(caplog, index_factory, tmp_path):
    img_dir = tmp_path / "slides"
    img_dir.mkdir()
    make_image(img_dir / "slide_1.png", (100, 20, 20))
    make_image(img_dir / "slide_2.png", (20, 100, 20))
    idx = index_factory()
    with caplog.at_level(logging.INFO, logger="app.visual_index"):
        added = idx.index_images_in_directory(img_dir, doc_id="lecture1")
    assert added == 2
    assert "Indexed 2/2 images for document lecture1" in caplog.text
    assert idx.count() == 2


def test_cli_fails_cleanly_without_endpoint_config(monkeypatch, tmp_path, capsys):
    """Negative case: offline runs must fail loudly, not hang or crash."""
    import app.env as env_mod
    import app.index_visual as cli

    monkeypatch.setattr(env_mod, "VISUAL_EMBED_BASE_URL", "")
    monkeypatch.setattr(env_mod, "VISUAL_EMBED_API_KEY", "")
    monkeypatch.setattr(env_mod, "VISUAL_EMBED_MODEL", "")
    monkeypatch.setattr(env_mod, "DATA_ROOT", tmp_path)

    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    make_image(img_dir / "s1.png", (1, 2, 3))

    code = cli.main([str(img_dir)])
    out = capsys.readouterr()
    assert code == 1
    assert "not configured" in out.err
