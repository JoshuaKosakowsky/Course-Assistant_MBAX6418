"""Smoke tests for the visual index — offline, on the committed fixture.

Wires the pieces together with a synthetic embedder: index the committed
slide fixture, query it by image, and confirm the index reports what it
holds. No network, no keys.
"""

import pytest
from pathlib import Path
from PIL import Image

from app.visual_index import VisualIndex
from app.embeddings import fake_embedder

FIXTURE = "tests/fixtures/slide_sample.png"


@pytest.mark.smoke
def test_index_fixture_and_query_by_image(tmp_path):
    index = VisualIndex(root=tmp_path / "idx", embedder=fake_embedder())
    slide = FIXTURE
    added = index.add_image(doc_id="fixture-doc", page=1, image_path=slide)
    assert added
    assert index.count() == 1

    hits = index.query_image(Image.open(slide), k=3)
    assert len(hits) == 1
    hit = hits[0]
    assert hit.doc_id == "fixture-doc"
    assert hit.page == 1
    # Compare normalized Path values: Chroma returns OS-native separators
    # (Windows backslashes) while the fixture path was written with
    # forward slashes.
    assert Path(hit.image_path) == Path(slide)
    assert hit.distance < 1e-6


@pytest.mark.smoke
def test_index_directory_and_query_roundtrip(tmp_path):
    slides = tmp_path / "slides"
    slides.mkdir()
    index = VisualIndex(root=tmp_path / "idx", embedder=fake_embedder())
    from PIL import ImageDraw

    img = Image.new("RGB", (64, 64), (15, 90, 160))
    ImageDraw.Draw(img).rectangle((8, 8, 40, 24), fill=(255, 220, 120))
    slides.joinpath("slide_1.png").write_bytes(_png_bytes(img))
    slides.joinpath("slide_2.png").write_bytes(_png_bytes(Image.new("RGB", (64, 64), (160, 90, 15))))

    added = index.index_images_in_directory(slides, doc_id="two-slides")
    assert added == 2
    assert index.count() == 2

    hits = index.query_image(slides / "slide_1.png", k=2)
    assert hits[0].page == 1


def _png_bytes(img: Image.Image) -> bytes:
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
