"""E2E tests: real class visual-embedding endpoint (issue #3, Phase 2).

Run deliberately with ``pytest -m e2e``. Auto-skip when .env lacks the
visual-embedding settings so the default suite stays offline. These tests
also document the live endpoint contract — the exact image input encoding
the class service accepts — and will fail loudly if it differs from our
data-URI assumption, which is exactly the point of having them.
"""

import pytest
from PIL import Image

import app.env as env
from app.embeddings import ClassEmbeddingClient
from app.visual_index import VisualIndex

pytestmark = pytest.mark.e2e


def _endpoint_configured() -> bool:
    return bool(
        env.VISUAL_EMBED_BASE_URL and env.VISUAL_EMBED_API_KEY and env.VISUAL_EMBED_MODEL
    )


@pytest.fixture(autouse=True)
def _require_endpoint():
    if not _endpoint_configured():
        pytest.skip("visual-embedding endpoint not configured in .env")


def _tiny_image() -> Image.Image:
    return Image.new("RGB", (64, 64), (10, 120, 220))


def test_class_client_embeds_image():
    client = ClassEmbeddingClient()
    vector = client.embed_image(_tiny_image())
    assert len(vector) > 8
    assert all(isinstance(v, float) for v in vector)
    assert any(v for v in vector)  # not all zeros


def test_real_embedding_index_and_query_roundtrip(tmp_path):
    index = VisualIndex(root=tmp_path / "idx")  # real class embedder
    image_path = tmp_path / "slide.png"
    _tiny_image().save(image_path)

    vid = index.add_image(doc_id="e2e-doc", page=1, image_path=image_path)
    assert index.count() == 1

    hits = index.query_image(image_path, k=3)
    assert hits
    assert hits[0].id == vid
    assert hits[0].page == 1
    assert hits[0].distance < 1.0  # same image should rank first and close
