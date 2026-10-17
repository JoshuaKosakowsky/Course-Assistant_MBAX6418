"""E2E tests: real class visual-embedding endpoint (issue #3, Phase 2).

Run deliberately with ``pytest -m e2e``. Auto-skip when .env lacks the
visual-embedding settings so the default suite stays offline. These tests
document the live endpoint contract: images must be sent as multimodal
``messages`` with an ``image_url`` content part. A plain data-URI string
under ``input`` is tokenized as text and exceeds the model context window
(verified HTTP 400 against the live service), so the client sends the
multimodal shape and these tests use a realistic page-sized synthetic
image rather than a tiny fixture that could pass either encoding.
"""

import pytest
from PIL import Image

import app.config as config
from app.visual_embeddings import ClassEmbeddingClient
from app.visual_index import VisualIndex

pytestmark = pytest.mark.e2e


def _endpoint_configured() -> bool:
    return config.visual_embed_endpoint_configured()


@pytest.fixture(autouse=True)
def _require_endpoint():
    if not _endpoint_configured():
        pytest.skip("visual-embedding endpoint not configured in .env")


def _page_like_image() -> Image.Image:
    """A high-entropy 512x512 page render, not a tiny or flat fixture.

    The old data-URI ``input`` encoding tokenizes the base64 as *text*;
    a solid-color image compresses so well that its URI stays under the
    context window and can slip through either encoding. Deterministic
    noise keeps the JPEG large enough to exceed the window (~8k tokens)
    under the text path while remaining a valid image for the multimodal
    messages path.
    """
    import random

    rng = random.Random(6418)
    data = [tuple(rng.randrange(256) for _ in range(3)) for _ in range(512 * 512)]
    image = Image.new("RGB", (512, 512))
    image.putdata(data)
    return image


def test_class_client_embeds_page_like_image():
    client = ClassEmbeddingClient()
    vector = client.embed_image(_page_like_image())
    assert len(vector) > 8
    assert all(isinstance(v, float) for v in vector)
    assert any(v for v in vector)  # not all zeros


def test_real_embedding_index_and_query_roundtrip(tmp_path):
    index = VisualIndex(root=tmp_path / "idx")  # real class embedder
    image_path = tmp_path / "slide.png"
    _page_like_image().save(image_path)

    vid = index.add_image(doc_id="e2e-doc", page=1, image_path=image_path)
    assert index.count() == 1

    hits = index.query_image(image_path, k=3)
    assert hits
    assert hits[0].id == vid
    assert hits[0].page == 1
    assert hits[0].distance < 1.0  # same image should rank first and close
