"""Unit tests for embedder behavior — incl. the two endpoint-alignment fixes
requested in review: same-model text-to-image bridge, and 512px downscaling
of embedding copies for large page renders."""

from __future__ import annotations

import base64
import io

import pytest
import requests
from PIL import Image

import app.visual_embeddings as em
from app.visual_embeddings import EMBED_IMAGE_MAX_EDGE, ClassEmbeddingClient, resize_for_embedding

pytestmark = pytest.mark.unit


def test_resize_for_embedding_scales_down_long_edge():
    big = Image.new("RGB", (1200, 300), (10, 10, 10))
    resized = resize_for_embedding(big)
    assert resized.size == (EMBED_IMAGE_MAX_EDGE, 128)
    # aspect ratio preserved
    assert resized.size[1] == round(300 * (512 / 1200))


def test_resize_for_embedding_keeps_small_images_untouched():
    small = Image.new("RGB", (300, 200), (10, 10, 10))
    assert resize_for_embedding(small) is small
    assert resize_for_embedding(small).size == (300, 200)


def test_resize_for_embedding_never_mutates_original():
    original = Image.new("RGB", (1000, 1000), (5, 5, 5))
    resize_for_embedding(original)
    assert original.size == (1000, 1000)


def test_embed_image_sends_resized_copy(monkeypatch):
    captured = {}

    def fake_embed_image(self, image_uri, model, base=None):
        captured["image_uri"] = image_uri
        captured["model"] = model
        return [0.0]

    monkeypatch.setattr(em, "VISUAL_EMBED_MODEL", "qwen3-vl")
    monkeypatch.setattr(ClassEmbeddingClient, "_embed_image", fake_embed_image)

    big = Image.new("RGB", (2048, 1024), (200, 30, 30))
    ClassEmbeddingClient(base_url="http://class.test", api_key="k").embed_image(big)

    assert captured["model"] == "qwen3-vl"
    raw = base64.b64decode(captured["image_uri"].split(",", 1)[1])
    decoded = Image.open(io.BytesIO(raw))
    assert decoded.size == (512, 256)


def test_embed_image_sends_multimodal_messages_payload(monkeypatch):
    """The class visual endpoint embeds an image only via the documented
    multimodal ``messages`` shape; a plain ``input`` string is tokenized as
    text and exceeds the context window (HTTP 400 in live probes)."""
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"embedding": [0.0]}]}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return FakeResponse()

    monkeypatch.setattr(em, "VISUAL_EMBED_MODEL", "qwen3-vl")
    monkeypatch.setattr(requests, "post", fake_post)

    client = ClassEmbeddingClient(base_url="http://class.test/v1", api_key="k")
    client.embed_image(Image.new("RGB", (64, 64), (10, 20, 30)))

    assert captured["url"] == "http://class.test/v1/embeddings"
    body = captured["kwargs"]["json"]
    assert body["model"] == "qwen3-vl"
    assert "input" not in body
    content = body["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    assert "data:image/jpeg;base64," in content[0]["image_url"]["url"]
    # The image is downscaled for the embedding request; original remains.
    big = Image.new("RGB", (2048, 1024), (200, 30, 30))
    captured.clear()
    client.embed_image(big)
    body = captured["kwargs"]["json"]
    uri = body["messages"][0]["content"][0]["image_url"]["url"]
    raw = base64.b64decode(uri.split(",", 1)[1])
    decoded = Image.open(io.BytesIO(raw))
    assert decoded.size == (512, 256)


def test_embed_text_visual_uses_visual_model_and_endpoint(monkeypatch):
    captured = {}

    def fake_embed(self, inputs, model, base=None):
        captured["model"] = model
        captured["base"] = base
        return [0.0]

    monkeypatch.setattr(em, "VISUAL_EMBED_MODEL", "qwen3-vl")
    monkeypatch.setattr(ClassEmbeddingClient, "_embed", fake_embed)

    client = ClassEmbeddingClient(base_url="http://9003.test", api_key="k")
    client.embed_text_visual("tokenization diagram")

    assert captured["model"] == "qwen3-vl"
    assert captured["base"] is None  # falls back to the visual base URL


def test_embed_text_uses_text_space_endpoint(monkeypatch):
    captured = {}

    def fake_embed(self, inputs, model, base=None):
        captured["model"] = model
        captured["base"] = base
        return [0.0]

    monkeypatch.setattr(em, "TEXT_EMBED_MODEL", "nemotron")
    monkeypatch.setattr(em, "TEXT_EMBED_BASE_URL", "http://9002.test")
    monkeypatch.setattr(ClassEmbeddingClient, "_embed", fake_embed)

    ClassEmbeddingClient(base_url="http://9003.test", api_key="k").embed_text("x")

    assert captured["model"] == "nemotron"
    assert captured["base"] == "http://9002.test"


def test_embed_image_requires_model_configured(monkeypatch):
    monkeypatch.setattr(em, "VISUAL_EMBED_MODEL", "")
    client = ClassEmbeddingClient(base_url="http://class.test", api_key="k")
    with pytest.raises(RuntimeError, match="VISUAL_EMBED_MODEL"):
        client.embed_image(Image.new("RGB", (8, 8)))


def test_visual_client_rejects_reserved_port_9000():
    """Assignment rule: port 9000 is reserved for Hermes agentic work."""
    with pytest.raises(ValueError, match="reserved"):
        ClassEmbeddingClient(base_url="http://dobolyi.com:9000/v1", api_key="k")
