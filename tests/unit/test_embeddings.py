"""Unit tests for embedder behavior — incl. the two endpoint-alignment fixes
requested in review: same-model text-to-image bridge, and 512px downscaling
of embedding copies for large page renders."""

from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

import app.embeddings as em
from app.embeddings import EMBED_IMAGE_MAX_EDGE, ClassEmbeddingClient, resize_for_embedding

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

    def fake_embed(self, inputs, model, base=None):
        captured["input"] = inputs[0]
        return [0.0]

    monkeypatch.setattr(em, "VISUAL_EMBED_MODEL", "qwen3-vl")
    monkeypatch.setattr(ClassEmbeddingClient, "_embed", fake_embed)

    big = Image.new("RGB", (2048, 1024), (200, 30, 30))
    ClassEmbeddingClient(base_url="http://class.test", api_key="k").embed_image(big)

    raw = base64.b64decode(captured["input"].split(",", 1)[1])
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
