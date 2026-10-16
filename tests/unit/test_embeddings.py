"""Unit tests for app.embeddings — both halves, fully offline.

- ``TextEmbeddingClient`` (Issue #2): fake-session HTTP tests covering
  request shape, response validation, error handling, the reserved-port
  guard, and key handling.
- Visual embedders (Issue #3): deterministic ``fake_embedder`` plus
  ``ClassEmbeddingClient`` behavior — 512px downscaling of embedding
  copies, the same-model text-to-image bridge, and the text-space
  endpoint distinction.
"""

from __future__ import annotations

import base64
import io
import json

import pytest
from PIL import Image

from app import config
import app.embeddings as em
from app.embeddings import (
    EMBED_IMAGE_MAX_EDGE,
    ClassEmbeddingClient,
    EmbeddingsError,
    TextEmbeddingClient,
    resize_for_embedding,
)

pytestmark = pytest.mark.unit


# =========================================================================
# TextEmbeddingClient (Issue #2 / Phase 2)
# =========================================================================

FAKE_KEY = "test-key-abc"
URL = "http://dobolyi.com:9002/v1"
MODEL = "nvidia/Nemotron-3-Embed-1B-BF16"


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=None):
        self.status_code = status_code
        self._payload = payload
        self.text = text if text is not None else json.dumps(payload)

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeSession:
    """Records the last request and returns a scripted response."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.kwargs = None

    def post(self, url, **kwargs):
        self.calls.append(url)
        self.kwargs = kwargs
        return self.responses.pop(0)


def embed_ok(texts):
    payload = {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": [0.1, 0.2, 0.3]}
            for i in range(len(texts))
        ],
        "model": MODEL,
        "usage": {"prompt_tokens": 4, "total_tokens": 4},
    }
    return payload


def make_client(session, key=FAKE_KEY):
    return TextEmbeddingClient(base_url=URL, api_key=key, model=MODEL, session=session)


# ---- happy path ----------------------------------------------------------

def test_embed_success_batch():
    session = FakeSession([FakeResponse(200, embed_ok(["a", "b"]))])
    client = make_client(session)
    vectors = client.embed(["hello world", "second text"])
    assert len(vectors) == 2
    assert all(v == [0.1, 0.2, 0.3] for v in vectors)
    # request shape
    assert session.calls == ["http://dobolyi.com:9002/v1/embeddings"]
    body = session.kwargs["json"]
    assert body["model"] == MODEL
    assert body["input"] == ["hello world", "second text"]
    assert session.kwargs["headers"]["Authorization"].startswith("Bearer ")
    assert FAKE_KEY not in repr(client)


def test_embed_one():
    session = FakeSession([FakeResponse(200, embed_ok(["x"]))])
    vector = make_client(session).embed_one("single")
    assert vector == [0.1, 0.2, 0.3]


def test_empty_input_returns_empty_list():
    session = FakeSession([])
    assert make_client(session).embed([]) == []


# ---- validation ----------------------------------------------------------

def test_whitespace_only_text_rejected():
    session = FakeSession([])
    with pytest.raises(EmbeddingsError, match="whitespace"):
        make_client(session).embed(["   \n\t "])
    assert session.calls == []  # nothing sent


def test_non_200_raises_with_status():
    session = FakeSession([FakeResponse(503, text="service unavailable")])
    with pytest.raises(EmbeddingsError, match="503"):
        make_client(session).embed(["x"])


def test_invalid_json_raises():
    session = FakeSession([FakeResponse(200, text="not json{{")])
    with pytest.raises(EmbeddingsError, match="JSON"):
        make_client(session).embed(["x"])


def test_row_count_mismatch_raises():
    # 2 inputs but 1 data row
    session = FakeSession([FakeResponse(200, embed_ok(["only-one"]))])
    with pytest.raises(EmbeddingsError, match="malformed"):
        make_client(session).embed(["a", "b"])


def test_missing_embedding_raises():
    payload = {"data": [{"object": "embedding", "index": 0}]}  # no embedding
    session = FakeSession([FakeResponse(200, payload)])
    with pytest.raises(EmbeddingsError, match="embedding"):
        make_client(session).embed(["x"])


def test_non_numeric_values_raise():
    payload = {"data": [{"embedding": ["a", 1.0]}]}
    session = FakeSession([FakeResponse(200, payload)])
    with pytest.raises(EmbeddingsError, match="non-numeric"):
        make_client(session).embed(["x"])


def test_inconsistent_dimension_raises():
    payload = {"data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.1]}]}
    session = FakeSession([FakeResponse(200, payload)])
    with pytest.raises(EmbeddingsError, match="inconsistent"):
        make_client(session).embed(["a", "b"])


# ---- security ------------------------------------------------------------

def test_reserved_port_9000_rejected():
    with pytest.raises(ValueError, match="9000"):
        TextEmbeddingClient(
            base_url="http://dobolyi.com:9000/v1", api_key=FAKE_KEY, model=MODEL
        )


def test_config_guard_rejects_9000_in_any_service():
    with pytest.raises(ValueError, match="9000"):
        config.validate_no_reserved_port("http://dobolyi.com:9001/v1", "http://x:9000/v1")


def test_missing_api_key_raises_helpful_error():
    with pytest.raises(EmbeddingsError, match="CLASS_API_KEY"):
        TextEmbeddingClient(base_url=URL, api_key="", model=MODEL)


def test_repr_never_contains_key():
    client = make_client(FakeSession([]))
    assert FAKE_KEY not in repr(client)


# =========================================================================
# Visual index embedders (Issue #3)
# =========================================================================

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


def test_image_to_data_uri_round_trips():
    image = Image.new("RGB", (4, 4), (200, 30, 30))
    uri = em.image_to_data_uri(image)
    assert uri.startswith("data:image/jpeg;base64,")
    raw = base64.b64decode(uri.split(",", 1)[1])
    assert Image.open(io.BytesIO(raw)).size == (4, 4)


def test_fake_embedder_is_deterministic_and_unit_length():
    embed = em.fake_embedder(dims=8)
    a = Image.new("RGB", (32, 32), (1, 2, 3))
    b = Image.new("RGB", (32, 32), (1, 2, 3))
    c = Image.new("RGB", (32, 32), (9, 9, 9))
    va, vb, vc = embed(a), embed(b), embed(c)
    assert va == vb
    assert va != vc
    assert abs(sum(v * v for v in va) - 1.0) < 1e-9
