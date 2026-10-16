"""Unit tests for app.embeddings.TextEmbeddingClient (offline).

The HTTP layer is a fake session; no network, no class endpoints. Tests
cover request shape, response validation, error handling, the reserved
port guard, and key handling.
"""

import json

import pytest

from app.embeddings import EmbeddingsError, TextEmbeddingClient
from app import config

pytestmark = pytest.mark.unit

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


# ---- happy path ---------------------------------------------------------

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


# ---- validation ---------------------------------------------------------

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


# ---- security -----------------------------------------------------------

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
