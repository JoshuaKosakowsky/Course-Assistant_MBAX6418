"""Unit tests for app.rerank (Issue #4 / Phase 3, Step 2).

Offline and deterministic: a fake transport stands in for the class
reranker (mocked responses/errors), and images are tiny pymupdf-rendered
PNGs. Live endpoint verification lives in tests/e2e (opt-in). Covers the
client/validation layer, candidate encoding, and orchestration/fallback.
"""

from pathlib import Path

import pytest
import requests

import pymupdf

from app.chunking import Chunk
from app.fusion import FusedHit, RankedEvidence
from app.rerank import (
    RerankClient,
    RerankError,
    encode_document,
    rerank_fused,
)

pytestmark = pytest.mark.unit

API = "http://rerank.test:9004/v1"


# ---- helpers ---------------------------------------------------------------


def mk_chunk(chunk_id, doc_id="doc-a", page_no=1, text="evidence text",
             filename="a.pdf", image_path=""):
    return Chunk(
        chunk_id=chunk_id, doc_id=doc_id, filename=filename, format=".pdf",
        page_no=page_no, index_in_page=0, text=text, image_path=image_path,
    )


def mk_fused(*chunks):
    return [
        FusedHit(
            chunk=c,
            fused_rank=i + 1,
            fused_score=(10.0 - i) / 100.0,
            methods=("keyword", "vector"),
            evidence=(RankedEvidence("keyword", i + 1, 1.0),
                      RankedEvidence("vector", i + 1, 0.5)),
        )
        for i, c in enumerate(chunks)
    ]


class FakeResp:
    def __init__(self, status=200, body=None, text=""):
        self.status_code = status
        self._body = body
        self.text = text

    def json(self):
        return self._body


def results_for(scores: dict) -> dict:
    """A valid reranker body where ``scores`` maps input index -> score."""
    entries = [
        {"index": i, "document": {"text": None, "multi_modal": None},
         "relevance_score": s}
        for i, s in sorted(scores.items(), key=lambda kv: -kv[1])
    ]
    return {"id": "x", "model": "m", "results": entries, "usage": {}}


class FakeTransport:
    """Stand-in for requests.Session: records calls, returns canned results."""

    def __init__(self, body_fn=None, error=None, status=200, body=None):
        self._body_fn = body_fn
        self._error = error
        self._status = status
        self._body = body
        self.calls = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json,
                           "timeout": timeout})
        if self._error is not None:
            raise self._error
        if self._status != 200:
            return FakeResp(self._status, {"detail": "boom"})
        if self._body_fn is not None:
            return FakeResp(200, self._body_fn(json))
        return FakeResp(200, self._body)


def make_client(transport=None, api_key="test-key", **kwargs):
    return RerankClient(base_url=API, model="test-model", api_key=api_key,
                        session=transport or FakeTransport(
                            body_fn=lambda req: results_for({})),
                        timeout=5, **kwargs)


def make_png(path, text="page content"):
    d = pymupdf.open()
    pg = d.new_page()
    pg.insert_text((72, 72), text)
    pg.get_pixmap(dpi=72).save(str(path))
    d.close()
    return path


# 1. valid requests -----------------------------------------------------------

def test_valid_request_shape():
    transport = FakeTransport(body_fn=lambda req: results_for({0: 0.9, 1: 0.4}))
    client = make_client(transport)
    scores = client.rank("q", ["d0", "d1"])
    assert len(transport.calls) == 1
    call = transport.calls[0]
    assert call["url"] == "http://rerank.test:9004/v1/rerank"
    assert call["headers"]["Authorization"].startswith("Bearer ")
    assert call["json"]["model"] == "test-model"
    assert call["json"]["query"] == "q"
    assert call["json"]["documents"] == ["d0", "d1"]
    assert call["timeout"] == 5
    assert scores == {0: 0.9, 1: 0.4}


def test_route_normalization():
    """Base URLs ending in /v1 (class convention) must not double it."""
    transport_v1 = FakeTransport(body_fn=lambda req: results_for({0: 0.9}))
    client_v1 = RerankClient(base_url="http://x.test:9004/v1", model="m",
                             api_key="k", session=transport_v1)
    client_v1.rank("q", ["d"])
    assert transport_v1.calls[0]["url"] == "http://x.test:9004/v1/rerank"

    transport_root = FakeTransport(body_fn=lambda req: results_for({0: 0.9}))
    client_root = RerankClient(base_url="http://x.test:9004", model="m",
                               api_key="k", session=transport_root)
    client_root.rank("q", ["d"])
    assert transport_root.calls[0]["url"] == "http://x.test:9004/v1/rerank"


def test_api_key_never_leaks_into_request_body():
    transport = FakeTransport(body_fn=lambda req: results_for({0: 0.9}))
    client = make_client(transport, api_key="super-secret-key")
    client.rank("q", ["d"])
    body = str(transport.calls[0]["json"])
    assert "super-secret-key" not in body


# 2. response parsing ----------------------------------------------------------

def test_response_parsing_scores():
    client = make_client(FakeTransport(
        body_fn=lambda req: results_for({0: 0.7, 2: 0.3, 1: 0.1})))
    scores = client.rank("q", ["a", "b", "c"])
    assert scores == {0: 0.7, 2: 0.3, 1: 0.1}


# 3. reranking changes order ---------------------------------------------------

def test_rerank_reorders_candidates():
    fused = mk_fused(
        mk_chunk("a", doc_id="doc-a"), mk_chunk("b", doc_id="doc-b"),
        mk_chunk("c", doc_id="doc-c"),
    )
    # fused order a,b,c; reranker says c > a > b
    transport = FakeTransport(body_fn=lambda req: results_for({2: 0.95, 0: 0.6, 1: 0.2}))
    outcome = rerank_fused("q", fused, make_client(transport))
    assert [c.original.chunk.chunk_id for c in outcome.ranked] == ["c", "a", "b"]
    assert [c.original.fused_rank for c in outcome.ranked] == [3, 1, 2]  # preserved
    assert len(transport.calls) == 1  # exactly one request


# 4. original fusion ranks/scores remain available -----------------------------

def test_original_fusion_fully_preserved():
    fused = mk_fused(mk_chunk("a"), mk_chunk("b"))
    transport = FakeTransport(body_fn=lambda req: results_for({0: 0.9, 1: 0.8}))
    outcome = rerank_fused("q", fused, make_client(transport))
    assert outcome.inputs == fused
    for c in outcome.ranked:
        orig = c.original
        assert orig.fused_rank >= 1 and orig.fused_score > 0
        assert orig.methods == ("keyword", "vector")
        assert len(orig.evidence) == 2


# 5. document/page metadata -----------------------------------------------------

def test_metadata_preserved():
    fused = mk_fused(mk_chunk("a", doc_id="doc-x", page_no=7, text="cash flow",
                              filename="f.pdf", image_path="p/7.png"))
    transport = FakeTransport(body_fn=lambda req: results_for({0: 0.9}))
    outcome = rerank_fused("q", fused, make_client(transport))
    c = outcome.ranked[0].original.chunk
    assert (c.doc_id, c.page_no, c.filename, c.text, c.image_path) == (
        "doc-x", 7, "f.pdf", "cash flow", "p/7.png")


# 6. image-to-candidate association ---------------------------------------------

def test_image_encoded_when_valid(tmp_path):
    img = tmp_path / "page-001.png"
    make_png(img, "slide about cash flow")
    fused = mk_fused(mk_chunk("a", page_no=1, image_path=str(img)))
    transport = FakeTransport(body_fn=lambda req: results_for({0: 0.9}))
    outcome = rerank_fused("q", fused, make_client(transport))
    payload = transport.calls[0]["json"]["documents"][0]
    assert isinstance(payload, dict) and payload["type"] == "content"
    parts = payload["content"]
    types = [p["type"] for p in parts]
    assert types == ["text", "image_url"]
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert outcome.images_included is True
    assert outcome.ranked[0].image_included is True


def test_encode_document_rejects_wrong_page_image(tmp_path):
    img = tmp_path / "page-005.png"  # wrong page number for this chunk
    make_png(img)
    payload, included, note = encode_document(
        mk_fused(mk_chunk("a", page_no=1, image_path=str(img)))[0])
    assert payload == "evidence text"
    assert included is False
    assert "page-005.png" in note and "expected" in note


# 7. missing images -------------------------------------------------------------

def test_missing_image_becomes_text_only():
    fused = mk_fused(mk_chunk("a", image_path=""))
    transport = FakeTransport(body_fn=lambda req: results_for({0: 0.9}))
    outcome = rerank_fused("q", fused, make_client(transport))
    payload = transport.calls[0]["json"]["documents"][0]
    assert payload == fused[0].chunk.text
    assert outcome.ranked[0].image_included is False
    assert "no image path" in outcome.ranked[0].note


# 8. corrupt images ---------------------------------------------------------------

def test_corrupt_image_excluded():
    import tempfile

    bad = Path(tempfile.mkdtemp()) / "page-001.png"
    bad.write_bytes(b"\x00\x01\x02 not a real png at all")
    payload, included, note = encode_document(
        mk_fused(mk_chunk("a", page_no=1, image_path=str(bad)))[0])
    assert included is False
    assert "corrupt" in note or "unreadable" in note


# 9. text-only candidates ----------------------------------------------------------

def test_text_only_mode():
    fused = mk_fused(mk_chunk("a"), mk_chunk("b"))
    transport = FakeTransport(body_fn=lambda req: results_for({0: 0.9, 1: 0.1}))
    outcome = rerank_fused("q", fused, make_client(transport),
                           include_images=False)
    for doc in transport.calls[0]["json"]["documents"]:
        assert isinstance(doc, str)
    assert outcome.text_only is True
    assert outcome.images_included is False


# 10. image size and count limits --------------------------------------------------

def test_oversize_image_excluded(tmp_path):
    img = tmp_path / "page-001.png"
    make_png(img)
    fused = mk_fused(mk_chunk("a", page_no=1, image_path=str(img)))
    payload, included, note = encode_document(fused[0], max_image_kb=0.001)
    assert included is False
    assert "too large" in note


def test_max_images_cap(tmp_path):
    img1 = tmp_path / "d1" ; img1.mkdir()
    img2 = tmp_path / "d2" ; img2.mkdir()
    p1 = make_png(img1 / "page-001.png", "one")
    p2 = make_png(img2 / "page-001.png", "two")
    fused = mk_fused(mk_chunk("a", doc_id="doc-1", page_no=1, image_path=str(p1)),
                     mk_chunk("b", doc_id="doc-2", page_no=1, image_path=str(p2)))
    transport = FakeTransport(body_fn=lambda req: results_for({0: 0.9, 1: 0.8}))
    outcome = rerank_fused("q", fused, make_client(transport), max_images=1)
    docs = transport.calls[0]["json"]["documents"]
    assert isinstance(docs[0], dict) and isinstance(docs[1], str)  # cap applied
    assert outcome.ranked[0].image_included is True
    assert outcome.ranked[1].image_included is False


# 11. invalid score values ----------------------------------------------------------

def test_non_finite_score_rejected():
    body = {"results": [{"index": 0, "relevance_score": float("nan")}]}
    client = make_client(FakeTransport(body=body))
    with pytest.raises(RerankError, match="non-finite"):
        client.rank("q", ["d"])


def test_non_numeric_score_rejected():
    body = {"results": [{"index": 0, "relevance_score": "high"}]}
    client = make_client(FakeTransport(body=body))
    with pytest.raises(RerankError, match="relevance_score"):
        client.rank("q", ["d"])


# 12. duplicate or invalid response indexes -------------------------------------------

def test_duplicate_index_rejected():
    body = {"results": [{"index": 0, "relevance_score": 0.9},
                        {"index": 0, "relevance_score": 0.5}]}
    client = make_client(FakeTransport(body=body))
    with pytest.raises(RerankError, match="duplicate"):
        client.rank("q", ["d", "e"])


def test_out_of_range_index_rejected():
    # one of two docs referenced by index 2: structural error beats the
    # count checks, so the specific diagnostic surfaces first
    body = {"results": [{"index": 2, "relevance_score": 0.9},
                        {"index": 1, "relevance_score": 0.8}]}
    client = make_client(FakeTransport(body=body))
    with pytest.raises(RerankError, match="out of range"):
        client.rank("q", ["d", "e"])


# 13. incomplete responses -----------------------------------------------------------

def test_missing_results_rejected():
    client = make_client(FakeTransport(body={"id": "x"}))
    with pytest.raises(RerankError, match="results"):
        client.rank("q", ["d"])


def test_result_count_mismatch_rejected():
    body = {"results": [{"index": 0, "relevance_score": 0.9}]}
    client = make_client(FakeTransport(body=body))  # 3 docs, 1 result, no top_n
    with pytest.raises(RerankError, match="incomplete"):
        client.rank("q", ["a", "b", "c"])


def test_too_many_results_rejected():
    # server returns 3 results although top_n=2 was requested
    body = {"results": [{"index": 0, "relevance_score": 0.1},
                        {"index": 1, "relevance_score": 0.2},
                        {"index": 2, "relevance_score": 0.3}]}
    client = make_client(FakeTransport(body=body))
    with pytest.raises(RerankError, match="more than the requested"):
        client.rank("q", ["d1", "d2", "d3"], top_n=2)


# 14. timeouts --------------------------------------------------------------------------

def test_timeout_raises_and_falls_back():
    fused = mk_fused(mk_chunk("a"), mk_chunk("b"))
    client = make_client(FakeTransport(error=requests.exceptions.Timeout()))
    with pytest.raises(RerankError, match="timed out"):
        client.rank("q", ["d"])
    outcome = rerank_fused("q", fused, client)
    assert outcome.applied is False and outcome.fallback is True
    assert "timed out" in outcome.degraded[0]


# 15. authentication/configuration failures ----------------------------------------------

def test_missing_api_key_rejected():
    with pytest.raises(RerankError, match="not configured"):
        RerankClient(base_url=API, model="m", api_key="")


def test_config_defaults_reject_dummy_key():
    """Skip-guard mirror: a dummy/empty key must never silently pass."""
    with pytest.raises(RerankError, match="not configured"):
        RerankClient(base_url=API, model="m", api_key="")


def test_non_200_raises():
    client = make_client(FakeTransport(status=500, body={"detail": "boom"}))
    with pytest.raises(RerankError, match="500"):
        client.rank("q", ["d"])


# 16. fallback keeps the original fused ordering ------------------------------------------

def test_fallback_keeps_fused_order():
    fused = mk_fused(mk_chunk("z"), mk_chunk("a"), mk_chunk("m"))
    client = make_client(FakeTransport(error=RuntimeError("connection refused")))
    outcome = rerank_fused("q", fused, client)
    assert outcome.applied is False
    assert [c.original.chunk.chunk_id for c in outcome.ranked] == ["z", "a", "m"]
    assert all(c.rerank_score is None for c in outcome.ranked)
    assert all("rerank failed" in c.note for c in outcome.ranked)
    # transport detail is deliberately NOT leaked (may embed credentials);
    # only the safe generic category is reported (review finding 3)
    assert outcome.degraded and "connection/transport error" in outcome.degraded[0]


# 17. deterministic ranking ties -----------------------------------------------------------

def test_deterministic_ties():
    fused = mk_fused(mk_chunk("b"), mk_chunk("a"))
    transport = FakeTransport(body_fn=lambda req: results_for({0: 0.5, 1: 0.5}))
    o1 = rerank_fused("q", fused, make_client(transport))
    o2 = rerank_fused("q", fused, make_client(transport))
    id_order = [c.original.chunk.chunk_id for c in o1.ranked]
    assert id_order == [c.original.chunk.chunk_id for c in o2.ranked]
    # equal scores: original fused_rank (b=1) wins over chunk_id
    assert id_order == ["b", "a"]


# 18. empty candidate lists ---------------------------------------------------------------

def test_empty_candidates():
    outcome = rerank_fused("q", [], make_client())
    assert outcome.applied is False
    assert outcome.ranked == []
    assert outcome.degraded == ["no candidates"]


# 19. configurable top-k -------------------------------------------------------------------

def test_top_k():
    fused = mk_fused(*(mk_chunk(c) for c in "abcde"))
    transport = FakeTransport(body_fn=lambda req: results_for(
        {i: s for i, s in enumerate([0.9, 0.8, 0.7, 0.6, 0.5])}))
    outcome = rerank_fused("q", fused, make_client(transport), top_k=3)
    assert len(outcome.ranked) == 3
    assert len(outcome.inputs) == 5  # inputs never truncated


# 20. no accidental use of port 9000 -----------------------------------------------------------

def test_port_9000_rejected():
    with pytest.raises(ValueError, match="9000"):
        RerankClient(base_url="http://dobolyi.com:9000/v1", model="m",
                     api_key="k")


# 21. ownership validation happens BEFORE the reranker request ---------------------------

def test_wrong_document_image_never_reaches_payload(tmp_path):
    """REGRESSION (Issue #4 Step 4): a matching page-001.png filename from
    the WRONG document must be excluded BEFORE the request is encoded —
    the actual payload must be text-only for that candidate."""
    # candidate chunk claims doc-a page 1 but its image lives in doc-b's dir
    foreign = tmp_path / "doc-b" / "pages" / "page-001.png"
    foreign.parent.mkdir(parents=True)
    make_png(foreign, "other document's page 1")
    chunk = mk_chunk("c0", doc_id="doc-a", page_no=1,
                     image_path=str(foreign))
    fused = mk_fused(chunk, mk_chunk("c1", doc_id="doc-a", page_no=1))

    def strict_validator(c):
        # manifest-authoritative equivalent: doc-b image cannot belong to doc-a
        if "doc-b" in c.image_path:
            return None, "wrong-document image rejected"
        return c.image_path, ""

    transport = FakeTransport(body_fn=lambda req: results_for({0: 0.9, 1: 0.8}))
    outcome = rerank_fused("q", fused, make_client(transport),
                           include_images=True, validate_image=strict_validator)
    docs = transport.calls[0]["json"]["documents"]
    assert isinstance(docs[0], str), "wrong-document image must not be sent"
    assert "image_url" not in str(docs[0])
    assert "wrong-document" in outcome.ranked[0].note
    assert outcome.ranked[0].image_included is False
    # the OWned candidate (c1, no image path) also stays text-only but harmless
    assert isinstance(docs[1], str)


# 14b. credential safety (review finding 3): transport detail must never be
# interpolated, chained, or echo an API key --------------------------------

def test_transport_error_never_leaks_credentials():
    # NOTE: the fake token deliberately avoids the repo's secret-shape
    # hygiene patterns; it is NOT a real-looking key
    token = "fake-leak-token-abc-12345"
    client = make_client(api_key=token, transport=FakeTransport(error=RuntimeError(
        "connection reset while sending Authorization: Bearer " + token)))
    with pytest.raises(RerankError) as ei:
        client.rank("q", ["d"])
    msg = str(ei.value)
    assert token not in msg and "Bearer" not in msg
    assert "connection/transport" in msg          # safe generic category
    assert ei.value.__cause__ is None             # no chained traceback leak


def test_malformed_json_error_redacts_and_never_chains():
    token = "fake-echo-999-xyz"
    class EchoBadBody:
        def post(self, *a, **k):
            return FakeResp(200, '{"echo": "' + token + '" !not-json')

    client = make_client(api_key=token, transport=EchoBadBody())
    with pytest.raises(RerankError) as ei:
        client.rank("q", ["d"])
    assert token not in str(ei.value)
    assert ei.value.__cause__ is None
