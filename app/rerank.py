"""Multimodal reranking over fused candidates (Issue #4 / Phase 3, Step 2).

Endpoint contract — VERIFIED live (2026-10-16) against the class service:

    POST {RERANK_BASE_URL}/v1/rerank   (default http://dobolyi.com:9004/v1)
    Authorization: Bearer {CLASS_API_KEY_RERANK}
    body: {"model": ..., "query": <str | multimodal>,
           "documents": [<str | multimodal>, ...], "top_n": <int>}
    - a query/document is a plain string (text) OR a multimodal object:
        {"type": "content", "content": [
            {"type": "text", "text": "..."},
            {"type": "image_url",
             "image_url": {"url": "data:image/png;base64,..."}}]}
    response:
        {"id", "model", "usage",
         "results": [{"index": <int>, "document": {...},
                      "relevance_score": <float>}]}
    results arrive sorted by relevance_score DESCENDING; "index" refers
    to the position in the submitted ``documents`` array.

Separation of concerns: the transport/validation layer (``RerankClient``),
the candidate encoding layer (``encode_document``), and the ranking
orchestration layer (``rerank_fused``) are independent so each can be
tested with mocks. The class service is the ONLY reranker used in
production; port 9000 remains reserved for Hermes agentic operations.
"""

from __future__ import annotations

import base64
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from app import config
from app.fusion import FusedHit

# ---- errors ---------------------------------------------------------------


class RerankError(Exception):
    """Raised for any transport, configuration, or response problem."""


# ---- verified response/request helpers -------------------------------------

IMAGE_MAGIC = {
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"\xff\xd8\xff": "image/jpeg",
}


def _mime_type(path: Path) -> Optional[str]:
    """Detect PNG/JPEG by magic bytes (no extra dependency)."""
    try:
        head = path.read_bytes()[:8]
    except OSError:
        return None
    for magic, mime in IMAGE_MAGIC.items():
        if head.startswith(magic):
            return mime
    return None


def encode_document(
    hit: FusedHit,
    include_image: bool = True,
    max_chars: int = 2000,
    max_image_kb: int = 4000,
    validate_image=None,
) -> Tuple[object, bool, str]:
    """Encode one fused candidate for the reranker.

    Returns ``(payload, image_included, note)`` where ``payload`` is a
    plain text string when no usable image is present, or the verified
    multimodal content object when an image is attached.

    ``validate_image`` (optional) is a callable ``(chunk) -> (path, note)``
    that performs AUTHORITATIVE ownership validation (artifact directory +
    manifest) BEFORE any payload is built — a matching ``page-001.png``
    filename alone never proves the image belongs to the candidate's
    document/page. When it returns ``(None, note)`` the image is excluded
    and the note recorded; when it returns a path, that validated path is
    used. Without the hook, a lightweight existence/decodability check is
    applied (used only for pre-vetted page-level candidates).
    """
    text = hit.chunk.text or ""
    note = ""
    if len(text) > max_chars:
        text = text[:max_chars]
        note = "text truncated"
    if not include_image:
        return text, False, note

    if validate_image is not None:
        vpath, vnote = validate_image(hit.chunk)
        if vpath is None:
            return text, False, note + ("; " if note else "") + vnote
        path = Path(vpath)
    else:
        path = Path(hit.chunk.image_path) if hit.chunk.image_path else None
        if path is None:
            return text, False, note + ("; " if note else "") + "no image path"

    # When an authoritative validator ran, OWNERSHIP is its job and the
    # parser's page-NNN name convention does not apply (page-level
    # candidates may carry vetted images with arbitrary names).
    img_note = _image_note(path, hit, max_image_kb,
                           enforce_page_name=validate_image is None)
    if img_note:
        return text, False, note + ("; " if note else "") + img_note

    mime = _mime_type(path)
    if not mime:
        return text, False, note + ("; " if note else "") + "unreadable/corrupt image"

    data_uri = f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"
    payload = {
        "type": "content",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": data_uri}},
        ],
    }
    return payload, True, note

def _image_note(path: Optional[Path], hit: FusedHit, max_image_kb: int,
                enforce_page_name: bool = True) -> str:
    """Return why an image cannot be used, or '' if it can."""
    if path is None or not path.exists():
        return "image missing"
    if enforce_page_name:
        expected = f"page-{hit.chunk.page_no:03d}.png"
        if path.name != expected:
            # safety: refuse to attach an image that does not correspond
            # to this candidate's document/page
            return f"image name {path.name!r} != expected {expected!r}"
    try:
        size_kb = path.stat().st_size / 1024
    except OSError:
        return "image unreadable"
    if size_kb > max_image_kb:
        return f"image too large ({size_kb:.0f} KB > {max_image_kb} KB)"
    return ""


# ---- transport/validation layer ---------------------------------------------


def _validate_response(body: object, n_documents: int,
                       top_n: Optional[int]) -> Dict[int, float]:
    """Validate a reranker response and map input index -> score.

    Raises RerankError on: non-dict body, missing/malformed results,
    duplicate or out-of-range indexes, non-finite or missing scores,
    results exceeding the submitted candidate count.
    """
    if not isinstance(body, dict):
        raise RerankError(f"reranker response is not a JSON object: {type(body).__name__}")
    results = body.get("results")
    if not isinstance(results, list):
        raise RerankError("reranker response missing 'results' list")

    # per-entry structural validation first (most specific diagnostics)
    scores: Dict[int, float] = {}
    for i, entry in enumerate(results):
        if not isinstance(entry, dict):
            raise RerankError(f"result {i}: entry is not an object")
        index = entry.get("index")
        if not isinstance(index, int) or isinstance(index, bool):
            raise RerankError(f"result {i}: 'index' missing or not an integer: {index!r}")
        if index < 0 or index >= n_documents:
            raise RerankError(f"result {i}: index {index} out of range for {n_documents} documents")
        if index in scores:
            raise RerankError(f"result {i}: duplicate index {index}")
        score = entry.get("relevance_score")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise RerankError(f"result {i}: 'relevance_score' missing or not numeric")
        score = float(score)
        if not math.isfinite(score):
            raise RerankError(f"result {i}: non-finite relevance_score {score!r}")
        scores[index] = score

    # aggregate count checks after the entries themselves are valid
    max_allowed = n_documents if top_n in (None, 0) else min(top_n, n_documents)
    if len(results) > n_documents:
        raise RerankError(
            f"reranker returned {len(results)} results for {n_documents} "
            "documents (more than submitted)"
        )
    if len(results) > max_allowed:
        raise RerankError(
            f"reranker returned {len(results)} results, more than the requested {max_allowed}"
        )
    if top_n in (None, 0) and len(results) != n_documents:
        raise RerankError(
            f"incomplete response: {len(results)} results for {n_documents} documents"
        )
    return scores


class RerankClient:
    """Validated HTTP client for the class multimodal reranker."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        session=None,
        timeout: float = 60.0,
        max_candidates: int = 50,
    ) -> None:
        self.base_url = (base_url or config.RERANK_BASE_URL).rstrip("/")
        self.model = model or config.RERANK_MODEL
        self.api_key = api_key if api_key is not None else config.RERANK_API_KEY
        config.validate_no_reserved_port(self.base_url)
        if not self.api_key:
            raise RerankError(
                "reranker not configured: set CLASS_API_KEY_RERANK (or "
                "CLASS_API_KEY) in .env"
            )
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
            raise ValueError(f"timeout must be a positive number, got {timeout!r}")
        if max_candidates < 1:
            raise ValueError(f"max_candidates must be >= 1, got {max_candidates!r}")
        self.timeout = timeout
        self.max_candidates = max_candidates
        self._session = session if session is not None else _default_session()

    # -- request path -----------------------------------------------------

    def rank(self, query: str, documents: Sequence[object],
             top_n: Optional[int] = None) -> Dict[int, float]:
        """Rerank ``documents`` against ``query``; returns {input_index: score}.

        Sends exactly one request. ``top_n=0``/``None`` requests scores for
        every candidate (needed for complete before/after evidence).
        """
        docs = list(documents)
        if not query.strip():
            raise RerankError("query must be non-empty")
        if not docs:
            return {}
        if len(docs) > self.max_candidates:
            raise RerankError(
                f"{len(docs)} candidates exceed the client limit of {self.max_candidates}"
            )
        payload = {"model": self.model, "query": query, "documents": docs}
        if top_n is not None:
            payload["top_n"] = top_n

        try:
            # the server mounts the rerank route at both /rerank and
            # /v1/rerank; normalize so either base convention works
            route = "/rerank" if self.base_url.endswith("/v1") else "/v1/rerank"
            response = self._session.post(
                f"{self.base_url}{route}",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
        except RerankError:
            raise
        except Exception as exc:
            import requests as _requests

            is_timeout = isinstance(exc, (_requests.exceptions.Timeout, TimeoutError))
            kind = "timeout" if is_timeout else type(exc).__name__
            raise RerankError(f"reranker request failed ({kind}): {exc}") from exc

        if response.status_code != 200:
            detail = ""
            try:
                detail = str(response.json()).replace(self.api_key, "[REDACTED]")[:200]
            except Exception:
                detail = str(response.text)[:200].replace(self.api_key, "[REDACTED]")
            raise RerankError(f"reranker returned HTTP {response.status_code}: {detail}")

        try:
            body = response.json()
        except Exception as exc:
            raise RerankError(f"reranker returned malformed JSON: {exc}") from exc
        scores = _validate_response(body, len(docs), top_n)
        return scores


def _default_session():
    import requests

    return requests.Session()


# ---- orchestration layer ----------------------------------------------------


@dataclass
class RerankedCandidate:
    """A fused candidate after reranking: everything about the original
    hit is preserved verbatim, plus the reranker's own verdict."""

    original: FusedHit
    rerank_score: Optional[float] = None  # None => reranker did not run
    final_rank: int = 0
    image_included: bool = False
    note: str = ""


@dataclass
class RerankOutcome:
    """Result of one reranking pass, with full before/after evidence."""

    query: str
    applied: bool                                   # reranker ran successfully
    ranked: List[RerankedCandidate]                # final order (fallback = fused order)
    inputs: Sequence[FusedHit]                      # fused ranking BEFORE rerank
    degraded: List[str] = field(default_factory=list)
    images_included: bool = False
    text_only: bool = False

    @property
    def fallback(self) -> bool:
        return not self.applied


def rerank_fused(
    query: str,
    fused_hits: Sequence[FusedHit],
    client: RerankClient,
    top_k: Optional[int] = None,
    include_images: bool = True,
    encode=encode_document,
    max_chars: int = 2000,
    max_image_kb: int = 4000,
    max_images: int = 10,
    validate_image=None,
) -> RerankOutcome:
    """Rerank fused candidates with the class reranker.

    - Never fabricates scores: on any failure the outcome is a clearly
      flagged fallback preserving the original fused order.
    - One HTTP request per query (plus no calls when there is nothing to
      rank or reranking is disabled).
    - Images are attached per candidate up to ``max_images``; each must
      pass ``encode_document``'s validation — including the optional
      ``validate_image`` ownership hook, so a wrong-document image can
      never reach the reranker request (regression-tested).
    """
    hits = list(fused_hits)
    if not hits:
        return RerankOutcome(query=query, applied=False, ranked=[],
                             inputs=[], degraded=["no candidates"])

    payloads: List[object] = []
    notes: List[str] = []
    included_flags: List[bool] = []
    images_attached = 0
    for hit in hits:
        attach_image = include_images and images_attached < max_images
        payload, included, note = encode(
            hit, include_image=attach_image,
            max_chars=max_chars, max_image_kb=max_image_kb,
            validate_image=validate_image,
        )
        payloads.append(payload)
        notes.append(note)
        included_flags.append(included)
        if included:
            images_attached += 1

    try:
        scores = client.rank(query, payloads)
    except Exception as exc:
        degraded = [f"reranker: {exc}"]
        ranked = [
            RerankedCandidate(
                original=hit, rerank_score=None, final_rank=i + 1,
                image_included=False, note="rerank failed; fused order kept",
            )
            for i, hit in enumerate(hits)
        ]
        if top_k is not None:
            ranked = ranked[:top_k]
        return RerankOutcome(
            query=query, applied=False, ranked=ranked, inputs=hits,
            degraded=degraded,
            images_included=False,
            text_only=not include_images,
        )

    candidates = []
    for i, hit in enumerate(hits):
        note = notes[i]
        if top_k is not None and i >= top_k:
            note = (note + "; " if note else "") + "beyond top-k, scored but not returned"
        candidates.append(
            RerankedCandidate(
                original=hit,
                rerank_score=scores.get(i),  # None only if server truncated
                final_rank=0,
                image_included=included_flags[i],
                note=note,
            )
        )

    # deterministic: score desc, then original fused rank, then chunk id
    candidates.sort(
        key=lambda c: (
            -(c.rerank_score if c.rerank_score is not None else -math.inf),
            c.original.fused_rank,
            c.original.chunk.chunk_id,
        )
    )
    for i, c in enumerate(candidates, start=1):
        c.final_rank = i
    if top_k is not None:
        candidates = candidates[:top_k]

    return RerankOutcome(
        query=query,
        applied=True,
        ranked=candidates,
        inputs=hits,
        images_included=any(c.image_included for c in candidates),
        text_only=not include_images,
    )
