"""Embedding clients: class text-embedding service and visual index embedders.

Two independently-tested halves:

- ``TextEmbeddingClient`` (Issue #2 / Phase 2): OpenAI-compatible text
  embeddings endpoint (``http://dobolyi.com:9002``,
  ``nvidia/Nemotron-3-Embed-1B-BF16``, 2048-dim). API key lives in the
  gitignored ``.env`` only — never printed, logged, or included in error
  messages; ``requests.Session`` is injectable for offline tests; the
  reserved port 9000 is refused.
- ``ClassEmbeddingClient`` + ``fake_embedder`` (Issue #3 / visual index):
  image embeddings in the visual model's space via the openai-compatible
  class endpoint. The exact image input encoding is a working assumption
  (JPEG data URI); e2e tests document the live contract once keys exist.
  ``fake_embedder`` is a deterministic offline stand-in: identical images
  yield identical unit-length vectors, different images differ.
"""

from __future__ import annotations

import base64
import hashlib
import io
import os
from typing import Callable, List, Optional, Sequence

import requests
from PIL import Image

from app import config
from .env import (
    TEXT_EMBED_BASE_URL,
    TEXT_EMBED_MODEL,
    VISUAL_EMBED_API_KEY,
    VISUAL_EMBED_BASE_URL,
    VISUAL_EMBED_MODEL,
)

ImageEmbedder = Callable[[Image.Image], list[float]]
TextEmbedder = Callable[[str], list[float]]

_IMAGE_CONTENT_TYPES = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "jpg": "image/jpeg",
    "webp": "image/webp",
}


class EmbeddingsError(RuntimeError):
    """Raised for any request/response problem with the embedding service."""


class TextEmbeddingClient:
    """Thin, validated client for an OpenAI-compatible embeddings endpoint.

    Parameters are overridable so offline tests can inject fakes; by
    default everything comes from ``app.config`` / the local ``.env``.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        session: Optional[requests.Session] = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = (base_url or config.TEXT_EMBED_BASE_URL).rstrip("/")
        self.api_key = api_key if api_key is not None else config.TEXT_EMBED_API_KEY
        self.model = model or config.TEXT_EMBED_MODEL
        self.timeout = timeout
        self._session = session if session is not None else requests.Session()

        config.validate_no_reserved_port(self.base_url)
        if not self.api_key:
            raise EmbeddingsError(
                "No CLASS_API_KEY configured. Create a local .env from "
                ".env.example and fill in the real class key (never commit it)."
            )

    @property
    def url(self) -> str:
        return f"{self.base_url}/embeddings"

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed a list of texts; returns one vector per input text.

        Validates the response shape (count, per-row vectors, consistent
        dimension) and raises EmbeddingsError for anything else. Empty
        inputs return an empty list; whitespace-only texts are rejected.
        """
        texts = list(texts)
        if not texts:
            return []
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                raise EmbeddingsError(
                    "cannot embed empty or whitespace-only text"
                )

        response = self._session.post(
            self.url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self.model, "input": texts},
            timeout=self.timeout,
        )
        if response.status_code != 200:
            snippet = (response.text or "")[:200].replace("\n", " ")
            raise EmbeddingsError(
                f"embedding request failed with status {response.status_code}: "
                f"{snippet}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise EmbeddingsError(
                f"embedding response was not valid JSON: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise EmbeddingsError(
                f"malformed embedding response: expected a JSON object, "
                f"got {type(payload).__name__}"
            )

        rows = payload.get("data")
        if not isinstance(rows, list) or len(rows) != len(texts):
            raise EmbeddingsError(
                f"malformed embedding response: expected data for "
                f"{len(texts)} input(s), got {rows!r}"
            )
        vectors: List[List[float]] = []
        dimension: Optional[int] = None
        for row in rows:
            embedding = row.get("embedding") if isinstance(row, dict) else None
            if not isinstance(embedding, list):
                raise EmbeddingsError(
                    f"malformed embedding response: missing per-row embedding "
                    f"({row!r})"
                )
            try:
                vector = [float(x) for x in embedding]
            except (TypeError, ValueError) as exc:
                raise EmbeddingsError(
                    f"malformed embedding response: non-numeric vector value "
                    f"({exc})"
                ) from exc
            if dimension is None:
                dimension = len(vector)
            elif len(vector) != dimension:
                raise EmbeddingsError(
                    f"inconsistent embedding dimensions: {dimension} vs "
                    f"{len(vector)}"
                )
            vectors.append(vector)
        return vectors

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]

    def __repr__(self) -> str:
        return f"TextEmbeddingClient(base_url={self.base_url!r}, model={self.model!r})"


def image_to_data_uri(image: Image.Image) -> str:
    """Encode a PIL image as a base64 data URI (assumed endpoint input)."""
    fmt = (image.format or "JPEG").lower()
    mime = _IMAGE_CONTENT_TYPES.get(fmt, "image/jpeg")
    buf = io.BytesIO()
    if mime == "image/jpeg":
        image.convert("RGB").save(buf, format="JPEG", quality=90)
    else:
        image.convert("RGB").save(buf, format=fmt.upper())
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode("ascii")


EMBED_IMAGE_MAX_EDGE = 512


def resize_for_embedding(image: Image.Image, max_edge: int = EMBED_IMAGE_MAX_EDGE) -> Image.Image:
    """Return a copy downscaled to at most ``max_edge`` on the longest side.

    The embedding endpoint rejects images above its context limit (150-dpi
    page/slide renders exceed it). The original image and its stored path
    are never modified — this copy exists only for the embedding request.
    Images already within the limit are returned as-is.
    """
    width, height = image.size
    longest = max(width, height)
    if longest <= max_edge:
        return image
    scale = max_edge / longest
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.convert("RGB").resize(new_size, Image.Resampling.LANCZOS)


class ClassEmbeddingClient:
    """OpenAI-compatible embeddings client for the class endpoints.

    Raises RuntimeError when .env settings are missing; callers that need
    offline behavior inject a fake embedder instead.
    """

    def __init__(self, *, base_url: str | None = None, api_key: str | None = None):
        self.base_url = base_url or VISUAL_EMBED_BASE_URL
        self.api_key = api_key or VISUAL_EMBED_API_KEY
        if not self.base_url or not self.api_key:
            raise RuntimeError(
                "Class embedding endpoint not configured: set "
                "BASE_URL_VISUAL_EMBED and CLASS_API_KEY in .env "
                "(see .env.example)."
            )

    def embed_image(self, image: Image.Image) -> list[float]:
        """Embed one page/slide image via the class visual endpoint.

        The image is downscaled (512px longest edge) for the embedding
        request; the caller's original file is never modified.
        """
        if not VISUAL_EMBED_MODEL:
            raise RuntimeError("Set VISUAL_EMBED_MODEL in .env (see .env.example).")
        return self._embed([image_to_data_uri(resize_for_embedding(image))], VISUAL_EMBED_MODEL)

    def embed_text_visual(self, text: str) -> list[float]:
        """Embed free text in the SAME space as images (Qwen3-VL, port 9003).

        Required for the text-to-image bridge: only vectors produced by the
        same model/endpoint are comparable against the visual index.
        """
        if not VISUAL_EMBED_MODEL:
            raise RuntimeError("Set VISUAL_EMBED_MODEL in .env for the text-to-image bridge.")
        return self._embed([text], VISUAL_EMBED_MODEL)

    def embed_text(self, text: str) -> list[float]:
        """Embed free text with the TEXT-embedding model (port 9002).

        This is the text-index space (issue #2) — NOT comparable with the
        visual index. For the text-to-image bridge use
        :meth:`embed_text_visual` with the visual model instead.
        """
        if not TEXT_EMBED_MODEL:
            raise RuntimeError(
                "Set TEXT_EMBED_MODEL in .env for text-space embeddings."
            )
        return self._embed([text], TEXT_EMBED_MODEL, base=TEXT_EMBED_BASE_URL)

    def _embed(self, inputs: list[str], model: str, base: str | None = None) -> list[float]:
        from openai import OpenAI  # deferred so offline paths never import it

        client = OpenAI(base_url=base or self.base_url, api_key=self.api_key, timeout=60)
        response = client.embeddings.create(model=model, input=inputs)
        return response.data[0].embedding


def fake_embedder(dims: int = 8, seed: str = "visual-index") -> ImageEmbedder:
    """Deterministic pseudo-embedding derived from image bytes.

    Designed for offline unit/smoke tests only: identical images produce
    identical vectors, different images produce different vectors, and all
    vectors are unit-length.
    """

    def embed(image: Image.Image) -> list[float]:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        digest = hashlib.sha256(seed.encode("utf-8") + buf.getvalue()).digest()
        vec = []
        for i in range(dims):
            b = hashlib.sha256(bytes([digest[i % len(digest)]]) + i.to_bytes(4, "little")).digest()
            vec.append(int.from_bytes(b[:4], "little") / (2**32) - 0.5)
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    return embed
