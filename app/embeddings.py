"""Embedders for the visual index.

Two roles:

- ``ClassEmbeddingClient`` talks to the class endpoints (openai-compatible
  embeddings API) using .env settings. The exact image input encoding is
  our working assumption (a JPEG data URI); the e2e tests in
  tests/e2e/ document the live contract once keys exist.
- ``fake_embedder`` is a deterministic offline stand-in for tests: the
  same image always yields the same vector, different images differ, and
  vectors are unit-length so distance comparisons are meaningful.
"""

from __future__ import annotations

import base64
import hashlib
import io
from typing import Callable

from PIL import Image

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
        """Embed one page/slide image via the class visual endpoint."""
        return self._embed([image_to_data_uri(image)], VISUAL_EMBED_MODEL)

    def embed_text(self, text: str) -> list[float]:
        """Embed free text via the class text-embedding endpoint (bridge)."""
        if not TEXT_EMBED_MODEL:
            raise RuntimeError(
                "Set TEXT_EMBED_MODEL in .env for the text-to-image bridge."
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
