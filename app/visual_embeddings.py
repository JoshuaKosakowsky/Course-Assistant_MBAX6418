"""Embedders for the visual index.

Two roles:

- ``ClassEmbeddingClient`` talks to the class visual-embedding endpoint
  (Qwen3-VL, port 9003 — OpenAI-compatible API) using app.config / .env
  settings. The exact image input encoding is our working assumption
  (a JPEG data URI); the e2e tests in tests/e2e/ document the live
  contract once keys exist. Text-to-image queries use the SAME model so
  vectors stay comparable with the visual index.
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

from . import config

# Module-level copies so tests can monkeypatch them (config is read once).
VISUAL_EMBED_BASE_URL = config.VISUAL_EMBED_BASE_URL
VISUAL_EMBED_API_KEY = config.VISUAL_EMBED_API_KEY
VISUAL_EMBED_MODEL = config.VISUAL_EMBED_MODEL
TEXT_EMBED_BASE_URL = config.TEXT_EMBED_BASE_URL
TEXT_EMBED_MODEL = config.TEXT_EMBED_MODEL

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

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.base_url = (base_url or VISUAL_EMBED_BASE_URL).rstrip("/")
        self.api_key = api_key if api_key is not None else VISUAL_EMBED_API_KEY
        self.model = model or VISUAL_EMBED_MODEL
        config.validate_no_reserved_port(self.base_url)
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
