"""Client for the class text-embedding service (Issue #2 / Phase 2).

Endpoint (verified in M0): ``POST {base}/v1/embeddings`` on
``http://dobolyi.com:9002``, OpenAI-compatible, model
``nvidia/Nemotron-3-Embed-1B-BF16``, 2048-dim output.

Security rules observed here:

- The API key comes from the environment / gitignored ``.env`` only; it is
  never printed, logged, or included in error messages.
- ``requests.Session`` is injectable so tests can run fully offline.
- Port 9000 (reserved for Hermes agentic operations) is refused.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

import requests

from app import config


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
