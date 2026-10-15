"""LLM clients for the class endpoints (Phase 4 Q&A).

``ClassVisionClient`` wraps the openai-compatible chat endpoint at
dobolyi.com (port 9001, Qwen3.6-35B — verified by the team). Image parts
are passed as base64 data URIs; the exact contract the endpoint accepts is
probed by the e2e tests in tests/e2e/test_qna_e2e.py once .env keys exist.
Offline tests inject a stub with the same ``chat(messages) -> str`` shape.

Endpoint settings are read directly from the gitignored .env here; once
the Phase 1 branch merges (app/config.py) and the visual-index branch
lands, these fold into the shared config module.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_REPO_ROOT / ".env")

VISION_BASE_URL = os.getenv("BASE_URL_VISION", "")
VISION_API_KEY = os.getenv("CLASS_API_KEY", "")
VISION_MODEL = os.getenv("VISION_MODEL", "")


class ClassVisionClient:
    """OpenAI-compatible chat client for the class vision LLM endpoint."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.base_url = base_url or VISION_BASE_URL
        self.api_key = api_key or VISION_API_KEY
        self.model = model or VISION_MODEL
        if not (self.base_url and self.api_key and self.model):
            raise RuntimeError(
                "Vision LLM not configured: set BASE_URL_VISION, CLASS_API_KEY "
                "and VISION_MODEL in .env (see .env.example)."
            )

    def chat(self, messages: list[dict[str, Any]]) -> str:
        from openai import OpenAI  # deferred so offline paths never import it

        client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=120)
        response = client.chat.completions.create(
            model=self.model, messages=messages, temperature=0.0
        )
        return response.choices[0].message.content or ""
