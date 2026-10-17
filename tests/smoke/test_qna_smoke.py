"""Smoke tests for the Q&A core — offline, repo persona, stubbed LLM."""

from __future__ import annotations

import pytest
from PIL import Image

from app.prompts import PERSONA_FILE, load_persona
from app.qna import QnaOrchestrator
from app.qna_schema import Source

VALID_ANSWER = (
    '{"answer": "The pipeline retrieves evidence and grounds the answer in it.", '
    '"grounded": true, "sources": [{"document": "rag-lecture.pdf", "page": 1, '
    '"excerpt": "retrieval ground evidence pipeline answer"}]}'
)


class StubLLM:
    def __init__(self, payload: str):
        self._payload = payload

    def chat(self, messages) -> str:
        return self._payload


@pytest.mark.smoke
def test_pipeline_with_repo_persona_and_image(tmp_path):
    persona = load_persona()
    assert "course assistant" in persona.lower()  # real repo persona resolves

    slide = tmp_path / "slide.png"
    Image.new("RGB", (64, 64), (30, 120, 220)).save(slide)

    evidence = (
        Source(document="rag-lecture.pdf", page=1, excerpt="retrieval ground evidence pipeline answer"),
        Source(document="slides.pdf", page=4, excerpt="diagram of the fusion step", image_path=str(slide)),
    )
    llm = StubLLM(VALID_ANSWER)
    result = QnaOrchestrator(llm).answer("How does the pipeline work?", evidence)

    assert result.grounded is True
    assert len(result.sources) == 1
    assert result.source_checks and result.source_checks[0].verdict in {"supported", "weak", "unsupported"}


@pytest.mark.smoke
def test_persona_file_is_committed_for_runtime():
    assert PERSONA_FILE.is_file(), "docs/PERSONA.md must exist (single source of truth)"
    assert len(load_persona()) > 500
