"""Unit tests for the Q&A orchestrator (stubbed LLM, fully offline)."""

from __future__ import annotations

from PIL import Image

import pytest

from app.qna import MISSING_INFO_FALLBACK, QnaConfig, QnaOrchestrator, QnaResult
from app.qna_schema import Source

pytestmark = pytest.mark.unit

VALID_ANSWER = (
    '{"answer": "Retrieval augmented generation combines retrieval with generation.", '
    '"grounded": true, "sources": [{"document": "rag-lecture.pdf", "page": 3, '
    '"excerpt": "RAG combines retrieval with generation."}]}'
)
MISSING_ANSWER = (
    '{"answer": "I don\'t have that in the materials.", "grounded": false, "sources": []}'
)
EVIDENCE = (Source(document="rag-lecture.pdf", page=3, excerpt="RAG combines retrieval with generation."),)


class StubLLM:
    """In-memory LLM returning scripted responses, recording messages."""

    def __init__(self, *responses: str):
        self._responses = list(responses)
        self.calls: list[list[dict]] = []

    def chat(self, messages) -> str:
        self.calls.append(messages)
        if not self._responses:
            raise AssertionError("StubLLM: no scripted responses left")
        return self._responses.pop(0)


def test_happy_path_returns_grounded_answer_with_checks():
    llm = StubLLM(VALID_ANSWER)
    result = QnaOrchestrator(llm).answer("What is RAG?", EVIDENCE)
    assert isinstance(result, QnaResult)
    assert result.grounded is True
    assert result.malformed is False
    assert len(result.sources) == 1
    assert result.sources[0].page == 3
    assert result.source_checks and result.source_checks[0].verdict == "supported"
    assert llm.calls[0][0]["role"] == "system"


def test_missing_info_path_is_honest():
    llm = StubLLM(MISSING_ANSWER)
    result = QnaOrchestrator(llm).answer("What is the exam date?", ())
    assert result.grounded is False
    assert result.sources == ()
    assert "don't have" in result.answer.lower()


def test_grounded_true_without_sources_is_normalized_to_not_grounded():
    llm = StubLLM('{"answer": "Star rating inflation.", "grounded": true, "sources": []}')
    result = QnaOrchestrator(llm).answer("Why so many 5 stars?", ())
    assert result.grounded is False
    assert result.sources == ()


def test_malformed_output_falls_back_honestly():
    llm = StubLLM("I refuse to answer that question.", "still not json")
    result = QnaOrchestrator(llm).answer("q", EVIDENCE)
    assert result.malformed is True
    assert result.grounded is False
    assert MISSING_INFO_FALLBACK in result.answer


def test_retry_recovers_after_one_malformed_reply():
    llm = StubLLM("not json", VALID_ANSWER)
    result = QnaOrchestrator(llm, QnaConfig(retries=1)).answer("What is RAG?", EVIDENCE)
    assert result.grounded is True
    assert result.malformed is False
    assert len(llm.calls) == 2


def test_zero_retries_means_single_attempt():
    llm = StubLLM("not json")
    result = QnaOrchestrator(llm, QnaConfig(retries=0)).answer("q", EVIDENCE)
    assert result.malformed is True
    assert len(llm.calls) == 1


def test_evidence_images_reach_the_llm(tmp_path):
    img = tmp_path / "slide.png"
    Image.new("RGB", (24, 24), (200, 30, 30)).save(img)
    evidence = (Source(document="d", page=1, excerpt="x", image_path=str(img)),)
    llm = StubLLM(VALID_ANSWER)
    QnaOrchestrator(llm).answer("q", evidence)
    user = llm.calls[0][1]["content"]
    assert any(isinstance(p, dict) and p.get("type") == "image_url" for p in user)
