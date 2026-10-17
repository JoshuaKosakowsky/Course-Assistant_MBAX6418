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
    assert result.citation_checks and all(ck.valid for ck in result.citation_checks)
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


def test_unsupported_sources_never_grounded():
    """An answer whose cited evidence does not support its claims must not
    be presented as grounded, even if the citation itself is authentic."""
    evidence = (Source(document="syllabus.pdf", page=1, excerpt="Grading policy is posted on Canvas."),)
    answer = (
        '{"answer": "The exam is on Friday.", "grounded": true, "sources": ['
        '{"document": "syllabus.pdf", "page": 1, "excerpt": "Grading policy is posted on Canvas."}]}'
    )
    llm = StubLLM(answer)
    result = QnaOrchestrator(llm).answer("When is the exam?", evidence)
    assert result.grounded is False
    assert result.sources == ()
    assert result.citation_checks and result.citation_checks[0].valid


def test_fabricated_citation_is_rejected_and_dropped():
    """A citation to a document/page that was never supplied must never
    reach the answer as evidence."""
    evidence = (Source(document="rag-lecture.pdf", page=3, excerpt="RAG combines retrieval with generation."),)
    answer = (
        '{"answer": "It is on slide 3 of the syllabus.", "grounded": true, "sources": ['
        '{"document": "secret-notes.pdf", "page": 9, "excerpt": "The final is in week 15."}]}'
    )
    llm = StubLLM(answer)
    result = QnaOrchestrator(llm).answer("Where is it stated?", evidence)
    assert result.grounded is False
    assert result.sources == ()
    assert len(result.citation_checks) == 1 and result.citation_checks[0].valid is False


def test_wrong_excerpt_for_real_document_is_rejected():
    """Same document/page but an excerpt that is not in the evidence is a
    fabricated quote and must be dropped."""
    evidence = (Source(document="rag-lecture.pdf", page=3, excerpt="RAG combines retrieval with generation."),)
    answer = (
        '{"answer": "RAG is a retrieval system.", "grounded": true, "sources": ['
        '{"document": "rag-lecture.pdf", "page": 3, "excerpt": "RAG stands for robot and goat."}]}'
    )
    llm = StubLLM(answer)
    result = QnaOrchestrator(llm).answer("What is RAG?", evidence)
    assert result.grounded is False
    assert result.sources == ()


def test_mixed_citations_keep_only_verified_sources():
    evidence = (
        Source(document="rag-lecture.pdf", page=3, excerpt="RAG combines retrieval with generation."),
        Source(document="syllabus.pdf", page=1, excerpt="Office hours are on Tuesdays."),
    )
    answer = (
        '{"answer": "RAG combines retrieval with generation per the lecture.", "grounded": true, "sources": ['
        '{"document": "rag-lecture.pdf", "page": 3, "excerpt": "RAG combines retrieval with generation."},'
        '{"document": "invented.pdf", "page": 2, "excerpt": "Made up evidence."}]}'
    )
    llm = StubLLM(answer)
    result = QnaOrchestrator(llm).answer("What is RAG?", evidence)
    assert result.grounded is True
    assert [s.document for s in result.sources] == ["rag-lecture.pdf"]
    assert len(result.citation_checks) == 2 and not result.citation_checks[1].valid


def test_grounded_string_false_is_normalized_to_not_grounded():
    """Regression: model emitting grounded: \"false\" must be ungrounded."""
    llm = StubLLM('{"answer": "unknown", "grounded": "false", "sources": []}')
    result = QnaOrchestrator(llm).answer("q", ())
    assert result.grounded is False


def test_unsupported_answer_text_is_replaced_by_fallback():
    """Regression: ungrounded paths must never surface the model's raw text —
    a claim with no valid supporting evidence is replaced wholesale."""
    evidence = (Source(document="syllabus.pdf", page=1, excerpt="Grading policy is posted on Canvas."),)
    answer = (
        '{"answer": "The final exam is Friday.", "grounded": true, "sources": ['
        '{"document": "syllabus.pdf", "page": 1, "excerpt": "Grading policy is posted on Canvas."}]}'
    )
    result = QnaOrchestrator(StubLLM(answer)).answer("When is the final?", evidence)
    assert result.grounded is False
    assert result.answer == MISSING_INFO_FALLBACK
    assert "final exam is Friday" not in result.answer


def test_ungrounded_claim_never_reaches_ui():
    """A grounded:false model reply that still states a claim must be replaced
    by the evidence-safe fallback, not passed through."""
    claim = '{"answer": "The final exam is Friday.", "grounded": false, "sources": []}'
    result = QnaOrchestrator(StubLLM(claim)).answer("When is the final?", ())
    assert result.grounded is False
    assert result.answer == MISSING_INFO_FALLBACK
    assert "Friday" not in result.answer


def test_unverified_fabricated_citation_replaced_by_fallback():
    """No citation survives verification -> evidence-safe fallback text."""
    evidence = (Source(document="rag-lecture.pdf", page=3, excerpt="RAG combines retrieval with generation."),)
    answer = (
        '{"answer": "The secret is out.", "grounded": true, "sources": ['
        '{"document": "secret-notes.pdf", "page": 9, "excerpt": "Top secret."}]}'
    )
    result = QnaOrchestrator(StubLLM(answer)).answer("What is the secret?", evidence)
    assert result.grounded is False
    assert result.answer == MISSING_INFO_FALLBACK
    assert "secret is out" not in result.answer


def test_fabricated_image_path_never_reaches_sources():
    """Citation authentic in doc/page/excerpt but image path fabricated:
    the citation survives, but the image path is dropped (evidence-derived)."""
    evidence = (Source(document="rag-lecture.pdf", page=3, excerpt="RAG combines retrieval with generation.",
                       image_path="slides/rag3.png"),)
    answer = (
        '{"answer": "RAG combines retrieval with generation per the lecture.", "grounded": true, "sources": ['
        '{"document": "rag-lecture.pdf", "page": 3, "excerpt": "RAG combines retrieval with generation.", '
        '"image_path": "/etc/passwd"}]}'
    )
    result = QnaOrchestrator(StubLLM(answer)).answer("What is RAG?", evidence)
    assert result.grounded is True
    assert result.sources[0].image_path is None
    assert "discarded" in result.citation_checks[0].reason


def test_model_image_path_derived_from_evidence_when_matching():
    """A model-supplied image path is kept only as the evidence's own path."""
    evidence = (Source(document="rag-lecture.pdf", page=3, excerpt="RAG combines retrieval with generation.",
                       image_path="slides/rag3.png"),)
    answer = (
        '{"answer": "RAG combines retrieval with generation per the lecture.", "grounded": true, "sources": ['
        '{"document": "rag-lecture.pdf", "page": 3, "excerpt": "RAG combines retrieval with generation.", '
        '"image_path": "slides/rag3.png"}]}'
    )
    result = QnaOrchestrator(StubLLM(answer)).answer("What is RAG?", evidence)
    assert result.grounded is True
    assert result.sources[0].image_path == "slides/rag3.png"


def test_contradictory_facts_never_grounded():
    """Answer citing an authentic excerpt whose facts contradict the answer
    (different exam date) must not be presented as grounded."""
    evidence = (Source(document="syllabus.pdf", page=1, excerpt="The exam is on October 15 in week 15 of the course."),)
    answer = (
        '{"answer": "The exam is on October 20.", "grounded": true, "sources": ['
        '{"document": "syllabus.pdf", "page": 1, "excerpt": "The exam is on October 15 in week 15 of the course."}]}'
    )
    result = QnaOrchestrator(StubLLM(answer)).answer("When is the exam?", evidence)
    assert result.grounded is False
    assert result.answer == MISSING_INFO_FALLBACK
    assert result.source_checks and result.source_checks[0].verdict == "weak"


def test_mixed_support_presents_only_supported_sources():
    """One supported + one unsupported citation: grounded, but only the
    supported source is presented; the unsupported verdict stays diagnostic."""
    evidence = (
        Source(document="rag-lecture.pdf", page=3, excerpt="RAG combines retrieval with generation."),
        Source(document="syllabus.pdf", page=1, excerpt="Office hours are on Tuesdays."),
    )
    answer = (
        '{"answer": "RAG combines retrieval with generation per the lecture.", "grounded": true, "sources": ['
        '{"document": "rag-lecture.pdf", "page": 3, "excerpt": "RAG combines retrieval with generation."},'
        '{"document": "syllabus.pdf", "page": 1, "excerpt": "Office hours are on Tuesdays."}]}'
    )
    result = QnaOrchestrator(StubLLM(answer)).answer("What is RAG?", evidence)
    assert result.grounded is True
    assert [s.document for s in result.sources] == ["rag-lecture.pdf"]
    assert all(ck.valid for ck in result.citation_checks)  # both citations were authentic
    assert [ck.verdict for ck in result.source_checks] == ["supported", "unsupported"]
