"""Synthetic end-to-end quiz flow: generation, concealment, scoring, evidence."""
import pytest

from app.quiz import EvidenceRef, Question, QuizSession
from app.quiz_generation import QuizRequest, generate_quiz

pytestmark = pytest.mark.smoke


def test_generate_answer_score_and_reveal_source():
    evidence = (EvidenceRef("Week 5", 9, excerpt="BM25 is keyword retrieval",
                            image_path="artifacts/week5/page-009.png", doc_id="week5"),)
    request = QuizRequest(("week5",), "BM25", 1, 2026)

    def fake_generator(scoped, topic, count):
        assert scoped == evidence
        assert topic == "BM25"
        assert count == 1
        return [Question("Which method matches keywords?",
                         ("BM25", "OCR", "ASR", "Diffusion"), 0,
                         "The excerpt identifies BM25 as keyword retrieval.", scoped[0])]

    quiz = generate_quiz(request, evidence, fake_generator)
    session = QuizSession(quiz)
    before = session.student_view()
    assert before["completed"] is False
    assert "answer_key" not in before
    assert "explanation" not in str(before)
    assert "BM25 is keyword retrieval" not in str(before)
    result = session.submit([quiz.answer_key[0]])
    assert (result["score"], result["total"]) == (1, 1)
    assert result["feedback"][0]["evidence"] == {
        "document": "Week 5", "page": 9,
        "excerpt": "BM25 is keyword retrieval",
        "image_path": "artifacts/week5/page-009.png",
    }
    assert session.student_view()["completed"] is True


def test_regeneration_same_seed_keeps_answers_stable():
    evidence = (EvidenceRef("Week 5", 9, excerpt="BM25 keyword retrieval", doc_id="week5"),)
    request = QuizRequest(("week5",), None, 1, 2026)

    def generator(items, topic, count):
        return [Question("Which matches words?", ("BM25", "OCR", "ASR", "Diffusion"),
                         0, "BM25 matches words", items[0])]

    first = generate_quiz(request, evidence, generator)
    second = generate_quiz(request, evidence, generator)
    assert first.answer_key == second.answer_key
    assert first.questions == second.questions
