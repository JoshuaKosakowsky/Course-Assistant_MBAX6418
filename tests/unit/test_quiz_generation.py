"""Quiz generation validates requested scope and evidence without network calls."""
import pytest

from app.quiz import EvidenceRef, Question
from app.quiz_generation import QuizRequest, generate_quiz

pytestmark = pytest.mark.unit


EVIDENCE = (
    EvidenceRef("Lecture A", 2, excerpt="BM25 retrieves exact keyword matches."),
    EvidenceRef("Lecture B", 3, excerpt="Visual retrieval finds charts."),
)


def sample_question(evidence):
    return Question("What retrieves exact terms?", ("BM25", "OCR", "Diffusion", "Captioning"),
                    0, "The cited page describes BM25.", evidence)


def test_selected_material_and_topic_are_scoped_before_generation():
    seen = []

    def generator(items, topic, count):
        seen.append((items, topic, count))
        return [sample_question(items[0])]

    quiz = generate_quiz(QuizRequest(("Lecture A",), "BM25", 1, 27), EVIDENCE, generator)
    assert seen == [((EVIDENCE[0],), "BM25", 1)]
    assert len(quiz.questions) == 1


def test_rejects_unsupported_source_from_generator():
    def generator(items, topic, count):
        return [sample_question(EvidenceRef("Invented lecture", 2, excerpt="fabricated"))]

    with pytest.raises(ValueError, match="not supplied"):
        generate_quiz(QuizRequest(("Lecture A",), None, 1, 4), EVIDENCE, generator)


def test_rejects_forged_excerpt_on_genuine_document_and_page():
    def generator(items, topic, count):
        return [sample_question(EvidenceRef("Lecture A", 2, excerpt="False excerpt"))]

    with pytest.raises(ValueError, match="not supplied"):
        generate_quiz(QuizRequest(("Lecture A",), None, 1, 4), EVIDENCE, generator)


def test_rejects_question_count_mismatch():
    with pytest.raises(ValueError, match="question count"):
        generate_quiz(QuizRequest(("Lecture A",), None, 2, 4), EVIDENCE,
                      lambda items, topic, count: [sample_question(items[0])])


def test_rejects_missing_topic_evidence_before_calling_generator():
    def never(items, topic, count):
        raise AssertionError("generator must not run")

    with pytest.raises(ValueError, match="no supporting evidence"):
        generate_quiz(QuizRequest(("Lecture A",), "unrelated topic", 1, 4), EVIDENCE, never)


def test_rejects_unselected_material():
    with pytest.raises(ValueError, match="no supporting evidence"):
        generate_quiz(QuizRequest(("Absent",), None, 1, 4), EVIDENCE,
                      lambda items, topic, count: [sample_question(items[0])])


def test_generator_receives_no_unselected_sources_even_without_topic():
    def generator(items, topic, count):
        assert items == (EVIDENCE[1],)
        return [sample_question(items[0])]

    quiz = generate_quiz(QuizRequest(("Lecture B",), None, 1, 4), EVIDENCE, generator)
    assert quiz.questions[0].evidence.document == "Lecture B"


def test_request_rejects_invalid_selection_count_or_seed():
    with pytest.raises(ValueError):
        QuizRequest((), None, 1, 4)
    with pytest.raises(ValueError):
        QuizRequest(("Lecture A",), None, True, 4)
    with pytest.raises(ValueError):
        QuizRequest(("Lecture A",), None, 1, True)
