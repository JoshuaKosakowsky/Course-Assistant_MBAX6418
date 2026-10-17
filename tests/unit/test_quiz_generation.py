"""Quiz generation validates requested scope and evidence without network calls."""
import pytest

from app.quiz import EvidenceRef, Question
from app.quiz_generation import QuizRequest, generate_quiz

pytestmark = pytest.mark.unit


EVIDENCE = (
    EvidenceRef("Lecture A", 2, excerpt="BM25 retrieves exact keyword matches.", doc_id="doc-a"),
    EvidenceRef("Lecture B", 3, excerpt="Visual retrieval finds charts.", doc_id="doc-b"),
)


def sample_question(evidence):
    return Question("What retrieves exact terms?", ("BM25", "OCR", "Diffusion", "Captioning"),
                    0, "The cited page describes BM25.", evidence)


def test_selected_document_and_topic_are_scoped_before_generation():
    seen = []

    def generator(items, topic, count):
        seen.append((items, topic, count))
        return [sample_question(items[0])]

    quiz = generate_quiz(QuizRequest(("doc-a",), "BM25", 1, 27), EVIDENCE, generator)
    assert seen == [((EVIDENCE[0],), "BM25", 1)]
    assert len(quiz.questions) == 1


def test_rejects_unsupported_source_from_generator():
    def generator(items, topic, count):
        return [sample_question(EvidenceRef("Invented lecture", 2, excerpt="fabricated",
                                            doc_id="invented"))]

    with pytest.raises(ValueError, match="not supplied"):
        generate_quiz(QuizRequest(("doc-a",), None, 1, 4), EVIDENCE, generator)


def test_rejects_forged_excerpt_on_genuine_document_and_page():
    def generator(items, topic, count):
        return [sample_question(EvidenceRef("Lecture A", 2, excerpt="False excerpt",
                                            doc_id="doc-a"))]

    with pytest.raises(ValueError, match="not supplied"):
        generate_quiz(QuizRequest(("doc-a",), None, 1, 4), EVIDENCE, generator)


def test_rejects_question_count_mismatch():
    with pytest.raises(ValueError, match="question count"):
        generate_quiz(QuizRequest(("doc-a",), None, 2, 4), EVIDENCE,
                      lambda items, topic, count: [sample_question(items[0])])


def test_rejects_missing_topic_evidence_before_calling_generator():
    def never(items, topic, count):
        raise AssertionError("generator must not run")

    with pytest.raises(ValueError, match="no supporting evidence"):
        generate_quiz(QuizRequest(("doc-a",), "unrelated topic", 1, 4), EVIDENCE, never)


def test_rejects_unselected_document():
    with pytest.raises(ValueError, match="no supporting evidence"):
        generate_quiz(QuizRequest(("absent-doc",), None, 1, 4), EVIDENCE,
                      lambda items, topic, count: [sample_question(items[0])])


def test_generator_receives_no_unselected_sources_even_without_topic():
    def generator(items, topic, count):
        assert items == (EVIDENCE[1],)
        return [sample_question(items[0])]

    quiz = generate_quiz(QuizRequest(("doc-b",), None, 1, 4), EVIDENCE, generator)
    assert quiz.questions[0].evidence.document == "Lecture B"
    assert quiz.questions[0].evidence.doc_id == "doc-b"


def test_same_filename_different_doc_ids_scope_by_doc_id():
    """Two documents named Lecture.pdf must not leak across selections."""
    deck_a = EvidenceRef("Lecture.pdf", 2, excerpt="BM25 matches exact terms.", doc_id="doc-1")
    deck_b = EvidenceRef("Lecture.pdf", 3, excerpt="Visual retrieval finds charts.", doc_id="doc-2")
    same_name = (deck_a, deck_b)

    def generator(items, topic, count):
        assert items == (deck_a,)
        return [sample_question(items[0])]

    quiz = generate_quiz(QuizRequest(("doc-1",), None, 1, 11), same_name, generator)
    assert quiz.questions[0].evidence.doc_id == "doc-1"
    assert quiz.questions[0].evidence.document == "Lecture.pdf"


def test_evidence_without_doc_id_cannot_match_document_selection():
    """Evidence with no doc_id fails closed under document-scoped requests."""
    def never(items, topic, count):
        raise AssertionError("generator must not run")

    legacy = (EvidenceRef("Lecture A", 2, excerpt="No doc id attached."),)
    with pytest.raises(ValueError, match="no supporting evidence"):
        generate_quiz(QuizRequest(("doc-a",), None, 1, 4), legacy, never)


def test_request_rejects_invalid_selection_count_or_seed():
    with pytest.raises(ValueError):
        QuizRequest((), None, 1, 4)
    with pytest.raises(ValueError):
        QuizRequest(("doc-a",), None, True, 4)
    with pytest.raises(ValueError):
        QuizRequest(("doc-a",), None, 1, True)
    with pytest.raises(ValueError):
        QuizRequest(("  ",), None, 1, 4)
