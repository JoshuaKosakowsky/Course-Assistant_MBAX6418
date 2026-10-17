"""Quiz core: deterministic answer placement and safe student views."""
import pytest

from app.quiz import EvidenceRef, Question, Quiz, QuizSession, make_quiz

pytestmark = pytest.mark.unit


def sample_question():
    return Question(
        prompt="Which method retrieves exact terms?",
        options=("BM25", "Cosine", "OCR", "Diffusion"),
        correct_index=0,
        explanation="BM25 matches terms in text.",
        evidence=EvidenceRef("Lecture 5", 12, excerpt="BM25 keyword retrieval"),
    )


def test_same_seed_and_question_yield_same_key_and_order():
    a = make_quiz([sample_question()], seed=42)
    b = make_quiz([sample_question()], seed=42)
    assert a.answer_key == b.answer_key
    assert a.questions == b.questions
    assert a.questions[0].options[a.answer_key[0]] == "BM25"


def test_student_view_never_contains_answer_explanation_or_source_before_submit():
    quiz = make_quiz([sample_question()], seed=42)
    session = QuizSession(quiz)
    view = session.student_view()
    assert "correct_index" not in str(view)
    assert "explanation" not in str(view)
    assert "BM25 keyword retrieval" not in str(view)
    assert "answer_key" not in str(view)
    assert view["questions"][0]["prompt"] == sample_question().prompt


def test_submit_scores_and_exposes_evidence_only_after_submission():
    quiz = make_quiz([sample_question()], seed=42)
    session = QuizSession(quiz)
    result = session.submit([quiz.answer_key[0]])
    assert result["score"] == 1
    assert result["total"] == 1
    assert result["feedback"][0]["correct"] is True
    assert result["feedback"][0]["evidence"]["page"] == 12
    assert result["feedback"][0]["explanation"] == "BM25 matches terms in text."
    assert session.student_view()["completed"] is True


def test_explicit_reveal_exposes_solution_without_scoring():
    session = QuizSession(make_quiz([sample_question()], seed=42))
    solution = session.reveal()
    assert solution["feedback"][0]["correct_index"] in range(4)
    assert solution["score"] is None
    assert session.student_view()["completed"] is True


def test_invalid_answer_count_and_index_rejected_without_reveal():
    session = QuizSession(make_quiz([sample_question()], seed=42))
    with pytest.raises(ValueError):
        session.submit([])
    with pytest.raises(ValueError):
        session.submit([True])
    with pytest.raises(ValueError):
        session.submit([4])
    assert session.student_view()["completed"] is False


def test_rejects_duplicate_options_and_missing_evidence():
    with pytest.raises(ValueError):
        Question("Prompt", ("A", "A", "B", "C"), 0, "why", EvidenceRef("Doc", 1, excerpt="fact"))
    with pytest.raises(ValueError):
        EvidenceRef("Doc", 1)


def test_answer_key_cannot_change_after_submission():
    quiz = make_quiz([sample_question()], seed=42)
    session = QuizSession(quiz)
    session.submit([0])
    with pytest.raises(RuntimeError):
        session.submit([1])
    assert quiz.answer_key == make_quiz([sample_question()], seed=42).answer_key
