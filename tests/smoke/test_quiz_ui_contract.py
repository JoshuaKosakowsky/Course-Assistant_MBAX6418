"""UI contract (Issue #7): emulate event handlers; no teammate UI changes."""
import pytest

from app.quiz import Question, QuizSession
from app.quiz_generation import QuizRequest, generate_quiz
from app.quiz_integration import evidence_from_pack

pytestmark = pytest.mark.smoke


class Item:
    filename = "Week 5"
    page_no = 9
    excerpt = "BM25 matches exact terms"
    image_path = None
    image_verified = False


class Pack:
    items = [Item()]


def _generator(scoped, topic, count):
    return [Question("Which method?", ("BM25", "OCR", "ASR", "Diffusion"),
                     0, "BM25 matches exact terms", scoped[0])]


def _generate_handler(materials, topic, count, seed):
    request = QuizRequest(tuple(materials), topic, count, seed)
    quiz = generate_quiz(request, evidence_from_pack(Pack()), _generator)
    session = QuizSession(quiz)  # server-side only; NEVER a Gradio component value
    return session, session.student_view()


def _submit_handler(session, answers):
    return session.submit(answers)


def _reveal_handler(session):
    return session.reveal()


def test_generate_event_shows_no_key_then_submit_returns_feedback():
    session, view = _generate_handler(["Week 5"], "BM25", 1, 42)
    assert set(view) == {"completed", "questions"}
    assert set(view["questions"][0]) == {"prompt", "options"}
    assert "correct_index" not in str(view)
    assert "BM25 matches exact terms" not in str(view)
    # Exercise the UI boundary without reading the server-side quiz key.
    result = _submit_handler(session, [0])
    assert result["total"] == 1
    assert result["score"] in (0, 1)
    assert result["feedback"][0]["evidence"]["page"] == 9
    with pytest.raises(RuntimeError):
        _reveal_handler(session)


def test_reveal_event_requires_explicit_action_and_prevents_late_submit():
    session, view = _generate_handler(["Week 5"], None, 1, 42)
    assert view["completed"] is False
    result = _reveal_handler(session)
    assert result["score"] is None
    assert result["feedback"][0]["correct_index"] in range(4)
    with pytest.raises(RuntimeError):
        _submit_handler(session, [0])


def test_invalid_ui_submission_does_not_unlock_solutions():
    session, _ = _generate_handler(["Week 5"], None, 1, 42)
    with pytest.raises(ValueError):
        _submit_handler(session, [])
    assert session.student_view()["completed"] is False
