"""Issue #7 quiz controller smoke tests."""
from types import SimpleNamespace

from app.quiz import EvidenceRef, Question, make_quiz
from app.quiz_ui import QuizController


def provider(request):
    evidence = EvidenceRef("lecture.pdf", 2, "BM25 matches terms", doc_id="doc1")
    question = Question("Which method?", ("BM25", "OCR", "ASR", "RNN"),
                        0, "BM25 is lexical retrieval.", evidence)
    return make_quiz([question] * request.count, request.seed)


def browser(sid):
    return SimpleNamespace(session_hash=sid)


def test_missing_materials_or_provider():
    controller = QuizController()
    assert "select" in controller.generate([], "", 1, 42, browser("a"))[0].lower()
    assert "not connected" in controller.generate(["doc1"], "", 1, 42, browser("a"))[0]
    assert not controller._sessions


def test_submit_and_source_feedback():
    controller = QuizController(provider)
    result = controller.generate(["doc1"], None, 1, 42, browser("a"))
    assert "Quiz ready" in result[0]
    assert "Correct answer" not in str(result)
    assert "BM25 is lexical retrieval" not in str(result)
    assert "Answer every" in controller.finish(False, [None] * 5, browser("a"))[0]
    view = controller._sessions["a"][0].student_view()
    choice = view["questions"][0]["options"][0]
    score, feedback = controller.finish(False, [choice] + [None] * 4, browser("a"))
    assert score.startswith("Score:")
    assert "lecture.pdf, page 2" in feedback
    assert "Correct answer:" in feedback
    assert controller.finish(True, [None] * 5, browser("a"))[0] == "This quiz is already complete."


def test_reveal_isolated_by_browser():
    controller = QuizController(provider)
    controller.generate(["doc1"], None, 1, 42, browser("a"))
    assert controller.finish(True, [None] * 5, browser("b"))[0] == "Generate a quiz first."
    score, feedback = controller.finish(True, [None] * 5, browser("a"))
    assert score == "Answers revealed without grading."
    assert "Correct answer:" in feedback


def test_new_attempt_invalidates_old_session():
    controller = QuizController(provider)
    controller.generate(["doc1"], None, 1, 42, browser("a"))
    controller.generate([], None, 1, 42, browser("a"))
    assert controller.finish(False, [None] * 5, browser("a"))[0] == "Generate a quiz first."
