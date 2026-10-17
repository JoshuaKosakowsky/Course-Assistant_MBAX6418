"""Server-side quiz session controller for the Gradio interface."""
from __future__ import annotations

from threading import Lock
import time

import gradio as gr

from app.quiz import Quiz, QuizSession
from app.quiz_generation import QuizRequest
from app.quiz_integration import verified_original_image

MAX_QUESTIONS = 5
SESSION_TTL_SECONDS = 3600


class QuizController:
    """Keep answer keys on the server, separated by browser session."""

    def __init__(self, provider=None, store=None):
        self.provider = provider
        self.store = store
        self._sessions = {}
        self._generations = {}
        self._completed_feedback = {}
        self._completed_images = {}
        self._lock = Lock()

    def _get_session_id(self, request: gr.Request) -> str | None:
        return getattr(request, "session_hash", None)

    def _expire(self):
        now = time.monotonic()
        for key, (_, created) in list(self._sessions.items()):
            if now - created > SESSION_TTL_SECONDS:
                del self._sessions[key]
                self._completed_feedback.pop(key, None)
                self._completed_images.pop(key, None)

    @staticmethod
    def _empty_answers():
        return [gr.update(value=None, visible=False, choices=[])
                for _ in range(MAX_QUESTIONS)]

    def generate(self, materials, topic, count, seed, request: gr.Request):
        empty = self._empty_answers()
        sid = self._get_session_id(request)
        if not sid:
            return ("Browser session unavailable.", *empty, "", "", [])
        with self._lock:
            self._expire()
            self._sessions.pop(sid, None)
            self._completed_feedback.pop(sid, None)
            self._completed_images.pop(sid, None)
            generation = self._generations.get(sid, 0) + 1
            self._generations[sid] = generation
        if not materials:
            return ("Please select at least one course document.", *empty, "", "", [])
        if self.provider is None:
            return ("Question generation is not connected yet. No quiz was created.", *empty, "", "", [])
        try:
            if isinstance(count, bool) or int(count) != count:
                raise ValueError("Question count must be a whole number.")
            if isinstance(seed, bool) or int(seed) != seed:
                raise ValueError("Seed must be a whole number.")
            quiz_request = QuizRequest(tuple(materials), topic, int(count), int(seed))
            if quiz_request.count > MAX_QUESTIONS:
                raise ValueError("Choose at most five questions.")
            quiz = self.provider(quiz_request)
            if not isinstance(quiz, Quiz):
                raise ValueError("Provider did not return a Quiz.")
            session = QuizSession(quiz)
            questions = session.student_view()["questions"]
            if len(questions) != quiz_request.count:
                raise ValueError("Generated question count does not match request.")
            controls = []
            for i in range(MAX_QUESTIONS):
                if i < len(questions):
                    q = questions[i]
                    controls.append(gr.update(
                        label=f"Question {i + 1}: {q['prompt']}",
                        choices=q["options"], value=None,
                        visible=True, interactive=True))
                else:
                    controls.append(empty[i])
            with self._lock:
                if self._generations.get(sid) != generation:
                    return ("A newer quiz request has replaced this one.", *empty, "", "", [])
                self._sessions[sid] = (session, time.monotonic())
            return (f"Quiz ready: {len(questions)} questions.", *controls, "", "", [])
        except Exception:
            return (
                "Unable to generate quiz. Please check your selections and try again.",
                *empty,
                "",
                "",
                [],
            )

    def finish(self, reveal: bool, answers, request: gr.Request):
        sid = self._get_session_id(request)
        with self._lock:
            self._expire()
            entry = self._sessions.get(sid) if sid else None
            if entry is None:
                return "Generate a quiz first.", "", []
            session, _ = entry
            view = session.student_view()
            if view["completed"]:
                score, feedback = self._completed_feedback.get(
                    sid, ("This quiz is already complete.", "")
                )
                return score, feedback, self._completed_images.get(sid, [])
            if reveal:
                result = session.reveal()
            else:
                selected = []
                for i, q in enumerate(view["questions"]):
                    if answers[i] is None:
                        return "Answer every question before submitting.", "", []
                    if answers[i] not in q["options"]:
                        return "An answer is invalid. Generate a new quiz.", "", []
                    selected.append(q["options"].index(answers[i]))
                result = session.submit(selected)

        summary = ("Answers revealed without grading." if result["score"] is None
                   else f"Score: {result['score']} / {result['total']}")
        lines = []
        images = []
        for item in result["feedback"]:
            q = view["questions"][item["question"]]
            evidence = item["evidence"]
            correct = q["options"][item["correct_index"]]
            lines.extend([
                f"**Question {item['question'] + 1}**",
                f"Correct answer: {correct}",
                item["explanation"],
                f"Source: {evidence['document']}, page {evidence['page']}",
            ])
            if evidence.get("excerpt"):
                lines.append(f"Evidence: {evidence['excerpt']}")
            if evidence.get("image_path"):
                original = (
                    verified_original_image(self.store, evidence)
                    if self.store is not None
                    else None
                )
                if original:
                    images.append((
                        original,
                        f"Question {item['question'] + 1}: original page {evidence['page']}"
                    ))
                    lines.append("Evidence: Verified original page image shown below.")
                else:
                    lines.append("Evidence: Original image could not be verified.")
            lines.append("")
        completed_output = (summary, "\n\n".join(lines), images)
        with self._lock:
            if self._sessions.get(sid, (None,))[0] is session:
                self._completed_feedback[sid] = completed_output[:2]
                self._completed_images[sid] = images
        return completed_output
