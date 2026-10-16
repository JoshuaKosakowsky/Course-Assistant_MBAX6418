"""Offline quiz domain: fixed answer keys and safe student-facing views.

Generation is separate from presentation. Never serialize ``Quiz`` directly to
students; use ``QuizSession.student_view`` until they submit or reveal.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class EvidenceRef:
    document: str
    page: int
    excerpt: str | None = None
    image_path: str | None = None
    doc_id: str | None = None

    def __post_init__(self) -> None:
        if not self.document.strip() or isinstance(self.page, bool) or not isinstance(self.page, int) or self.page < 1:
            raise ValueError("evidence needs a document and positive page")
        if not (self.excerpt and self.excerpt.strip()) and not (self.image_path and self.image_path.strip()):
            raise ValueError("evidence needs an excerpt or image")
        if self.doc_id is not None and (not isinstance(self.doc_id, str) or not self.doc_id.strip()):
            raise ValueError("doc_id must be a non-empty string or None")

    def public(self) -> dict:
        return {"document": self.document, "page": self.page,
                "excerpt": self.excerpt, "image_path": self.image_path}


@dataclass(frozen=True)
class Question:
    prompt: str
    options: tuple[str, ...]
    correct_index: int
    explanation: str
    evidence: EvidenceRef

    def __post_init__(self) -> None:
        if not self.prompt.strip() or not self.explanation.strip():
            raise ValueError("question needs a prompt and explanation")
        if len(self.options) != 4 or any(not x.strip() for x in self.options) or len(set(self.options)) != 4:
            raise ValueError("question needs four distinct nonempty options")
        if isinstance(self.correct_index, bool) or not isinstance(self.correct_index, int) or not 0 <= self.correct_index < 4:
            raise ValueError("correct index must select an option")


@dataclass(frozen=True)
class Quiz:
    questions: tuple[Question, ...]
    seed: int

    @property
    def answer_key(self) -> tuple[int, ...]:
        return tuple(q.correct_index for q in self.questions)


def make_quiz(questions: Sequence[Question], seed: int) -> Quiz:
    """Shuffle answer placement deterministically; retain question order."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not questions:
        raise ValueError("quiz must contain at least one question")
    shuffled = []
    for q in questions:
        fingerprint = repr((seed, q.prompt, q.options)).encode("utf-8")
        rng = random.Random(int.from_bytes(hashlib.sha256(fingerprint).digest(), "big"))
        order = list(range(4))
        rng.shuffle(order)
        shuffled.append(Question(q.prompt, tuple(q.options[i] for i in order),
                                 order.index(q.correct_index), q.explanation, q.evidence))
    return Quiz(tuple(shuffled), seed)


class QuizSession:
    def __init__(self, quiz: Quiz) -> None:
        self._quiz = quiz
        self._completed = False

    def student_view(self) -> dict:
        """No key, explanation, source excerpt, or grading metadata before reveal."""
        return {"completed": self._completed,
                "questions": [{"prompt": q.prompt, "options": list(q.options)}
                              for q in self._quiz.questions]}

    def _feedback(self, answers: Sequence[int] | None) -> dict:
        feedback = []
        for i, q in enumerate(self._quiz.questions):
            feedback.append({"question": i, "correct_index": q.correct_index,
                             "chosen_index": None if answers is None else answers[i],
                             "correct": None if answers is None else answers[i] == q.correct_index,
                             "explanation": q.explanation, "evidence": q.evidence.public()})
        return {"score": None if answers is None else sum(item["correct"] for item in feedback),
                "total": len(feedback), "feedback": feedback}

    def submit(self, answers: Sequence[int]) -> dict:
        if self._completed:
            raise RuntimeError("quiz is already complete")
        if len(answers) != len(self._quiz.questions) or any(
            isinstance(a, bool) or not isinstance(a, int) or not 0 <= a < 4 for a in answers
        ):
            raise ValueError("submit exactly one valid option index per question")
        self._completed = True
        return self._feedback(answers)

    def reveal(self) -> dict:
        if self._completed:
            raise RuntimeError("quiz is already complete")
        self._completed = True
        return self._feedback(None)
