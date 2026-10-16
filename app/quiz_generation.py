"""Evidence-scoped, offline quiz generation contract.

The caller supplies verified evidence; a production model adapter can later
implement ``QuestionGenerator`` without coupling this module to PR #15 or #14.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from app.quiz import EvidenceRef, Question, Quiz, make_quiz

QuestionGenerator = Callable[[tuple[EvidenceRef, ...], str | None, int], Sequence[Question]]


@dataclass(frozen=True)
class QuizRequest:
    materials: tuple[str, ...]          # unique document IDs (EvidenceRef.doc_id), not filenames
    topic: str | None
    count: int
    seed: int

    def __post_init__(self) -> None:
        if not self.materials or any(not m.strip() for m in self.materials):
            raise ValueError("select at least one document")
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count < 1:
            raise ValueError("question count must be positive")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")


def generate_quiz(
    request: QuizRequest, evidence: Sequence[EvidenceRef], generator: QuestionGenerator
) -> Quiz:
    """Reject unsupported sources before making a student-visible quiz.

    Materials select evidence by unique ``doc_id`` — never by filename, which
    is display metadata only and is not unique across the document store.
    Evidence without a ``doc_id`` can never match a document selection.
    """
    topic = (request.topic or "").strip() or None
    scoped = tuple(e for e in evidence if e.doc_id in request.materials and
                   (topic is None or topic.casefold() in (e.excerpt or "").casefold()))
    if not scoped:
        raise ValueError("no supporting evidence for the selected document/topic")
    questions = tuple(generator(scoped, topic, request.count))
    if len(questions) != request.count:
        raise ValueError("generated question count does not match request")
    for question in questions:
        if not isinstance(question, Question):
            raise ValueError("generator returned an invalid question")
        if question.evidence not in scoped:
            raise ValueError("question cites evidence not supplied for this request")
    return make_quiz(questions, request.seed)
