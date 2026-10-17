"""Callable quiz provider; retrieval and model dependencies are injected.

The production retriever must supply manifest-vetted evidence. This module
never imports an unmerged retrieval/UI branch or constructs fake fallbacks.
"""
from __future__ import annotations

from app.quiz import Quiz
from app.quiz_generation import QuizRequest, generate_quiz
from app.quiz_integration import evidence_from_pack

MAX_QUESTIONS = 5
MAX_EVIDENCE_IMAGES = 4
DEFAULT_QUERY = "key concepts definitions methods examples"


class QuizProviderError(ValueError):
    """Safe caller-facing error; never includes upstream exception text."""


class QuizProvider:
    """Implement ``provider(QuizRequest) -> Quiz`` for the server-side UI.

    ``retriever.retrieve(query, doc_ids=..., top_k=8, include_images=True)``
    returns an outcome with an ``evidence`` pack. ``generator`` implements
    the existing QuestionGenerator protocol. No network calls at construction.
    """

    def __init__(self, retriever, generator):
        self._retriever = retriever
        self._generator = generator

    def __call__(self, request: QuizRequest) -> Quiz:
        if not isinstance(request, QuizRequest):
            raise QuizProviderError("Invalid quiz request.")
        if (type(request.count) is not int or not 1 <= request.count <= MAX_QUESTIONS
                or type(request.seed) is not int
                or not isinstance(request.materials, tuple)
                or not request.materials
                or any(not isinstance(m, str) or not m.strip() for m in request.materials)
                or (request.topic is not None and not isinstance(request.topic, str))):
            raise QuizProviderError("Choose documents and between one and five questions.")
        topic = (request.topic or "").strip() or None
        if topic and len(topic) > 2000:
            raise QuizProviderError("Topic must be at most 2,000 characters.")
        try:
            outcome = self._retriever.retrieve(
                topic or DEFAULT_QUERY, doc_ids=request.materials,
                top_k=8, include_images=True,
            )
            evidence = evidence_from_pack(outcome.evidence)
        except Exception:
            raise QuizProviderError("Quiz evidence retrieval is unavailable.") from None
        scoped = tuple(e for e in evidence if e.doc_id in request.materials and
                       (topic is None or topic.casefold() in (e.excerpt or "").casefold()))
        if not scoped:
            raise QuizProviderError("No supporting evidence for the selected documents/topic.")
        # Preserve retrieval rank and whole references within the model image budget.
        selected = []
        images = 0
        for ref in scoped:
            if ref.image_path is not None:
                if images >= MAX_EVIDENCE_IMAGES:
                    continue
                images += 1
            selected.append(ref)
        try:
            return generate_quiz(request, tuple(selected), self._generator)
        except Exception:
            raise QuizProviderError("Unable to generate a valid evidence-backed quiz.") from None
