"""Opt-in real quiz-model checks using generated PDFs, never course materials.

Run explicitly with RUN_LIVE_QUIZ=1 and pytest -m e2e (see quiz contract).
Only the vision model is remote here. Parsing, BM25, evidence validation,
provider, and session are real; semantic/visual indexes and reranking are
intentionally disabled so their configuration cannot mask model compatibility.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pymupdf
import pytest
import requests

from app import config
from app.documents import DocumentStore
from app.indexing import CorpusIndexer
from app.parse import parse_file
from app.quiz import QuizSession
from app.quiz_generation import QuizRequest
from app.quiz_model import QuizModelError, QuizVisionClient, VisionQuizGenerator
from app.quiz_provider import QuizProvider, QuizProviderError
from app.retrieval import RetrievalService

pytestmark = pytest.mark.e2e


def _configured(value):
    return bool(value and value.strip()) and not any(
        marker in value.casefold() for marker in ("dummy", "replace-with", "your-key", "your-model")
    )


@pytest.fixture
def live_quiz():
    if os.getenv("RUN_LIVE_QUIZ") != "1":
        pytest.skip("set RUN_LIVE_QUIZ=1 for deliberate real-model usage")
    if not all(_configured(value) for value in (
        config.VISION_BASE_URL, config.VISION_API_KEY, config.VISION_MODEL,
    )):
        pytest.skip("requires non-placeholder server-side vision settings")

    class CountingTransport:
        def __init__(self, session):
            self.session = session
            self.calls = 0
            self.image_counts = []

        def post(self, url, **kwargs):
            # Observe only counts, never persist headers, bodies, or credentials.
            self.calls += 1
            messages = kwargs["json"]["messages"]
            self.image_counts.append(sum(
                part["type"] == "image_url"
                for message in messages if isinstance(message["content"], list)
                for part in message["content"]
            ))
            return self.session.post(url, **kwargs)

    with requests.Session() as session:
        transport = CountingTransport(session)
        client = QuizVisionClient(session=transport)
        yield client, transport


@pytest.fixture
def library(tmp_path):
    store = DocumentStore(root=tmp_path / "data")
    records = []
    for directory, statement in (
        ("selected", "The Aster pilot lasts exactly 14 days."),
        ("excluded", "The Zephyr pilot lasts exactly 91 days."),
    ):
        folder = tmp_path / directory
        folder.mkdir()
        pdf = folder / "Pilot.pdf"  # duplicate display names must not defeat ID scoping
        with pymupdf.open() as document:
            document.new_page().insert_text((72, 72), statement, fontsize=18)
            document.save(str(pdf))
        record = store.add(pdf)
        parse_file(store.corpus_path(record.doc_id), record.doc_id,
                   store.artifacts_dir / record.doc_id)
        records.append(record)
    indexer = CorpusIndexer(store, persist_dir=tmp_path / "text-index")
    for record in records:
        stats = indexer.index_document(record.doc_id)
        assert not stats.errors
        assert stats.keyword_chunks > 0
    service = RetrievalService(keyword=indexer.keyword, store=store)
    return service, records[0], records[1]


@pytest.mark.parametrize("with_image", [True, False], ids=["original-image", "text-fallback"])
def test_live_provider_generates_then_scores_scoped_synthetic_quiz(live_quiz, library, with_image):
    client, transport = live_quiz
    service, chosen, foreign = library
    initial = service.retrieve("Aster", doc_ids=(chosen.doc_id,), include_images=True)
    assert initial.evidence.items
    assert initial.stages["semantic"] == initial.stages["visual"] == initial.stages["rerank"] == "skipped"
    source = initial.evidence.items[0]
    assert source.image_verified and source.image_path
    if not with_image:
        Path(source.image_path).unlink()

    quiz = QuizProvider(service, VisionQuizGenerator(client))(
        QuizRequest((chosen.doc_id,), "Aster", 1, 42)
    )
    assert transport.calls == 1
    assert transport.image_counts == [int(with_image)]
    assert len(quiz.questions) == 1
    question = quiz.questions[0]
    assert question.evidence.doc_id == chosen.doc_id != foreign.doc_id
    assert question.evidence.document == "Pilot.pdf" and question.evidence.page == 1
    assert question.evidence.excerpt is not None
    assert "Aster" in question.evidence.excerpt and "Zephyr" not in question.evidence.excerpt
    assert bool(question.evidence.image_path) is with_image
    assert len(question.options) == len(set(question.options)) == 4
    assert question.explanation.strip()
    # A narrow live factual check, not a general answer-entailment validator.
    assert re.search(r"\b(?:14|fourteen)\b", question.options[question.correct_index], re.I)
    assert "Zephyr" not in question.prompt + question.explanation

    session = QuizSession(quiz)
    view = session.student_view()
    assert set(view["questions"][0]) == {"prompt", "options"}
    assert not view["completed"]
    with pytest.raises(ValueError):
        session.submit([True])
    assert not session.student_view()["completed"]
    feedback = session.submit(list(quiz.answer_key))
    assert feedback["score"] == feedback["total"] == 1
    assert feedback["feedback"][0]["evidence"] == question.evidence.public()
    assert QuizSession(quiz).submit([(quiz.answer_key[0] + 1) % 4])["score"] == 0
    assert QuizSession(quiz).reveal()["score"] is None
    with pytest.raises(RuntimeError):
        session.reveal()
    assert transport.calls == 1  # submit/reveal never regenerate


def test_missing_selected_topic_stops_before_live_model(live_quiz, library):
    client, transport = live_quiz
    service, chosen, _ = library
    with pytest.raises(QuizProviderError, match="^No supporting evidence"):
        QuizProvider(service, VisionQuizGenerator(client))(
            QuizRequest((chosen.doc_id,), "quantum-giraffe-unknown", 1, 42)
        )
    assert transport.calls == 0


def test_live_model_http_failure_stays_generic(live_quiz):
    _, transport = live_quiz
    # Deliberately invalid synthetic credential: one rejected request, no guessing.
    client = QuizVisionClient(api_key="synthetic-invalid-live-test", session=transport)
    with pytest.raises(QuizModelError, match="^Quiz model request failed\\.$") as failure:
        client.chat([{"role": "user", "content": "Return a JSON object."}])
    assert failure.value.__suppress_context__
    assert transport.calls == 1
