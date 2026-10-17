"""Real parse/index/retrieval/provider replay with a changing offline model."""
import json
from pathlib import Path

import pymupdf
import pytest

from app.documents import DocumentStore
from app.indexing import CorpusIndexer
from app.parse import parse_file
from app.quiz import QuizSession
from app.quiz_generation import QuizRequest
from app.quiz_model import VisionQuizGenerator
from app.quiz_provider import QuizProvider, QuizProviderError
from app.quiz_replay import QuizReplayProvider
from app.retrieval import RetrievalService

pytestmark = pytest.mark.smoke


def test_real_retrieval_replays_changing_model_with_independent_sessions(tmp_path):
    store = DocumentStore(tmp_path / "data")
    records = []
    for name, text in [("selected", "BM25 matches terms."),
                       ("foreign", "BM25 foreign document must not enter this quiz.")]:
        folder = tmp_path / name
        folder.mkdir()
        source = folder / "Lecture.pdf"  # display names deliberately collide
        with pymupdf.open() as pdf:
            pdf.new_page().insert_text((72, 72), text)
            pdf.save(str(source))
        record = store.add(source)
        parse_file(store.corpus_path(record.doc_id), record.doc_id,
                   store.artifacts_dir / record.doc_id)
        records.append(record)
    selected, foreign = records
    indexer = CorpusIndexer(store, persist_dir=tmp_path / "vectors")
    for record in records:
        assert not indexer.index_document(record.doc_id).errors
    service = RetrievalService(keyword=indexer.keyword, store=store)
    calls = []

    class Model:
        def chat(self, messages):
            calls.append(messages)
            assert "foreign document" not in json.dumps(messages)
            assert "data:image/" in json.dumps(messages)
            return json.dumps({"questions": [{
                "prompt": f"Which method matches terms? Variant {len(calls)}",
                "options": ["BM25", "OCR", "ASR", "RNN"],
                "correct_index": 0, "explanation": "BM25 matches terms.",
                "evidence_id": "E1",
            }]})

    provider = QuizProvider(service, VisionQuizGenerator(Model()))
    replay = QuizReplayProvider(store, provider)
    request = QuizRequest((selected.doc_id,), "BM25", 1, 42)
    first = replay(request)
    assert replay(request) is first
    assert len(calls) == 1
    assert first.questions[0].evidence.doc_id == selected.doc_id
    assert first.questions[0].evidence.doc_id != foreign.doc_id
    first_session, second_session = QuizSession(first), QuizSession(replay(request))
    initial = first_session.student_view()
    assert set(initial["questions"][0]) == {"prompt", "options"}
    assert "explanation" not in json.dumps(initial)
    assert "image_path" not in json.dumps(initial)
    with pytest.raises(ValueError):
        first_session.submit([True])
    assert not first_session.student_view()["completed"]
    result = first_session.submit(first.answer_key)
    assert result["score"] == result["total"] == 1
    evidence = result["feedback"][0]["evidence"]
    assert evidence["document"] == "Lecture.pdf" and evidence["page"] == 1
    assert evidence["excerpt"].startswith("BM25 matches terms")
    assert Path(evidence["image_path"]).is_file()
    assert not second_session.student_view()["completed"]
    assert second_session.reveal()["feedback"][0]["correct_index"] == first.answer_key[0]
    replay.clear()
    assert replay(request) != first
    assert len(calls) == 2
    store.remove(selected.doc_id)
    with pytest.raises(QuizProviderError):
        replay(request)
    assert len(calls) == 2
