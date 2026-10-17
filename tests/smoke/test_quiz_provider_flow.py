"""Synthetic provider/model/session wiring; no live service or browser."""
import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from app.quiz import QuizSession
from app.quiz_generation import QuizRequest
from app.quiz_model import VisionQuizGenerator
from app.quiz_provider import QuizProvider

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize("image_only", [False, True])
def test_generated_quiz_conceals_then_reveals_original_evidence(tmp_path, image_only):
    image = tmp_path / "synthetic-page.png"
    Image.new("RGB", (40, 40), "white").save(image)
    source = SimpleNamespace(doc_id="chosen", filename="Lecture.pdf", page_no=2,
                             excerpt="" if image_only else "BM25 retrieves by term matching",
                             image_path=str(image), image_verified=True)

    class Retriever:
        def retrieve(self, query, *, doc_ids, top_k, include_images):
            assert doc_ids == ("chosen",)
            assert include_images
            return SimpleNamespace(evidence=SimpleNamespace(items=[source]))

    class Model:
        def chat(self, messages):
            serialized = json.dumps(messages)
            assert "data:image/" in serialized
            assert "E1" in serialized
            return json.dumps({"questions": [{
                "prompt": "Which method retrieves using term matching?",
                "options": ["BM25", "OCR", "ASR", "RNN"],
                "correct_index": 0,
                "explanation": "BM25 retrieves by term matching.",
                "evidence_id": "E1",
            }]})

    provider = QuizProvider(Retriever(), VisionQuizGenerator(Model()))
    request = QuizRequest(("chosen",), None, 1, 42)
    quiz = provider(request)
    assert quiz == provider(request)  # identical fake output, not live determinism
    session = QuizSession(quiz)
    initial = session.student_view()
    assert not initial["completed"]
    assert set(initial["questions"][0]) == {"prompt", "options"}
    assert "explanation" not in json.dumps(initial)
    assert str(image) not in json.dumps(initial)
    with pytest.raises(ValueError):
        session.submit([True])
    assert not session.student_view()["completed"]
    correct = initial["questions"][0]["options"].index("BM25")
    result = session.submit([correct])
    assert result["score"] == result["total"] == 1
    assert result["feedback"][0]["evidence"]["document"] == "Lecture.pdf"
    assert result["feedback"][0]["evidence"]["page"] == 2
    assert result["feedback"][0]["evidence"]["image_path"] == str(image)
    assert result["feedback"][0]["explanation"]
    assert QuizSession(quiz).reveal()["score"] is None
    with pytest.raises(RuntimeError):
        session.submit([correct])


@pytest.mark.parametrize("image_count,layout", [
    (5, "image-text"), (8, "image-text"),
    (5, "image-only"), (8, "image-only"), (5, "mixed"),
])
def test_image_budget_reaches_real_generator_with_ranked_evidence(tmp_path, image_count, layout):
    sources = []
    for page in range(1, image_count + 1):
        image = tmp_path / f"synthetic-{page}.png"
        Image.new("RGB", (8, 8), (page, 0, 0)).save(image)
        sources.append(SimpleNamespace(
            doc_id="chosen", filename="Lecture.pdf", page_no=page,
            excerpt="" if layout == "image-only" else f"BM25 ranked evidence {page}",
            image_path=str(image), image_verified=True,
        ))
    if layout == "mixed":
        def text(page):
            return SimpleNamespace(doc_id="chosen", filename="Lecture.pdf", page_no=page,
                                   excerpt=f"BM25 text-only evidence {page}", image_path=None,
                                   image_verified=False)

        sources = [text(101), *sources[:2], text(102), *sources[2:], text(103)]
    pages = [101, 1, 2, 102, 3, 4, 103] if layout == "mixed" else [1, 2, 3, 4]
    selected = [source for source in sources if source.page_no in pages]
    calls = []

    class Retriever:
        def retrieve(self, query, *, doc_ids, top_k, include_images):
            assert doc_ids == ("chosen",)
            assert top_k == 8
            assert include_images is True
            return SimpleNamespace(evidence=SimpleNamespace(items=sources))

    class Model:
        def chat(self, messages):
            calls.append(messages)
            payload = json.loads(messages[1]["content"])
            assert payload == {"count": 1, "topic": None, "evidence": [
                {"evidence_id": f"E{index}", "text": source.excerpt or None}
                for index, source in enumerate(selected, 1)
            ]}
            image_messages = messages[2:]
            assert len(image_messages) == 4
            expected_images = [(index, source) for index, source in enumerate(selected, 1)
                               if source.image_path is not None]
            for message, (index, source) in zip(image_messages, expected_images):
                assert json.loads(message["content"][0]["text"]) == {"evidence_id": f"E{index}"}
                url = message["content"][1]["image_url"]["url"]
                assert url.startswith("data:image/png;base64,")
                assert base64.b64decode(url.split(",", 1)[1]) == Path(source.image_path).read_bytes()
            return json.dumps({"questions": [{
                "prompt": "Which method matches terms?",
                "options": ["BM25", "OCR", "ASR", "RNN"], "correct_index": 0,
                "explanation": "BM25 matches terms.", "evidence_id": f"E{len(selected)}",
            }]})

    provider = QuizProvider(Retriever(), VisionQuizGenerator(Model()))
    request = QuizRequest(("chosen",), None, 1, 42)
    quiz = provider(request)
    assert quiz == provider(request)
    assert len(calls) == 2
    assert calls[0] == calls[1]
    cited = selected[-1]
    ref = quiz.questions[0].evidence
    assert (ref.doc_id, ref.document, ref.page, ref.excerpt, ref.image_path) == (
        cited.doc_id, cited.filename, cited.page_no, cited.excerpt or None, cited.image_path,
    )


@pytest.mark.parametrize("missing_image", [False, True])
def test_merged_retrieval_service_to_quiz_session(tmp_path, missing_image):
    """Real parse/BM25/fusion/evidence with fake model; no class services."""
    import pymupdf
    from app.documents import DocumentStore
    from app.indexing import CorpusIndexer
    from app.parse import parse_file
    from app.retrieval import RetrievalService
    from app.evidence import EvidencePack

    store = DocumentStore(root=tmp_path / "data")
    records = []
    for folder, text in [
        ("chosen", "BM25 matches terms. Precision measures relevant retrieved items."),
        ("foreign", "BM25 foreign document must never enter the selected quiz."),
    ]:
        directory = tmp_path / folder
        directory.mkdir()
        pdf = directory / "Lecture.pdf"  # same display name, distinct identities
        document = pymupdf.open()
        document.new_page().insert_text((72, 72), text)
        document.save(str(pdf))
        document.close()
        record = store.add(pdf)
        parse_file(store.corpus_path(record.doc_id), record.doc_id,
                   store.artifacts_dir / record.doc_id)
        records.append(record)
    chosen, foreign = records
    indexer = CorpusIndexer(store, persist_dir=tmp_path / "vectors")
    for record in records:
        stats = indexer.index_document(record.doc_id)
        assert not stats.errors
    service = RetrievalService(keyword=indexer.keyword, store=store)
    if missing_image:
        manifest = json.loads((store.artifacts_dir / chosen.doc_id / "manifest.json").read_text(encoding="utf-8"))
        Path(manifest["pages"][0]["image_path"]).unlink()
    outcome = service.retrieve("BM25", doc_ids=(chosen.doc_id,), include_images=True)
    assert isinstance(outcome.evidence, EvidencePack)
    assert outcome.evidence.items
    assert {item.doc_id for item in outcome.evidence.items} == {chosen.doc_id}
    source = outcome.evidence.items[0]
    assert source.image_verified is (not missing_image)
    assert outcome.stages["semantic"] == outcome.stages["rerank"] == "skipped"

    class Model:
        def chat(self, messages):
            payload = json.loads(messages[1]["content"])
            assert payload["evidence"][0]["text"].startswith("BM25 matches terms")
            assert "foreign document" not in json.dumps(messages)
            assert ("data:image/" in json.dumps(messages)) is (not missing_image)
            return json.dumps({"questions": [{
                "prompt": "Which method matches terms?",
                "options": ["BM25", "OCR", "ASR", "RNN"],
                "correct_index": 0, "explanation": "BM25 matches terms.",
                "evidence_id": "E1",
            }]})

    provider = QuizProvider(service, VisionQuizGenerator(Model()))
    quiz = provider(QuizRequest((chosen.doc_id,), "BM25", 1, 42))
    assert quiz.questions[0].evidence.doc_id == chosen.doc_id
    assert quiz.questions[0].evidence.doc_id != foreign.doc_id
    session = QuizSession(quiz)
    view = session.student_view()
    assert set(view["questions"][0]) == {"prompt", "options"}
    result = session.submit([view["questions"][0]["options"].index("BM25")])
    assert result["score"] == 1
    evidence = result["feedback"][0]["evidence"]
    assert evidence["document"] == "Lecture.pdf" and evidence["page"] == 1
    assert evidence["excerpt"].startswith("BM25 matches terms")
    assert bool(evidence["image_path"]) is (not missing_image)
