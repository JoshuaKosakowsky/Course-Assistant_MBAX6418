"""Student flows over real offline retrieval, with an injected model."""
import json
from types import SimpleNamespace

import pytest

from app import config
from app.documents import DocumentStore
from app.indexing import CorpusIndexer
from app.qna import QnaOrchestrator
from app.retrieval import RetrievalService
from app.study_ui import QnaController, production_study_services
from app.ui import add_file, build_ui

pytestmark = pytest.mark.smoke


@pytest.fixture
def library(tmp_path, tiny_pdf):
    store = DocumentStore(tmp_path / "data")
    indexer = CorpusIndexer(store, persist_dir=tmp_path / "index")
    add_file(store, str(tiny_pdf), indexer)
    return store, indexer, store.list()[0]


class EvidenceModel:
    def __init__(self, filename):
        self.filename = filename
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        assert "image_url" in str(messages)
        return json.dumps({"answer": "Retrieval augmented generation quiz prep slides.",
                           "grounded": True, "sources": [{"document": self.filename,
                           "page": 1, "excerpt": "retrieval augmented generation quiz prep slides"}]})


def test_question_retrieves_and_displays_original(library):
    store, indexer, doc = library
    model = EvidenceModel(doc.filename)
    controller = QnaController(store, RetrievalService(indexer.keyword, store=store),
                               QnaOrchestrator(model))
    status, html, images = controller.ask([doc.doc_id], "retrieval augmented generation")
    assert model.calls == 1
    assert "checked" in status
    assert doc.filename in html and "Page/slide 1" in html
    assert "Retrieval augmented generation" in html
    assert images and images[0][0].endswith("page-001.png")
    assert "Limited retrieval" in status


def test_no_evidence_does_not_call_model(library):
    store, indexer, doc = library
    model = EvidenceModel(doc.filename)
    controller = QnaController(store, RetrievalService(indexer.keyword, store=store),
                               QnaOrchestrator(model))
    status, html, images = controller.ask([doc.doc_id], "nonexistent_asteroid_xyz")
    assert model.calls == 0 and images == []
    assert "No supporting" in status and "can't answer" in html.replace("&#x27;", "'")


def test_selected_document_deleted_or_foreign_is_rejected(library):
    store, indexer, doc = library
    model = EvidenceModel(doc.filename)
    controller = QnaController(store, RetrievalService(indexer.keyword, store=store),
                               QnaOrchestrator(model))
    assert "no longer available" in controller.ask(["foreign"], "retrieval")[0]
    store.remove(doc.doc_id)
    assert "no longer available" in controller.ask([doc.doc_id], "retrieval")[0]
    assert model.calls == 0


def test_errors_and_unavailable_models_are_safe(library):
    store, _, doc = library
    assert "unavailable" in QnaController(store).ask([doc.doc_id], "retrieval")[0]

    class Broken:
        def retrieve(self, *args, **kwargs):
            raise RuntimeError("PRIVATE_TOKEN_DO_NOT_DISPLAY")

    result = QnaController(store, Broken(), object()).ask([doc.doc_id], "retrieval")
    assert "PRIVATE_TOKEN" not in str(result)
    assert result[1:] == ("", [])


def test_source_text_and_answers_are_escaped(library):
    store, indexer, doc = library

    class Model:
        def answer(self, question, evidence):
            return SimpleNamespace(grounded=True, answer='<script>alert("x")</script>',
                                   sources=(evidence[0],))

    controller = QnaController(store, RetrievalService(indexer.keyword, store=store), Model())
    _, html, _ = controller.ask([doc.doc_id], "retrieval")
    assert "<script>" not in html and "&lt;script&gt;" in html


def test_production_services_bind_after_reconcile(library, monkeypatch):
    import app.ui as ui
    store, indexer, doc = library
    previous = indexer.keyword
    seen = []

    def factory(bound_indexer):
        assert bound_indexer.keyword is not previous
        assert bound_indexer.keyword.search("retrieval")
        seen.append(bound_indexer)
        return RetrievalService(bound_indexer.keyword, store=store), object(), lambda request: None

    monkeypatch.setattr(ui, "production_study_services", factory)
    demo = build_ui(store, indexer, connect_services=True)
    assert seen == [indexer]
    assert any(c.get("props", {}).get("label") == "Your question"
               for c in demo.config["components"])


def test_missing_configuration_never_constructs_model(library, monkeypatch):
    import app.quiz_model as model
    _, indexer, _ = library
    monkeypatch.setattr(config, "VISION_API_KEY", "replace-with-class-key")
    monkeypatch.setattr(model, "QuizVisionClient", lambda: pytest.fail("must not construct model"))
    retrieval, qna, quiz = production_study_services(indexer)
    assert retrieval.keyword is indexer.keyword and qna is quiz is None


def test_production_quiz_provider_is_connected(library, monkeypatch):
    import app.quiz_model as model
    from app.quiz_generation import QuizRequest
    from app.quiz_ui import QuizController
    store, indexer, doc = library

    class QuizModel:
        def chat(self, messages):
            return json.dumps({"questions": [{"prompt": "Which approach is named?",
                "options": ["Retrieval augmented generation", "Decision tree", "Naive Bayes", "Clustering"],
                "correct_index": 0, "explanation": "The source names retrieval augmented generation.",
                "evidence_id": "E1"}]})

    monkeypatch.setattr(config, "VISION_BASE_URL", "https://model.invalid/v1")
    monkeypatch.setattr(config, "VISION_API_KEY", "synthetic-test-key")
    monkeypatch.setattr(config, "VISION_MODEL", "synthetic-model")
    monkeypatch.setattr(model, "QuizVisionClient", QuizModel)
    # Keep this wiring test strictly offline while exercising the real provider.
    import app.retrieval as retrieval
    monkeypatch.setattr(retrieval, "build_retrieval_service",
                        lambda i: RetrievalService(i.keyword, store=store))
    service, qna, provider = production_study_services(indexer)
    assert service and qna and provider
    request = QuizRequest((doc.doc_id,), "retrieval", 1, 42)
    assert provider(request) is provider(request)
    controller = QuizController(provider, store)
    browser = SimpleNamespace(session_hash="student")
    generated = controller.generate([doc.doc_id], "retrieval", 1, 42, browser)
    assert "Quiz ready" in generated[0]
    assert "The source names" not in str(generated)
    score, feedback, images = controller.finish(True, [None] * 5, browser)
    assert "without grading" in score and doc.filename in feedback and images


def test_image_only_evidence_reaches_model_and_source_gallery(library):
    from dataclasses import replace
    from app.evidence import EvidencePack
    from app.visual_support import VisualSupportVerifier
    store, indexer, doc = library
    outcome = RetrievalService(indexer.keyword, store=store).retrieve("retrieval")
    item = replace(outcome.evidence.items[0], excerpt=None)

    class VisualRetrieval:
        def retrieve(self, *args, **kwargs):
            assert kwargs["doc_ids"] == (doc.doc_id,)
            return SimpleNamespace(evidence=EvidencePack("q", [item]), stages={"visual": "ok"})

    class VisualModel:
        def __init__(self):
            self.calls = 0

        def chat(self, messages):
            self.calls += 1
            assert "image_url" in str(messages)
            if self.calls == 2:
                return '{"supported":true}'
            return json.dumps({"answer": "The slide discusses retrieval.", "grounded": True,
                "sources": [{"document": doc.filename, "page": item.page_no,
                             "excerpt": None, "image_path": item.image_path}]})

    model = VisualModel()
    controller = QnaController(store, VisualRetrieval(), QnaOrchestrator(
        model, visual_verifier=VisualSupportVerifier(model)))
    status, html, images = controller.ask([doc.doc_id], "Explain the picture")
    assert "checked" in status and model.calls == 2
    assert "Visual evidence" in html and images
    assert images[0][0] == item.image_path
