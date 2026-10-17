"""Offline tests for the Issue #6 callable provider boundary."""
from types import SimpleNamespace

import pytest

from app.quiz import Question, QuizSession
from app.quiz_generation import QuizRequest
from app.quiz_integration import evidence_from_pack
from app.quiz_provider import QuizProvider, QuizProviderError

pytestmark = pytest.mark.unit


def item(doc_id="chosen", filename="Lecture.pdf", **updates):
    fields = dict(doc_id=doc_id, filename=filename, page_no=1,
                  excerpt="BM25 matches terms", image_path=None, image_verified=False)
    fields.update(updates)
    return SimpleNamespace(**fields)


class Retriever:
    def __init__(self, items):
        self.items = items
        self.calls = []

    def retrieve(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return SimpleNamespace(evidence=SimpleNamespace(items=self.items))


def generator(evidence, topic, count):
    return [Question("Which retrieval method?", ("BM25", "OCR", "ASR", "RNN"),
                     0, "BM25 matches terms", evidence[0])] * count


def test_provider_scopes_both_retrieval_and_generation_by_document_id():
    retriever = Retriever([item("foreign"), item()])
    seen = []

    def capture(evidence, topic, count):
        seen.extend(evidence)
        return generator(evidence, topic, count)

    quiz = QuizProvider(retriever, capture)(QuizRequest(("chosen",), None, 1, 42))
    assert [ref.doc_id for ref in seen] == ["chosen"]
    assert quiz.questions[0].evidence.doc_id == "chosen"
    assert retriever.calls[0][1] == {
        "doc_ids": ("chosen",), "top_k": 8, "include_images": True
    }
    assert "explanation" not in str(QuizSession(quiz).student_view())


def test_topic_is_forwarded_and_literal_evidence_filter_is_preserved():
    retriever = Retriever([item(excerpt="BM25 matches terms"), item(excerpt="OCR reads text")])
    provider = QuizProvider(retriever, generator)
    quiz = provider(QuizRequest(("chosen",), "BM25", 1, 42))
    assert retriever.calls[0][0] == "BM25"
    assert quiz.questions[0].evidence.excerpt == "BM25 matches terms"
    with pytest.raises(QuizProviderError, match="supporting evidence"):
        provider(QuizRequest(("chosen",), "missing topic", 1, 42))


@pytest.mark.parametrize("items", [[], [item("foreign")], [item(doc_id=None)],
                                     [item(page_no=True)], [item(excerpt="")]])
def test_empty_or_invalid_evidence_never_calls_model(items):
    def forbidden(*args):
        pytest.fail("model must not run without supporting selected evidence")

    with pytest.raises(QuizProviderError, match="supporting evidence"):
        QuizProvider(Retriever(items), forbidden)(QuizRequest(("chosen",), None, 1, 0))


@pytest.mark.parametrize("stage", ["retrieval", "generation"])
def test_failures_are_generic_and_do_not_expose_exception_text(stage):
    marker = "SYNTHETIC_PRIVATE_VALUE"

    class BrokenRetriever(Retriever):
        def retrieve(self, *args, **kwargs):
            raise RuntimeError(marker)

    def broken_generator(*args):
        raise ValueError(marker)

    retriever = BrokenRetriever([]) if stage == "retrieval" else Retriever([item()])
    gen = broken_generator if stage == "generation" else generator
    with pytest.raises(QuizProviderError) as caught:
        QuizProvider(retriever, gen)(QuizRequest(("chosen",), None, 1, 0))
    assert marker not in str(caught.value)
    assert caught.value.__suppress_context__


def test_model_cannot_introduce_unselected_evidence():
    from app.quiz import EvidenceRef

    def fabricated(evidence, topic, count):
        return [Question("Which?", ("A", "B", "C", "D"), 0, "Explanation",
                         EvidenceRef("Lecture.pdf", 1, "BM25 matches terms", doc_id="foreign"))]

    with pytest.raises(QuizProviderError):
        QuizProvider(Retriever([item()]), fabricated)(QuizRequest(("chosen",), None, 1, 0))


def test_count_bound_prevents_retrieval_and_request_types_fail_closed():
    retriever = Retriever([item()])
    provider = QuizProvider(retriever, generator)
    for invalid in [None, "request", QuizRequest(("chosen",), None, 6, 0),
                    QuizRequest(("chosen",), 123, 1, 0)]:
        with pytest.raises(QuizProviderError):
            provider(invalid)
    assert not retriever.calls


def test_same_generated_questions_and_seed_preserve_answer_key():
    provider = QuizProvider(Retriever([item()]), generator)
    request = QuizRequest(("chosen",), None, 1, 42)
    assert provider(request) == provider(request)


def test_unverified_images_never_reach_generator():
    seen = []

    def capture(evidence, topic, count):
        seen.extend(evidence)
        return generator(evidence, topic, count)

    QuizProvider(Retriever([item(image_path="untrusted.png", image_verified=False)]),
                 capture)(QuizRequest(("chosen",), None, 1, 42))
    assert seen[0].image_path is None


def ranked_items(image_count, layout):
    sources = [item(page_no=page, excerpt="" if layout == "image-only" else
                    f"BM25 ranked evidence {page}",
                    image_path=f"synthetic-{page}.png", image_verified=True)
               for page in range(1, image_count + 1)]
    if layout == "mixed":
        sources = [item(page_no=101), *sources[:2], item(page_no=102),
                   *sources[2:], item(page_no=103)]
    return sources


@pytest.mark.parametrize("image_count,layout", [
    (5, "image-text"), (8, "image-text"),
    (5, "image-only"), (8, "image-only"), (5, "mixed"),
])
def test_image_budget_keeps_ranked_refs_and_all_text(image_count, layout):
    sources = ranked_items(image_count, layout)
    refs = evidence_from_pack(SimpleNamespace(items=sources))
    pages = [101, 1, 2, 102, 3, 4, 103] if layout == "mixed" else [1, 2, 3, 4]
    expected = tuple(ref for ref in refs if ref.page in pages)
    retriever = Retriever(sources)
    seen = []

    def capture(evidence, topic, count):
        seen.append(evidence)
        return generator(evidence[-1:], topic, count)

    provider = QuizProvider(retriever, capture)
    request = QuizRequest(("chosen",), None, 1, 42)
    quiz = provider(request)
    assert quiz == provider(request)
    assert seen == [expected, expected]
    assert [ref.page for ref in seen[0]] == pages
    assert quiz.questions[0].evidence == expected[-1]
    assert retriever.calls == [("key concepts definitions methods examples", {
        "doc_ids": ("chosen",), "top_k": 8, "include_images": True,
    })] * 2


def test_image_budget_is_applied_after_document_and_topic_filters():
    sources = [item("foreign", page_no=90, image_path="foreign.png", image_verified=True),
               item(page_no=91, excerpt="OCR only", image_path="off-topic.png",
                    image_verified=True), *ranked_items(6, "image-text")]
    seen = []

    def capture(evidence, topic, count):
        seen.extend(evidence)
        return generator(evidence, topic, count)

    QuizProvider(Retriever(sources), capture)(QuizRequest(("chosen",), "BM25", 1, 42))
    assert [ref.page for ref in seen] == [1, 2, 3, 4]
    assert [ref.image_path for ref in seen] == [f"synthetic-{page}.png" for page in range(1, 5)]


@pytest.mark.parametrize("image_count", [5, 8])
def test_image_budget_excludes_dropped_refs_from_evidence_validation(image_count):
    sources = ranked_items(image_count, "image-text")
    omitted = evidence_from_pack(SimpleNamespace(items=sources))[-1]

    def cite_omitted(evidence, topic, count):
        return [Question("Which?", ("A", "B", "C", "D"), 0, "Explanation", omitted)]

    with pytest.raises(QuizProviderError, match="valid evidence-backed quiz"):
        QuizProvider(Retriever(sources), cite_omitted)(QuizRequest(("chosen",), None, 1, 42))
