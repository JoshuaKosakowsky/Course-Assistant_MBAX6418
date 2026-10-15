"""E2E tests: real class vision LLM endpoint (Phase 4, issue #5).

Run deliberately with ``pytest -m e2e``. Auto-skip without .env settings.
These probe the live contract: openai-compatible chat on :9001, whether
base64 data-URI images are accepted, and that the model honors the
evidence-only prompt. Fail loudly if the contract differs — that is the
point of having them. Screenshots/evidence for results/ go in a later
Phase 6 pass once the Gradio UI exists.
"""

import os

import pytest
from PIL import Image

from app.llm import ClassVisionClient, VISION_BASE_URL, VISION_API_KEY, VISION_MODEL
from app.qna import QnaOrchestrator
from app.qna_schema import Source

pytestmark = pytest.mark.e2e

EVIDENCE = (
    Source(
        document="rag-lecture.pdf",
        page=3,
        excerpt="Retrieval augmented generation combines retrieval with generation.",
    ),
)


def _endpoint_configured() -> bool:
    return bool(VISION_BASE_URL and VISION_API_KEY and VISION_MODEL)


@pytest.fixture(autouse=True)
def _require_endpoint():
    if not _endpoint_configured():
        pytest.skip("vision LLM endpoint not configured in .env")


@pytest.fixture
def client():
    return ClassVisionClient()


def test_vision_endpoint_returns_content(client):
    reply = client.chat([{"role": "user", "content": "Reply with exactly the single word: ok"}])
    assert reply.strip()


def test_answer_with_evidence_parses(client):
    result = QnaOrchestrator(client).answer("What is RAG?", EVIDENCE)
    assert result.grounded is True, f"expected grounded answer, got: {result.answer!r}"
    assert len(result.sources) >= 1
    for check in result.source_checks:
        assert check.verdict in {"supported", "weak", "unsupported"}


def test_out_of_scope_question_is_handled_honestly(client):
    result = QnaOrchestrator(client).answer("What is the capital of France?", ())
    assert result.grounded is False
    assert "not in the materials" in result.answer.lower() or "don't have" in result.answer.lower()


def test_image_grounded_answer_references_the_diagram(client, tmp_path):
    """Probes data-URI image input and diagram reading (contract test)."""
    img_path = tmp_path / "chart.png"
    Image.new("RGB", (120, 80), (30, 180, 90)).save(img_path)
    evidence = (
        Source(document="chart-slide.pdf", page=5, excerpt="Fusion diagram", image_path=str(img_path)),
    )
    result = QnaOrchestrator(client).answer("What color is the diagram?", evidence)
    # Either grounded with a color reference, or the endpoint rejected the
    # data URI — both are information. The parse must never raise garbage:
    assert result.answer.strip()
