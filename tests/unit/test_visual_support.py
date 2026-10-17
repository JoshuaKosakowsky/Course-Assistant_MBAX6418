"""Image-only Q&A must have authentic evidence and explicit visual support."""
import json

import pytest
from PIL import Image

from app.evidence import EvidenceItem, EvidencePack
from app.qna import QnaOrchestrator
from app.qna_schema import Source, parse_answer
from app.source_check import verify_citations
from app.visual_support import VisualSupportVerifier

pytestmark = pytest.mark.unit


@pytest.fixture
def source(tmp_path):
    image = tmp_path / "page.png"
    Image.new("RGB", (32, 32), "red").save(image)
    return Source("diagram.pdf", 1, None, str(image))


def payload(source):
    return json.dumps({"answer": "The square is red.", "grounded": True,
                       "sources": [vars(source)]})


class Model:
    def __init__(self, result):
        self.result = result

    def chat(self, messages):
        return self.result


def test_image_only_citation_requires_matching_original(source):
    parsed = parse_answer(payload(source))
    assert parsed.sources[0].excerpt is None
    assert verify_citations(parsed.sources, (source,))[0].valid
    foreign = Source(source.document, source.page, None, "foreign.png")
    assert not verify_citations((foreign,), (source,))[0].valid
    assert not verify_citations((source,), (Source(source.document, 1, None),))[0].valid


def test_image_only_answer_requires_explicit_support(source):
    client = Model(payload(source))
    assert not QnaOrchestrator(client).answer("What color?", (source,)).grounded
    for verdict in (False, True):
        result = QnaOrchestrator(client, visual_verifier=lambda answer, ref: verdict).answer(
            "What color?", (source,))
        assert result.grounded is verdict
        if verdict:
            assert result.sources[0].excerpt is None


@pytest.mark.parametrize("raw, expected", [
    ('{"supported":true}', True), ('{"supported":false}', False),
    ('{"supported":"true"}', False), ('{"supported":1}', False),
    ('not json', False), ('{"supported":true,"extra":1}', False),
])
def test_visual_verdict_is_strict(source, raw, expected):
    assert VisualSupportVerifier(Model(raw))("The square is red.", source) is expected


def test_unavailable_image_never_reaches_verifier_model():
    class Never:
        def chat(self, messages):
            pytest.fail("must not send a missing image")
    assert not VisualSupportVerifier(Never())("Red", Source("d", 1, None, "missing.png"))


def test_evidence_adapter_retains_only_verified_image_only_pages(source):
    from dataclasses import replace
    item = EvidenceItem(doc_id="doc", filename=source.document, page_no=1, excerpt=None,
                        image_path=source.image_path, chunk_id="c", final_rank=1,
                        rerank_score=None, fused_rank=1, fused_score=0.1, methods=(),
                        evidence_ranks=(), image_included=False, image_source="artifact",
                        image_verified=True)
    assert EvidencePack("q", [item]).to_sources()[0].excerpt is None
    assert EvidencePack("q", [replace(item, image_verified=False)]).to_sources() == []


def test_visual_claim_absent_from_slide_heading_requires_explicit_image_verdict(source):
    heading = Source(source.document, source.page, "Color diagram", source.image_path)
    result = json.dumps({"answer": "The square is not green.", "grounded": True,
                         "sources": [vars(heading)]})
    # Negation is not in the extracted heading; the original image can still
    # support the claim. Text-only checks cannot establish visual entailment.
    assert not QnaOrchestrator(Model(result)).answer("Describe the color", (heading,)).grounded
    assert not QnaOrchestrator(Model(result), visual_verifier=lambda *args: False).answer(
        "Describe the color", (heading,)).grounded
    assert QnaOrchestrator(Model(result), visual_verifier=lambda *args: True).answer(
        "Describe the color", (heading,)).grounded
