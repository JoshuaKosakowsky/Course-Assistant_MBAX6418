"""Unit tests for the source-support verification (heuristic + judge)."""

from __future__ import annotations

import json

import pytest

from app.qna_schema import Source
from app.source_check import (
    WEAK_THRESHOLD,
    SUPPORTED_THRESHOLD,
    content_terms,
    overlap_fraction,
    check_sources,
    judge_sources,
    verify_citations,
)

pytestmark = pytest.mark.unit

EXCERPT = "Retrieval augmented generation combines retrieval with generation."


def test_content_terms_filters_stopwords():
    terms = content_terms("The answer is retrieval generation of the model")
    assert "the" not in terms
    assert "answer" in terms
    assert "retrieval" in terms


def test_overlap_full_and_partial():
    assert overlap_fraction("retrieval generation", EXCERPT) == 1.0
    assert overlap_fraction("retrieval pizza", EXCERPT) == 0.5
    assert overlap_fraction("pizza unicorn", EXCERPT) == 0.0


def test_overlap_stopwords_only_is_zero():
    assert overlap_fraction("the of and it", EXCERPT) == 0.0
    assert content_terms("the of and it") == set()


def test_verdict_thresholds():
    src = Source(document="d", page=1, excerpt=EXCERPT)
    supported = check_sources("retrieval augmented generation combines retrieval generation", (src,))[0]
    assert supported.overlap >= SUPPORTED_THRESHOLD
    assert supported.verdict == "supported"

    weak = check_sources("retrieval pizza unicorn robot llama", (src,))[0]
    assert weak.overlap >= WEAK_THRESHOLD
    assert weak.verdict == "weak"

    none = check_sources("pizza unicorn", (src,))[0]
    assert none.overlap == 0.0
    assert none.verdict == "unsupported"


def test_check_sources_keeps_order_and_indices():
    a = Source(document="a", page=1, excerpt="retrieval generation papers")
    b = Source(document="b", page=2, excerpt="unrelated content here")
    checks = check_sources("retrieval generation", (a, b))
    assert [c.index for c in checks] == [0, 1]
    assert checks[0].verdict == "supported"
    assert checks[1].verdict == "unsupported"


class JudgeStub:
    def __init__(self, verdicts):
        self._verdicts = list(verdicts)
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        supported = self._verdicts.pop(0)
        return json.dumps({"supported": supported, "reason": "check"})


def test_judge_source_overrides_heuristic():
    src = Source(document="d", page=1, excerpt=EXCERPT)
    # Heuristic says supported, judge says unsupported -> unsupported wins.
    judge = JudgeStub([False])
    checks = judge_sources("retrieval generation", (src,), judge)
    assert checks[0].verdict == "unsupported"
    assert judge.calls == 1


def test_judge_confirms_supported_when_heuristic_weak():
    src = Source(document="d", page=1, excerpt=EXCERPT)
    judge = JudgeStub([True])
    checks = judge_sources("retrieval", (src,), judge)  # heuristic: weak
    assert checks[0].verdict == "supported"


def test_judge_unparseable_reply_falls_back_to_heuristic():
    src = Source(document="d", page=1, excerpt=EXCERPT)
    judge = JudgeStub(["not json at all"])
    checks = judge_sources("retrieval generation", (src,), judge)
    assert checks[0].verdict == "supported"  # heuristic kept


def test_judge_sources_documents_unsupported():
    src = Source(document="d", page=1, excerpt="animal husbandry notes")
    judge = JudgeStub([False])
    checks = judge_sources("retrieval generation", (src,), judge)
    assert checks[0].verdict == "unsupported"


def test_judge_string_false_is_treated_as_false():
    src = Source(document="d", page=1, excerpt=EXCERPT)
    judge = JudgeStub(["false"])  # stub serializes this into JSON: "supported": "false"
    checks = judge_sources("retrieval generation", (src,), judge)
    assert checks[0].verdict == "unsupported"  # overrides the supported heuristic


def test_judge_invalid_string_falls_back_to_heuristic():
    src = Source(document="d", page=1, excerpt=EXCERPT)
    judge = JudgeStub(["notabool"])
    checks = judge_sources("retrieval generation", (src,), judge)
    assert checks[0].verdict == "supported"  # heuristic kept


# -- citation verification vs the supplied evidence -----------------------

def test_verify_citation_matches_evidence():
    evidence = (Source(document="rag.pdf", page=3, excerpt="RAG combines retrieval with generation."),)
    cited = (Source(document="rag.pdf", page=3, excerpt="combines retrieval with generation"),)
    checks = verify_citations(cited, evidence)
    assert checks[0].valid is True


def test_verify_citation_is_case_and_whitespace_tolerant():
    evidence = (Source(document="rag.pdf", page=3, excerpt="RAG  combines\nretrieval with generation."),)
    cited = (Source(document="rag.pdf", page=3, excerpt="combines Retrieval with generation"),)
    assert verify_citations(cited, evidence)[0].valid is True


def test_verify_citation_unknown_document_page_is_invalid():
    evidence = (Source(document="rag.pdf", page=3, excerpt="RAG combines retrieval."),)
    cited = (Source(document="other.pdf", page=9, excerpt="RAG combines retrieval."),)
    check = verify_citations(cited, evidence)[0]
    assert check.valid is False
    assert "no evidence" in check.reason


def test_verify_citation_wrong_excerpt_is_invalid():
    evidence = (Source(document="rag.pdf", page=3, excerpt="RAG combines retrieval."),)
    cited = (Source(document="rag.pdf", page=3, excerpt="The exam is on Friday."),)
    check = verify_citations(cited, evidence)[0]
    assert check.valid is False
    assert "excerpt not found" in check.reason
