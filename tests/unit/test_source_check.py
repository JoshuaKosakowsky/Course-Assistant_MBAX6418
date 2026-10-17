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
    factual_conflict,
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


# -- factual-conflict awareness (numbers, dates, negation) ------------------

def test_number_conflict_is_never_supported():
    src = Source(document="d", page=1, excerpt="The exam is on October 15 in week 15 of the course.")
    check = check_sources("The exam is on October 20.", (src,))[0]
    assert check.conflict is True
    assert check.overlap >= SUPPORTED_THRESHOLD  # content terms all match...
    assert check.verdict == "weak"  # ...but the date contradicts: never "supported"


def test_matching_numbers_remain_supported():
    src = Source(document="d", page=1, excerpt="The exam is on October 20 in week 15 of the course.")
    check = check_sources("The exam is on October 20.", (src,))[0]
    assert check.conflict is False
    assert check.verdict == "supported"


def test_contradictory_quantity_is_conflict():
    src = Source(document="d", page=1, excerpt="This course awards 4 credits.")
    check = check_sources("This course awards 5 credits.", (src,))[0]
    assert check.conflict is True
    assert check.verdict == "weak"


def test_negation_conflict_is_never_supported():
    src = Source(document="d", page=1, excerpt="The exam is on Friday in week 15.")
    check = check_sources("The exam is not on Friday.", (src,))[0]
    assert check.conflict is True
    assert check.overlap >= SUPPORTED_THRESHOLD
    assert check.verdict == "weak"


def test_shared_negation_is_not_a_conflict():
    src = Source(document="d", page=1, excerpt="The exam is not on Friday this semester.")
    check = check_sources("The exam is not on Friday.", (src,))[0]
    assert check.conflict is False
    assert check.verdict == "supported"


def test_evidence_negation_conflicts_with_affirmative_answer():
    """Answer affirmative, evidence negated — the reverse of the previous
    case, which the asymmetrical check used to miss."""
    src = Source(document="d", page=1, excerpt="The exam is not on Friday in week 15.")
    check = check_sources("The exam is on Friday.", (src,))[0]
    assert check.conflict is True
    assert check.overlap >= SUPPORTED_THRESHOLD
    assert check.verdict == "weak"


def test_uppercase_negation_is_detected():
    src = Source(document="d", page=1, excerpt="The exam is on Friday in week 15.")
    check = check_sources("The exam is NOT on Friday.", (src,))[0]
    assert check.conflict is True
    assert check.verdict == "weak"  # case-normalized negation never certified


def test_uppercase_negation_in_evidence_is_detected():
    src = Source(document="d", page=1, excerpt="The exam is NOT on Friday this semester.")
    check = check_sources("The exam is on Friday.", (src,))[0]
    assert check.conflict is True
    assert check.verdict == "weak"


def test_matching_negative_claims_still_supported():
    src = Source(document="d", page=1, excerpt="The exam is NOT on Friday this semester.")
    check = check_sources("The exam is not on Friday.", (src,))[0]
    assert check.conflict is False
    assert check.verdict == "supported"


def test_affirmative_claim_with_affirmative_evidence_still_supported():
    """Control: genuinely supported affirmative claims must stay supported."""
    src = Source(document="d", page=1, excerpt="The exam is on Friday in week 15.")
    check = check_sources("The exam is on Friday.", (src,))[0]
    assert check.conflict is False
    assert check.verdict == "supported"


def test_factual_conflict_flag():
    assert factual_conflict("5 credits", "4 credits") is True
    assert factual_conflict("October 20", "October 20") is False
    assert factual_conflict("not on Friday", "on Friday") is True
    assert factual_conflict("on Friday", "not on Friday") is True  # either direction
    assert factual_conflict("NOT on Friday", "on Friday") is True  # case-insensitive
    assert factual_conflict("not on Friday", "not on Friday") is False
    assert factual_conflict("no make-up", "no make-up") is False
    assert factual_conflict("plain text", "plain text") is False


def test_judge_conflict_caps_supported_to_weak():
    src = Source(document="d", page=1, excerpt="The exam is on Friday in week 15.")
    judge = JudgeStub([True])  # judge says supported, but facts conflict
    check = judge_sources("The exam is not on Friday.", (src,), judge)[0]
    assert check.verdict == "weak"  # judge cannot overturn a factual conflict


def test_judge_cannot_promote_evidence_side_negation():
    """Judge says supported, but the EVIDENCE is negated while the answer is
    affirmative — the judge must not certify the contradiction."""
    src = Source(document="d", page=1, excerpt="The exam is not on Friday in week 15.")
    judge = JudgeStub([True])
    check = judge_sources("The exam is on Friday.", (src,), judge)[0]
    assert check.conflict is True
    assert check.verdict == "weak"


# -- image path provenance -------------------------------------------------

def test_verify_citation_keeps_evidence_image_path():
    evidence = (Source(document="rag.pdf", page=3, excerpt="RAG combines retrieval.", image_path="notes/slide3.png"),)
    cited = (Source(document="rag.pdf", page=3, excerpt="RAG combines retrieval.", image_path="notes/slide3.png"),)
    check = verify_citations(cited, evidence)[0]
    assert check.valid is True
    assert check.image_path == "notes/slide3.png"


def test_verify_citation_normalizes_path_separators():
    evidence = (Source(document="rag.pdf", page=3, excerpt="RAG combines retrieval.", image_path="notes/slide3.png"),)
    cited = (Source(document="rag.pdf", page=3, excerpt="RAG combines retrieval.", image_path=r"notes\slide3.png"),)
    check = verify_citations(cited, evidence)[0]
    assert check.valid is True
    assert check.image_path == "notes/slide3.png"  # evidence-derived, POSIX


def test_verify_citation_discards_fabricated_image_path():
    evidence = (Source(document="rag.pdf", page=3, excerpt="RAG combines retrieval.", image_path="notes/slide3.png"),)
    cited = (Source(document="rag.pdf", page=3, excerpt="RAG combines retrieval.", image_path="/etc/passwd"),)
    check = verify_citations(cited, evidence)[0]
    assert check.valid is True  # citation itself is authentic...
    assert check.image_path is None  # ...but the model's path is never trusted
    assert "discarded" in check.reason


def test_verify_citation_wrong_document_image_is_discarded():
    evidence = (
        Source(document="rag.pdf", page=3, excerpt="RAG combines retrieval.", image_path="notes/slide3.png"),
        Source(document="syllabus.pdf", page=1, excerpt="Office hours are on Tuesdays.", image_path="notes/office.png"),
    )
    # Authentic doc/page/excerpt, but image path pointing at a *different* document's image.
    cited = (Source(document="rag.pdf", page=3, excerpt="RAG combines retrieval.", image_path="notes/office.png"),)
    check = verify_citations(cited, evidence)[0]
    assert check.valid is True
    assert check.image_path is None
    assert "discarded" in check.reason


def test_verify_citation_without_image_path_stays_none():
    evidence = (Source(document="rag.pdf", page=3, excerpt="RAG combines retrieval.", image_path="notes/slide3.png"),)
    cited = (Source(document="rag.pdf", page=3, excerpt="RAG combines retrieval."),)
    check = verify_citations(cited, evidence)[0]
    assert check.valid is True
    assert check.image_path is None  # no image claimed, none injected
