"""Evaluation must not fabricate ratings or silently compare degraded runs."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.evidence import EvidencePack
from scripts.evaluate_course import evaluate, main, resolve_documents, summarize, validate_questions

pytestmark = pytest.mark.unit


@pytest.fixture
def questions():
    path = Path(__file__).resolve().parents[2] / "docs" / "course-evaluation.json"
    return json.loads(path.read_text(encoding="utf-8"))["questions"]


def test_course_set_has_visual_missing_and_verified_source_locations(questions):
    validate_questions(questions)
    assert len(questions) == 8
    assert next(q for q in questions if q["id"] == "prod-meme")["expected_sources"][0]["page"] == 33


def test_invalid_set_is_rejected_before_services(questions):
    with pytest.raises(ValueError):
        validate_questions(questions[:3])
    questions[1]["id"] = questions[0]["id"]
    with pytest.raises(ValueError):
        validate_questions(questions)


def test_no_live_opt_in_exits_before_constructing_services():
    with pytest.raises(SystemExit) as error:
        main([])
    assert error.value.code == 2


def test_identical_materials_and_no_fabricated_ratings(questions):
    calls = []

    class Service:
        def __init__(self, label):
            self.label = label

        def retrieve(self, question, **kwargs):
            calls.append((self.label, question, kwargs))
            return SimpleNamespace(evidence=EvidencePack(question),
                                   stages={"rerank": "ok" if self.label == "on" else "skipped"})

    class NeverModel:
        def answer(self, *args):
            pytest.fail("empty evidence must not call model")

    ticks = iter(range(32))
    rows = evaluate(questions, {"rerank_on": Service("on"), "rerank_off": Service("off")},
                    NeverModel(), ("doc1", "doc2"), clock=lambda: next(ticks))
    assert len(rows) == 16 and all(row["seconds"] == 1 for row in rows)
    assert all(row["correct"] is row["sources_support_answer"] is None for row in rows)
    assert all(call[2]["doc_ids"] == ("doc1", "doc2") for call in calls)
    assert calls[0][0] == "off" and calls[2][0] == "on"
    assert summarize(rows)["fully_reviewed_valid_pairs"] == 0
    assert summarize(rows)["approaches"]["rerank_on"]["answer_accuracy"] is None


def test_failures_are_generic_and_invalidate_comparison(questions):
    class Broken:
        def retrieve(self, *args, **kwargs):
            raise RuntimeError("PRIVATE_ENDPOINT_TOKEN")
    rows = evaluate(questions, {"rerank_on": Broken(), "rerank_off": Broken()}, object(), ("doc1",))
    assert "PRIVATE_ENDPOINT_TOKEN" not in json.dumps(rows)
    assert all(row["error"] and not row["rerank_comparison_valid"] for row in rows)


def test_summary_uses_paired_human_ratings_and_rejects_degraded_runs():
    rows = [{"question_id": "q", "approach": approach, "kind": "visual",
             "correct": correct, "sources_support_answer": True, "original_image_matches": True,
             "rerank_comparison_valid": True, "seconds": seconds}
            for approach, correct, seconds in (("rerank_off", False, 1), ("rerank_on", True, 2))]
    summary = summarize(rows)
    assert summary["fully_reviewed_valid_pairs"] == 1
    assert summary["approaches"]["rerank_on"]["answer_accuracy"] == 1
    assert summary["approaches"]["rerank_off"]["answer_accuracy"] == 0
    rows[1]["rerank_comparison_valid"] = False
    assert summarize(rows)["fully_reviewed_valid_pairs"] == 0
    with pytest.raises(ValueError):
        summarize(rows + [rows[0]])


def test_document_resolution_fails_closed_on_missing_or_duplicate_filenames():
    store = SimpleNamespace(list=lambda: [SimpleNamespace(filename="a.pdf", doc_id="a")])
    assert resolve_documents(store, ["a.pdf"]) == ("a",)
    with pytest.raises(ValueError):
        resolve_documents(store, ["missing.pdf"])
    store.list = lambda: [SimpleNamespace(filename="a.pdf", doc_id=value) for value in ("a", "b")]
    with pytest.raises(ValueError):
        resolve_documents(store, ["a.pdf"])
