"""Compare answer quality with reranking on/off; live requests require --live.

Results contain source content: keep them under gitignored results/raw until
the team reviews what is appropriate to publish. Correctness is human-rated.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter


def validate_questions(questions):
    if not isinstance(questions, list) or not 5 <= len(questions) <= 10:
        raise ValueError("Use five to ten evaluation questions.")
    ids = set()
    kinds = []
    for question in questions:
        if (not isinstance(question, dict) or not isinstance(question.get("id"), str)
                or not question["id"].strip() or question["id"] in ids
                or not isinstance(question.get("question"), str)
                or not question["question"].strip() or len(question["question"]) > 2000
                or question.get("kind") not in {"text", "visual", "missing"}
                or not isinstance(question.get("expected_sources"), list)):
            raise ValueError("Invalid or duplicate evaluation question.")
        for source in question["expected_sources"]:
            if (not isinstance(source, dict) or not isinstance(source.get("document"), str)
                    or not source["document"].strip() or type(source.get("page")) is not int
                    or source["page"] < 1):
                raise ValueError("Invalid expected source.")
        if (question["kind"] == "missing") != (not question["expected_sources"]):
            raise ValueError("Answerable questions need expected sources; missing questions do not.")
        ids.add(question["id"])
        kinds.append(question["kind"])
    if kinds.count("visual") < 2 or "missing" not in kinds:
        raise ValueError("Include two visual questions and one missing-information question.")


def evaluate(questions, services, qna, doc_ids, *, clock=perf_counter):
    """Identical questions and document IDs for both real/injected services.

    Retrieval hit and model grounding flags are diagnostic, never human
    correctness or entailment scores. Source excerpts remain local by default.
    """
    validate_questions(questions)
    if set(services) != {"rerank_off", "rerank_on"}:
        raise ValueError("Supply exactly rerank_off and rerank_on services.")
    if not doc_ids or any(not isinstance(value, str) or not value for value in doc_ids):
        raise ValueError("Choose document IDs before evaluating.")
    rows = []
    for position, question in enumerate(questions):
        # Counterbalance order to reduce systematic warmup/order effects.
        order = ("rerank_off", "rerank_on") if position % 2 == 0 else ("rerank_on", "rerank_off")
        for approach in order:
            start = clock()
            row = {"question_id": question["id"], "kind": question["kind"],
                   "question": question["question"], "approach": approach,
                   "correct": None, "sources_support_answer": None,
                   "original_image_matches": None, "review_notes": "", "error": None}
            try:
                outcome = services[approach].retrieve(
                    question["question"], doc_ids=tuple(doc_ids), top_k=5, include_images=True)
                if any(item.doc_id not in doc_ids for item in outcome.evidence.items):
                    raise ValueError("Evidence escaped document selection.")
                sources = outcome.evidence.to_sources()
                if sources:
                    result = qna.answer(question["question"], tuple(sources))
                    row.update(answer=result.answer, grounded=result.grounded,
                               sources=[asdict(source) for source in result.sources])
                else:
                    from app.qna import MISSING_INFO_FALLBACK
                    row.update(answer=MISSING_INFO_FALLBACK, grounded=False, sources=[])
                row["stages"] = {stage: state for stage, state in outcome.stages.items()
                                 if stage in {"keyword", "semantic", "visual", "fusion", "rerank"}
                                 and state in {"ok", "skipped", "degraded"}}
                row["expected_page_retrieved"] = (None if question["kind"] == "missing" else any(
                    item.filename == expected["document"] and item.page_no == expected["page"]
                    for item in outcome.evidence.items for expected in question["expected_sources"]))
                row["rerank_comparison_valid"] = (
                    outcome.stages.get("rerank") == ("ok" if approach == "rerank_on" else "skipped"))
            except Exception:
                # Do not serialize raw upstream bodies, configuration or credentials.
                row.update(error="Evaluation failed; inspect server configuration privately.",
                           answer=None, grounded=False, sources=[], stages={},
                           expected_page_retrieved=None, rerank_comparison_valid=False)
            row["seconds"] = round(clock() - start, 6)
            rows.append(row)
    return rows


def resolve_documents(store, names):
    if (not isinstance(names, list) or not names
            or any(not isinstance(name, str) or not name.strip() for name in names)
            or len(set(names)) != len(names)):
        raise ValueError("List unique material filenames in the question set.")
    ids = []
    for name in names:
        matches = [doc for doc in store.list() if doc.filename == name]
        if len(matches) != 1:
            raise ValueError("Every selected filename must identify exactly one library document.")
        ids.append(matches[0].doc_id)
    return tuple(ids)


def summarize(rows):
    """Compare only complete reviewed pairs; disclose all missing/invalid rows."""
    by_question = {}
    for row in rows:
        approach = row.get("approach")
        if approach not in {"rerank_off", "rerank_on"}:
            raise ValueError("Unknown comparison approach.")
        pair = by_question.setdefault(row["question_id"], {})
        if approach in pair:
            raise ValueError("Duplicate comparison row.")
        pair[approach] = row
    reviewed = []
    for pair in by_question.values():
        if set(pair) != {"rerank_off", "rerank_on"}:
            continue
        if all(not row.get("error") and row.get("rerank_comparison_valid") is True
               and type(row.get("correct")) is bool
               and type(row.get("sources_support_answer")) is bool
               and (row.get("kind") != "visual" or type(row.get("original_image_matches")) is bool)
               for row in pair.values()):
            reviewed.append(pair)
    result = {"questions": len(by_question), "fully_reviewed_valid_pairs": len(reviewed),
              "unreviewed_or_invalid_pairs": len(by_question) - len(reviewed), "approaches": {}}
    for approach in ("rerank_off", "rerank_on"):
        selected = [pair[approach] for pair in reviewed]
        visual = [row for row in selected if row.get("kind") == "visual"]
        result["approaches"][approach] = {
            "answer_accuracy": sum(row["correct"] for row in selected) / len(selected) if selected else None,
            "source_support_rate": sum(row["sources_support_answer"] for row in selected) / len(selected) if selected else None,
            "mean_seconds": sum(row["seconds"] for row in selected) / len(selected) if selected else None,
            "original_image_match_rate": (sum(row["original_image_matches"] for row in visual)
                                          / len(visual) if visual else None),
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=Path("docs/course-evaluation.json"))
    parser.add_argument("--output", type=Path, default=Path("results/raw/course-evaluation.json"))
    parser.add_argument("--live", action="store_true", help="Deliberately use configured class services")
    parser.add_argument("--summarize", type=Path, help="Summarize reviewed local results without requests")
    args = parser.parse_args(argv)
    if args.summarize:
        try:
            data = json.loads(args.summarize.read_text(encoding="utf-8"))
            print(json.dumps(summarize(data["rows"]), indent=2))
            return 0
        except Exception:
            print("Unable to summarize results. Check the local review fields and row identities.")
            return 1
    if not args.live:
        parser.error("No requests sent. Pass --live only when deliberately using class services.")
    try:
        specification = json.loads(args.questions.read_text(encoding="utf-8"))
        validate_questions(specification["questions"])
        from app.documents import DocumentStore
        from app.study_ui import production_study_services
        from app.retrieval import build_retrieval_service
        from app.ui import _production_indexer
        # Validate selection before any index repair could make a remote request.
        store = DocumentStore()
        ids = resolve_documents(store, specification["materials"])
        _, indexer = _production_indexer(store)
        service_on, qna, _ = production_study_services(indexer)
        if qna is None:
            raise ValueError("Vision generation unavailable.")
        indexer.register_hooks()
        indexer.reconcile()
        # Reconcile replaces keyword; reconstruct services against the current index.
        service_on, qna, _ = production_study_services(indexer)
        service_off = build_retrieval_service(indexer, reranker_factory=lambda: None)
        rows = evaluate(specification["questions"],
                        {"rerank_off": service_off, "rerank_on": service_on}, qna, ids)
        document = {"status": "Unreviewed model results; human correctness ratings required",
                    "comparison": "rerank off vs on; identical questions/materials/models",
                    "question_specification": specification, "selected_doc_ids": ids, "rows": rows}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
        print("Evaluation saved locally. Review correctness, source support and images before publishing.")
        return 1 if any(row["error"] for row in rows) else 0
    except Exception:
        print("Evaluation unavailable. Check material selection and server-side configuration privately.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
