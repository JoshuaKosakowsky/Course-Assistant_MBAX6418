"""Offline BM25 comparison of three chunk sizes."""

import csv
import json
from pathlib import Path

from app.chunking import PageText, chunk_pages
from app.keyword import KeywordIndex


CHUNK_SIZES = (250, 500, 1000)
OVERLAP = 64
TOP_K = 3
OUTPUT = Path("results")

# Synthetic lecture pages with known answers.
PAGES = [
    PageText(1, (
        "Retrieval augmented generation combines information retrieval "
        "with language model generation. A retriever identifies relevant "
        "passages from course documents. The generator uses those passages "
        "to produce a grounded response. Citations help students verify "
        "where the supporting information originated. "
    ) * 5),
    PageText(2, (
        "Precision measures the proportion of retrieved items that are "
        "relevant. Recall measures the proportion of relevant items "
        "successfully retrieved. Evaluation requires a defined ground "
        "truth and consistent search queries. "
    ) * 5),
    PageText(3, (
        "Data leakage occurs when information from evaluation data "
        "improperly influences model training. A separate test set helps "
        "estimate generalization to unseen examples. Cross validation "
        "can support model selection on training data. "
    ) * 5),
    PageText(4, (
        "Clinical decision support systems help healthcare professionals "
        "interpret patient information. Human oversight remains important "
        "for safety, accountability, and appropriate clinical judgment. "
    ) * 5),
]

# Each query has one known relevant page.
QUERIES = [
    ("retrieval augmented generation citations", 1),
    ("precision recall evaluation ground truth", 2),
    ("data leakage unseen test set", 3),
    ("clinical decision support human oversight", 4),
]


def evaluate(chunk_size):
    chunks = chunk_pages(
        doc_id="synthetic-course",
        filename="synthetic-lecture.pdf",
        pages=PAGES,
        chunk_size=chunk_size,
        chunk_overlap=OVERLAP,
        format=".pdf",
    )

    index = KeywordIndex()
    index.add(chunks)

    details = []
    reciprocal_ranks = []

    for query, expected_page in QUERIES:
        hits = index.search(query, top_k=TOP_K)
        pages = [hit.chunk.page_no for hit in hits]

        rank = (
            pages.index(expected_page) + 1
            if expected_page in pages
            else None
        )

        reciprocal_ranks.append(1 / rank if rank else 0)

        details.append({
            "chunk_size": chunk_size,
            "query": query,
            "expected_page": expected_page,
            "retrieved_pages": pages,
            "correct_at_1": bool(pages and pages[0] == expected_page),
            "correct_at_3": rank is not None,
            "rank": rank,
        })

    summary = {
        "chunk_size": chunk_size,
        "chunk_overlap": OVERLAP,
        "chunk_count": len(chunks),
        "query_count": len(QUERIES),
        "accuracy_at_1": sum(
            item["correct_at_1"] for item in details
        ) / len(details),
        "hit_rate_at_3": sum(
            item["correct_at_3"] for item in details
        ) / len(details),
        "mrr_at_3": sum(reciprocal_ranks) / len(reciprocal_ranks),
    }

    return summary, details


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    summaries = []
    details = []

    for size in CHUNK_SIZES:
        summary, query_details = evaluate(size)
        summaries.append(summary)
        details.extend(query_details)

    with (OUTPUT / "chunk-size-comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(summaries[0])
        )
        writer.writeheader()
        writer.writerows(summaries)

    (OUTPUT / "chunk-size-details.json").write_text(
        json.dumps(details, indent=2), encoding="utf-8"
    )

    print("\nBM25 chunk-size comparison")
    print("-" * 65)
    print(
        f"{'Size':>6} {'Chunks':>8} "
        f"{'Acc@1':>10} {'Hit@3':>10} {'MRR@3':>10}"
    )

    for row in summaries:
        print(
            f"{row['chunk_size']:>6} "
            f"{row['chunk_count']:>8} "
            f"{row['accuracy_at_1']:>10.2%} "
            f"{row['hit_rate_at_3']:>10.2%} "
            f"{row['mrr_at_3']:>10.3f}"
        )

    print("\nResults saved in results/")


if __name__ == "__main__":
    main()
