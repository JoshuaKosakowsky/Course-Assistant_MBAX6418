"""Offline tests for the chunk-size comparison."""

import pytest

from app.chunking import PageText, chunk_pages
from app.keyword import KeywordIndex
from scripts.compare_chunk_sizes import (
    CHUNK_SIZES,
    OVERLAP,
    QUERIES,
    evaluate,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("size", CHUNK_SIZES)
def test_comparison_produces_valid_metrics(size):
    summary, details = evaluate(size)

    assert summary["chunk_size"] == size
    assert summary["chunk_count"] > 0
    assert summary["query_count"] == len(QUERIES)
    assert len(details) == len(QUERIES)

    for metric in ("accuracy_at_1", "hit_rate_at_3", "mrr_at_3"):
        assert 0 <= summary[metric] <= 1


@pytest.mark.parametrize("size", CHUNK_SIZES)
def test_comparison_is_repeatable(size):
    assert evaluate(size) == evaluate(size)


@pytest.mark.parametrize(
    "size,overlap",
    [(0, 0), (-1, 0), (250, 250), (250, -1)],
)
def test_invalid_chunk_settings_raise(size, overlap):
    with pytest.raises(ValueError):
        chunk_pages(
            "doc",
            "lecture.pdf",
            [PageText(1, "sample text")],
            chunk_size=size,
            chunk_overlap=overlap,
        )


def test_empty_index_returns_no_results():
    assert KeywordIndex().search("retrieval") == []


def test_empty_query_returns_no_results():
    chunks = chunk_pages(
        "doc",
        "lecture.pdf",
        [PageText(1, "retrieval augmented generation")],
    )
    index = KeywordIndex()
    index.add(chunks)

    assert index.search("") == []
    assert index.search("   ") == []


def test_document_filter_excludes_other_documents():
    index = KeywordIndex()

    for doc_id in ("first", "second"):
        index.add(chunk_pages(
            doc_id,
            f"{doc_id}.pdf",
            [PageText(1, "retrieval augmented generation")],
        ))

    hits = index.search("retrieval", doc_ids=["first"])

    assert hits
    assert all(hit.chunk.doc_id == "first" for hit in hits)
    assert index.search("retrieval", doc_ids=[]) == []


def test_removed_document_does_not_reappear():
    chunks = chunk_pages(
        "removed-doc",
        "lecture.pdf",
        [PageText(1, "retrieval augmented generation")],
        chunk_size=250,
        chunk_overlap=OVERLAP,
    )

    index = KeywordIndex()
    index.add(chunks)

    assert index.search("retrieval")
    index.remove_document("removed-doc")

    assert index.search("retrieval") == []
    assert index.size == 0
