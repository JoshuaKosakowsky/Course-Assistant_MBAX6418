"""Opt-in e2e tests against the REAL class text-embedding endpoint.

Port 9002 (http://dobolyi.com:9002/v1), model
nvidia/Nemotron-3-Embed-1B-BF16 (verified 2048-dim in M0).

These tests hit a shared keyed class service and REQUIRE a real
CLASS_API_KEY in the local gitignored .env, so they auto-skip without it
and are excluded from the default pytest run (e2e marker). Run with:
    pytest -m e2e
"""

import pytest

from app import config
from app.chunking import PageText, chunk_pages
from app.embeddings import TextEmbeddingClient
from app.keyword import KeywordIndex
from app.vector import TextVectorIndex

pytestmark = pytest.mark.e2e

requires_env = pytest.mark.skipif(
    not config.TEXT_EMBED_API_KEY or "dummy" in config.TEXT_EMBED_API_KEY.lower(),
    reason="requires a real CLASS_API_KEY in the local .env",
)

FIN_TEXT = ("Free cash flow equals operating cash flow minus capital "
            "expenditures. Discounted cash flow valuation estimates "
            "enterprise value from projected cash flows.")
MKT_TEXT = ("Brand positioning defines how customers perceive a product. "
            "Customer segmentation groups buyers by behavior and needs.")
OPS_TEXT = ("Inventory management balances holding costs against stockouts. "
            "Supply chain logistics optimizes delivery and warehouse flow.")


def make_chunks():
    out = []
    for doc_id, fname, text in (
        ("doc-finance", "valuation.pdf", FIN_TEXT),
        ("doc-marketing", "brand-strategy.pdf", MKT_TEXT),
        ("doc-ops", "supply-chain.pdf", OPS_TEXT),
    ):
        out.extend(
            chunk_pages(doc_id, fname, [PageText(page_no=1, text=text)],
                        chunk_size=200, chunk_overlap=20, format=".pdf")
        )
    return out


@requires_env
def test_real_embeddings_dimension(tmp_path):
    client = TextEmbeddingClient()
    vectors = client.embed([
        "the quick brown fox jumps",
        "free cash flow and valuation",
        "brand positioning strategy",
    ])
    assert len(vectors) == 3
    assert all(len(v) == 2048 for v in vectors)  # documented model output


@requires_env
def test_semantic_retrieval_with_real_embeddings(tmp_path):
    """The case BM25 fails: a paraphrased question with no word overlap.

    Query chosen empirically against the live endpoint: "company purchase
    price estimation" shares no vocabulary with the finance chunk ("free
    cash flow ... enterprise value ...") yet returns it first. (The
    sibling paraphrase "what is a firm worth" ranks finance a close
    second to the ops doc on this tiny corpus — model behavior documented
    in results/semantic-search-demo.txt, not asserted here.)
    """
    keyword = KeywordIndex()
    semantic = TextVectorIndex(
        TextEmbeddingClient(), persist_dir=tmp_path / "db"
    )
    chunks = make_chunks()
    keyword.add(chunks)
    semantic.add(chunks)

    query = "company purchase price estimation"
    kw_hits = keyword.search(query)
    vec_hits = semantic.search(query)

    assert kw_hits == [], "BM25 should miss the paraphrase (no word overlap)"
    assert vec_hits, "semantic retrieval should return candidates"
    assert vec_hits[0].chunk.doc_id == "doc-finance"
    assert vec_hits[0].score > vec_hits[-1].score
