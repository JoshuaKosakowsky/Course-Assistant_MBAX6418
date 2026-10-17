"""Opt-in e2e tests against the REAL class multimodal reranker.

Port 9004 (http://dobolyi.com:9004/v1/rerank), model
Qwen/Qwen3-VL-Reranker-2B. Contract verified live 2026-10-16:

  POST /v1/rerank  {"model", "query", "documents": [str | multimodal], "top_n"?}
  -> {"id", "model", "usage", "results": [{"index", "document", "relevance_score"}]}

The multimodal document form is
  {"type": "content", "content": [{"type": "text", ...},
                                   {"type": "image_url",
                                    "image_url": {"url": "data:image/png;base64,..."}}]}

These hit a shared keyed class service and REQUIRE a real reranker key in
the local gitignored .env, so they auto-skip without it and are excluded
from the default pytest run (e2e marker). Run with:  pytest -m e2e
"""

import base64
from pathlib import Path

import pymupdf
import pytest

from app import config
from app.rerank import RerankClient

pytestmark = pytest.mark.e2e

requires_env = pytest.mark.skipif(
    not (config.RERANK_API_KEY) or "dummy" in (config.RERANK_API_KEY or "").lower()
    or "replace" in (config.RERANK_API_KEY or "").lower(),
    reason="requires a real CLASS_API_KEY_RERANK in the local .env",
)

FIN_TEXT = ("Free cash flow equals operating cash flow minus capital "
            "expenditures. Discounted cash flow valuation estimates "
            "enterprise value from projected cash flows.")
MKT_TEXT = ("Brand positioning defines how customers perceive a product. "
            "Customer segmentation groups buyers by behavior and needs.")
OPS_TEXT = ("Inventory management balances holding costs against stockouts. "
            "Supply chain logistics optimizes delivery and warehouse flow.")


@requires_env
def test_text_only_rerank_live():
    client = RerankClient(timeout=180)
    scores = client.rank(
        "What is free cash flow?",
        [FIN_TEXT, MKT_TEXT, OPS_TEXT],
    )
    assert len(scores) == 3
    assert all(isinstance(s, float) for s in scores.values())
    assert scores[0] > scores[1] and scores[0] > scores[2]  # finance wins


@requires_env
def test_image_aware_rerank_live(tmp_path):
    """One genuine image-aware request: an image-carrying candidate must
    beat a text-only one for an explicitly visual question."""
    png = tmp_path / "page-001.png"
    doc = pymupdf.open()
    pg = doc.new_page()
    pg.insert_text((72, 72), "Free cash flow and discounted cash flow valuation")
    pg.get_pixmap(dpi=100).save(str(png))
    doc.close()

    data_uri = f"data:image/png;base64,{base64.b64encode(png.read_bytes()).decode()}"
    docs = [
        MKT_TEXT,
        {
            "type": "content",
            "content": [
                {"type": "text", "text": "A slide about free cash flow and valuation."},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        },
    ]
    client = RerankClient(timeout=180)
    scores = client.rank("Which candidate shows a slide (an image)?", docs)
    assert len(scores) == 2
    assert scores[1] > scores[0], f"image candidate should win a visual query: {scores}"
