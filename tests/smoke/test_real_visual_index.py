"""Smoke test: MSpanny's REAL VisualIndex class (PR #11, unmerged) driven
through a read-only worktree with INJECTED deterministic fake embedders.

Requires VISUAL_INDEX_SRC (default: the review worktree path) pointing at
a checkout of origin/feat/visual-index. Skips cleanly when absent — this
is mechanics verification against her real implementation, NOT live model
verification (fake embeddings; port 9003 behavior lives in e2e tests).
The test never imports her branch's app package into this process; it
runs a small runner subprocess inside the worktree so package names never
collide.
"""

import json
import os
import subprocess
import sys

import pytest

WORKTREE = os.environ.get(
    "VISUAL_INDEX_SRC",
    r"C:\Users\matty\Documents\review-wt-visual-index",
)

pytestmark = pytest.mark.smoke

requires_worktree = pytest.mark.skipif(
    not os.path.isdir(WORKTREE) or not os.path.isfile(
        os.path.join(WORKTREE, "app", "visual_index.py")),
    reason="PR #11 read-only worktree checkout not present "
           "(set VISUAL_INDEX_SRC)",
)

RUNNER = r"""
import json, sys, tempfile
from pathlib import Path

from app.visual_index import VisualIndex, VisualHit, _image_id
from app.embeddings import fake_embedder

from PIL import Image, ImageDraw

root = Path(tempfile.mkdtemp())
emb = fake_embedder(dims=8)  # deterministic unit-length image vectors

def make_img(path, text="page"):
    im = Image.new("RGB", (40, 40), "white")
    ImageDraw.Draw(im).text((4, 14), text, fill="black")
    im.save(path)
    return path

idx = VisualIndex(root=root, embedder=emb)
paths = [make_img(root / f"page-{n}.png", f"p{n}") for n in (1, 2, 3)]
ids = [idx.add_image("doc-9", n, p) for n, p in enumerate(paths, start=1)]
assert len(set(ids)) == 3

# query by embed vector: exact image must come back first (distance ~0)
target = emb(Image.open(paths[1]))
hits = idx.query_embedding(target, k=3)
assert hits[0].page == 2, [h.page for h in hits]

# query_text via the injected (fake, deterministic) text bridge
def text_bridge(text):
    n = int(text.strip().split()[-1])  # "page 3" -> 3
    return emb(Image.open(paths[n - 1]))

thits = idx.query_text("page 3", k=3, text_embedder=text_bridge)
assert thits[0].doc_id == "doc-9" and thits[0].page == 3

# image id stability + removal
same = idx.add_image("doc-9", 1, paths[0])
assert same == ids[0]
assert idx.count() == 3
removed = idx.remove_doc("doc-9")
assert removed == 3 and idx.count() == 0

print(json.dumps({
    "pages": [h.page for h in hits],
    "text_page": thits[0].page,
    "stable_id": same == ids[0],
    "removed": removed,
}))
"""


@requires_worktree
def test_real_visual_index_class_with_fake_embeddings():
    result = subprocess.run(
        [sys.executable, "-c", RUNNER],
        cwd=WORKTREE,
        env={**os.environ, "PYTHONPATH": WORKTREE, "PYTHONIOENCODING": "utf-8"},
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    data = json.loads(result.stdout.strip().splitlines()[-1])
    assert data["pages"][0] == 2        # exact-image query: page 2 first
    assert data["text_page"] == 3       # text bridge: page 3 first
    assert data["stable_id"] is True    # content-addressed image id stable
    assert data["removed"] == 3         # remove_doc purged everything
