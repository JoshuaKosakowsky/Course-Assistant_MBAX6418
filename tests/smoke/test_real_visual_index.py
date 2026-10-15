"""Smoke test: MSpanny's REAL VisualIndex class (PR #11, unmerged) driven
with INJECTED deterministic fake embedders — this is mechanics
verification, NOT live model verification.

Reproducibility (no developer paths):
- VISUAL_INDEX_SRC env var -> run against that checkout (dev-time).
- otherwise, if app.visual_index is importable from THIS repo (i.e. PR #11
  has merged), run against the installed module directly.
- otherwise SKIP with a clear reason (CI without the checkout).

The test runs a subprocess in the target checkout so package names never
collide with this repo's app package.
"""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.smoke


def _target_checkout():
    """Return the directory to run the real-VisualIndex runner in, or None."""
    src = os.environ.get("VISUAL_INDEX_SRC")
    if src and os.path.isdir(src) and os.path.isfile(
            os.path.join(src, "app", "visual_index.py")):
        return Path(src)
    # PR #11 merged? then the module lives in this repo
    if importlib.util.find_spec("app.visual_index") is not None:
        return REPO_ROOT
    return None


requires_worktree = pytest.mark.skipif(
    _target_checkout() is None,
    reason="no PR #11 VisualIndex checkout available: set VISUAL_INDEX_SRC "
           "to a feat/visual-index checkout (or merge PR #11); "
           "unit tests cover the adapter with stubs regardless",
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

# ---- integrity scenario: collection loss + repair via PUBLIC API -------
idx2 = VisualIndex(root / "i2", embedder=emb)
ids2 = [idx2.add_image("doc-a", n, p) for n, p in enumerate(paths, start=1)]
assert idx2.count() == 3
# simulate partial collection loss (TEST HARNESS ONLY: direct Chroma delete;
# the app code itself never touches the private collection)
lost = ids2[0]
idx2._collection.delete(ids=[lost])
assert idx2.count() == 2
# repair through the public interface, exactly as app/visual.py does:
# embed the page -> query k=1 -> distance > eps => vector lost -> re-add
restored = 0
for n, p in enumerate(paths, start=1):
    vec = emb(Image.open(p))
    top = idx2.query_embedding(vec, k=1)
    if not top or top[0].distance > 0.05:
        idx2.add_image("doc-a", n, p, embedding=vec)
        restored += 1
assert restored == 1 and idx2.count() == 3
verify = idx2.query_embedding(emb(Image.open(paths[0])), k=1)
assert verify[0].page == 1 and verify[0].distance <= 0.05

print(json.dumps({
    "pages": [h.page for h in hits],
    "text_page": thits[0].page,
    "stable_id": same == ids[0],
    "removed": removed,
    "lost_repaired": restored,
    "post_repair_self_distance": round(verify[0].distance, 6),
}))
"""


@requires_worktree
def test_real_visual_index_class_with_fake_embeddings():
    checkout = _target_checkout()
    result = subprocess.run(
        [sys.executable, "-c", RUNNER],
        cwd=str(checkout),
        env={**os.environ, "PYTHONPATH": str(checkout),
             "PYTHONIOENCODING": "utf-8"},
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    data = json.loads(result.stdout.strip().splitlines()[-1])
    assert data["pages"][0] == 2        # exact-image query: page 2 first
    assert data["text_page"] == 3       # text bridge: page 3 first
    assert data["stable_id"] is True    # content-addressed image id stable
    assert data["removed"] == 3         # remove_doc purged everything
    # integrity scenario on her REAL class: lost vector detected via the
    # public count()/query_embedding() API and re-added, distance ~0
    assert data["lost_repaired"] == 1
    assert data["post_repair_self_distance"] <= 0.05
