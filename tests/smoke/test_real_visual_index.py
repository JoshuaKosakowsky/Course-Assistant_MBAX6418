"""Smoke test: MSpanny's REAL VisualIndex class (PR #11, unmerged) driven
with INJECTED deterministic fake embedders — this is mechanics
verification, NOT live model verification.

Reproducibility (no developer paths):
- VISUAL_INDEX_SRC env var -> run against that checkout (dev-time).
- otherwise, if app.visual_index is importable from THIS repo (i.e. PR #11
  has merged), run against the installed module directly.
- otherwise SKIP with a clear reason (CI without the checkout).

Two suites:

A) Her-class mechanics against the FINAL PR #11 API: STABLE slot ids
   (``doc_id:page`` — the stale-vector-replacement fix), upsert
   replacement of changed content, identical-content twins on distinct
   slots, collection-loss repair by slot identity, remove_doc.

B) CROSS-PR integration: OUR ``app.visual`` VisualIndexer (Phase 3 state
   machine, replacement/repair/retry logic) driving HER real
   ``VisualIndex`` with deterministic fake embeddings — changed-page
   replacement, identical-content pages, missing-vector repair,
   equal-count corruption, embedding failure, and restart
   reconciliation.

The runner runs in subprocesses so package names never collide: suite B
loads her tree under the ``pr11`` alias so ``app.*`` stays THIS repo.
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

RUNNER_A = r"""
import json, sys, tempfile
from pathlib import Path

from app.visual_index import VisualIndex
from app.visual_embeddings import fake_embedder

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
assert ids == ["doc-9:1", "doc-9:2", "doc-9:3"], ids   # STABLE slot ids
assert idx.count() == 3

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

# ---- slot-id semantics: re-indexing CHANGED content replaces in place --
new1 = make_img(root / "new1.png", "changed-page-1-content")
idx.add_image("doc-9", 1, new1)
assert idx.count() == 3                       # upsert: never stacked
hits_old = idx.query_embedding(emb(Image.open(paths[0])), k=3)
assert hits_old[0].page != 1                  # stale embedding is GONE
hits_new = idx.query_embedding(emb(Image.open(new1)), k=3)
assert hits_new[0].page == 1                  # new content is searchable
same = idx.add_image("doc-9", 1, new1)        # identical re-index: no-op
assert same == "doc-9:1" and idx.count() == 3

removed = idx.remove_doc("doc-9")
assert removed == 3 and idx.count() == 0

# ---- integrity scenario: collection loss + repair via PUBLIC API -------
idx2 = VisualIndex(root / "i2", embedder=emb)
# TWIN images on pages 1 and 2 (identical bytes) + a distinct page 3; the
# stable slot ids embed the page number, so twins never share a slot
twin = root / "twin.png"
make_img(twin, "twin-content")
vid1 = idx2.add_image("doc-a", 1, twin)          # page 1 = twin bytes
vid2 = idx2.add_image("doc-a", 2, twin)          # page 2 = SAME twin bytes
vid3 = idx2.add_image("doc-a", 3, root / "page-3.png")
assert {vid1, vid2, vid3} == {"doc-a:1", "doc-a:2", "doc-a:3"}
assert idx2.count() == 3
# simulate partial collection loss: page 1's vector vanishes (harness only;
# app code never touches the private collection)
idx2._collection.delete(ids=[vid1])
assert idx2.count() == 2
# repair through the PUBLIC interface with SLOT-identity matching (exactly
# as app/visual.py now does): require the expected slot among returned ids
restored = 0
for n, p in [(1, twin), (2, twin), (3, root / "page-3.png")]:
    vec = emb(Image.open(p))
    top = idx2.query_embedding(vec, k=2)
    if f"doc-a:{n}" not in [h.id for h in top]:
        idx2.add_image("doc-a", n, p, embedding=vec)
        restored += 1
assert restored == 1 and idx2.count() == 3
present = [h.id for h in idx2.query_embedding(emb(Image.open(twin)), k=2)]
assert "doc-a:1" in present and "doc-a:2" in present  # BOTH twins indexed

print(json.dumps({
    "pages": [h.page for h in hits],
    "text_page": thits[0].page,
    "slot_ids": ids,
    "replaced": hits_old[0].page != 1 and hits_new[0].page == 1,
    "removed": removed,
    "lost_repaired": restored,
    "twins_present": "doc-a:1" in present and "doc-a:2" in present,
}))
"""

RUNNER_B = r"""
import importlib.util, json, sys, tempfile
from pathlib import Path

checkout = Path(sys.argv[1])

# load MSpanny's tree under the "pr11" alias so app.* stays THIS repo
app_dir = checkout / "app"
spec = importlib.util.spec_from_file_location(
    "pr11", str(app_dir / "__init__.py"),
    submodule_search_locations=[str(app_dir)])
pr11 = importlib.util.module_from_spec(spec)
sys.modules["pr11"] = pr11
spec.loader.exec_module(pr11)
from pr11.visual_index import VisualIndex as RealVisualIndex
from pr11.visual_embeddings import fake_embedder as real_fake

from app.documents import DocumentStore
from app.parse import parse_file
from app.visual import VisualIndexer, VisualState, resize_for_embedding

import pymupdf
from PIL import Image, ImageDraw

root = Path(tempfile.mkdtemp())
emb = real_fake(dims=8)

def make_pdf(path, pages=2):
    d = pymupdf.open()
    for i in range(pages):
        pg = d.new_page()
        pg.insert_text((72, 72), "Free cash flow = OCF - capex (%d)" % i)
    d.save(str(path)); d.close()
    return path

def make_img(path, text="page"):
    im = Image.new("RGB", (40, 40), "white")
    ImageDraw.Draw(im).text((4, 14), text, fill="black")
    im.save(path)
    return path

store = DocumentStore(root=root / "data")
rec = store.add(make_pdf(root / "deck.pdf", 2))
parse_file(store.corpus_path(rec.doc_id), rec.doc_id,
           store.artifacts_dir / rec.doc_id)
doc_id = rec.doc_id
manifest = store.artifacts_dir / doc_id / "manifest.json"
import json as _json

vi = RealVisualIndex(root=root / "visual-idx", embedder=emb)
visual = VisualIndexer(vi, emb, store, state=VisualState(root / "vs.json"))

# 1) baseline index + STABLE slot ids on her real class
stats = visual.index_document(doc_id)
assert vi.count() == 2, vi.count()
slots = set(vi._collection.get(include=["metadatas"])["ids"])
assert slots == {f"{doc_id}:1", f"{doc_id}:2"}, slots
page1_path = Path(_json.loads(manifest.read_text(encoding="utf-8"))
                  ["pages"][0]["image_path"])
old_vec = emb(resize_for_embedding(Image.open(page1_path)))
assert vi.query_embedding(old_vec, k=2)[0].page == 1

# 2) CHANGED page -> SAME slots, stale embedding replaced, nothing stacked
make_img(page1_path, "completely different page-1 content")
stats2 = visual.index_document(doc_id)
assert vi.count() == 2, vi.count()
slots2 = set(vi._collection.get(include=["metadatas"])["ids"])
assert slots2 == slots, (slots2, slots)
new_vec = emb(resize_for_embedding(Image.open(page1_path)))
assert vi.query_embedding(new_vec, k=2)[0].page == 1      # new content
assert vi.query_embedding(old_vec, k=2)[0].page != 1      # stale gone

# 3) (twin slots are suite A's job — distinct slot ids per page number)

# 4) MISSING-vector repair by slot identity (loss via harness delete)
vi._collection.delete(ids=[f"{doc_id}:1"])
assert vi.count() == 1
sr = visual.reconcile()
assert sr.visual_repaired == 1
assert vi.count() == 2
assert f"{doc_id}:1" in set(vi._collection.get(include=["metadatas"])["ids"])

# 5) EQUAL-COUNT corruption: drop a slot AND upsert an unrelated vector
twin = root / "twin.png"
make_img(twin, "twin-content")
vi._collection.delete(ids=[f"{doc_id}:2"])
vi._collection.upsert(ids=["other-doc:9"], embeddings=[[1.0] + [0.0] * 7],
                      metadatas=[{"doc_id": "other-doc", "page": 9,
                                  "image_path": str(twin)}])
assert vi.count() == 2  # 1 + 1 == expected 2 -> counts match, must still fix
sc = visual.reconcile()
assert sc.visual_repaired == 1                      # doc:2 restored
ids_after = set(vi._collection.get(include=["metadatas"])["ids"])
assert f"{doc_id}:2" in ids_after
assert "other-doc:9" in ids_after                   # unrelated entry KEPT
# harness cleanup: drop the unrelated entry so later counts are predictable
vi._collection.delete(ids=["other-doc:9"])
assert vi.count() == 2

# 6) EMBEDDING failure: changed page -> preflight abort keeps old vectors
backend_ok = {"ok": True}
class Flaky:
    def __call__(self, image):
        if not backend_ok["ok"]:
            raise RuntimeError("embedding service down (simulated)")
        return emb(image)
visual.image_embedder = Flaky()
make_img(page1_path, "yet another page-1 content")
backend_ok["ok"] = False
sf = visual.index_document(doc_id)
assert sf.visual_failed and vi.count() == 2        # nothing deleted
assert f"{doc_id}:1" in set(vi._collection.get(include=["metadatas"])["ids"])
backend_ok["ok"] = True
sr2 = visual.index_document(doc_id)
assert not sr2.visual_failed and vi.count() == 2   # retry recovers

# 7) RESTART reconciliation: fresh state load + fresh handle on same data
visual2 = VisualIndexer(RealVisualIndex(root=root / "visual-idx",
                                        embedder=emb),
                        emb, store, state=VisualState(root / "vs.json"))
emb_before = int(vi._collection.count())
sr3 = visual2.reconcile()
assert sr3.visual_repaired == 0
assert vi._collection.count() == emb_before        # healthy start: zero churn

print(json.dumps({
    "slots": sorted(slots2),
    "replaced": True,
    "lost_repaired": sr.visual_repaired,
    "equal_count_repaired": sc.visual_repaired,
    "embed_failure_aborted": bool(sf.visual_failed) and vi.count() == 2,
    "restart_repaired": sr3.visual_repaired,
}))
"""


@requires_worktree
def test_real_visual_index_class_with_fake_embeddings():
    checkout = _target_checkout()
    result = subprocess.run(
        [sys.executable, "-c", RUNNER_A],
        cwd=str(checkout),
        env={**os.environ, "PYTHONPATH": str(checkout),
             "PYTHONIOENCODING": "utf-8"},
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    data = json.loads(result.stdout.strip().splitlines()[-1])
    assert data["pages"][0] == 2        # exact-image query: page 2 first
    assert data["text_page"] == 3       # text bridge: page 3 first
    assert data["slot_ids"] == ["doc-9:1", "doc-9:2", "doc-9:3"]
    assert data["replaced"] is True     # changed content replaces the slot
    assert data["removed"] == 3         # remove_doc purged everything
    assert data["lost_repaired"] == 1   # lost slot restored by identity
    assert data["twins_present"] is True


# (suite B's own assertions live inside the runner; the host asserts the
#  subprocess exit + JSON contract here)
@requires_worktree
def test_real_visual_index_cross_pr_integration():
    """OUR Phase 3 VisualIndexer driving MSpanny's REAL VisualIndex
    (final slot-id API): changed-page replacement, missing-vector repair,
    equal-count corruption, embedding failure, restart reconciliation."""
    checkout = _target_checkout()
    result = subprocess.run(
        [sys.executable, "-c", RUNNER_B, str(checkout)],
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT),
             "PYTHONIOENCODING": "utf-8"},
        capture_output=True, text=True, timeout=240,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    data = json.loads(result.stdout.strip().splitlines()[-1])
    assert len(data["slots"]) == 2
    assert data["slots"][0].endswith(":1") and data["slots"][1].endswith(":2")
    assert data["lost_repaired"] == 1
    assert data["equal_count_repaired"] == 1
    assert data["embed_failure_aborted"] is True
    assert data["restart_repaired"] == 0
