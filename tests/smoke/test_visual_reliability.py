"""Focused reliability audit tests (Issue #4, Step 8) — offline.

Covers the three confirmed defects:
  A) tracking state can outlive a lost ChromaDB vector  -> integrity repair
  B) a failed visual deletion abandons stale vectors    -> durable pending
                                                           retry across restart
  C) explicit --visual-backfill silently succeeds when the visual index
     is unavailable                                       -> non-zero exit

The fake backend behaves like MSpanny's VisualIndex (real cosine
distance over stored embeddings) so the repair path is exercised
exactly as it will run against ChromaDB.
"""

from __future__ import annotations

import json
import math
import pymupdf
import pytest
from pathlib import Path

from app.documents import DocumentStore
from app.indexing import CorpusIndexer, main as indexing_main
from app.keyword import KeywordIndex
from app.vector import TextVectorIndex
from app.visual import VisualIndexer, VisualState
from tests.fake_embeddings import FakeEmbedder
from PIL import Image

pytestmark = pytest.mark.smoke


# --------------------------------------------------------------------------
# Chroma-like backend
# --------------------------------------------------------------------------

class ChromaLikeBackend:
    """Duck-typed VisualIndex stand-in with REAL cosine distance.

    Public surface: add_image / remove_doc / count / query_embedding —
    exactly what MSpanny's VisualIndex exposes to our app code. Extra
    helpers (drop_vector, fail_next_remove, fail_next_add) exist purely
    for simulating failure/loss in tests.
    """

    def __init__(self):
        self.vectors: dict[str, dict] = {}
        self.added: list[tuple] = []
        self.removed: list[str] = []
        self.embed_calls = 0
        self.fail_remove = False
        self.add_fail_on_n: int | None = None   # fail add after N successes

    # -- public API (mirrors VisualIndex) --------------------------------

    def add_image(self, doc_id, page, image_path, embedding=None):
        if self.add_fail_on_n is not None:
            if self.embed_calls > self.add_fail_on_n:
                raise RuntimeError("simulated embed failure")
        # content-addressed id, exactly like MSpanny's _image_id():
        # {doc_id}:{page}:{sha256(full file bytes)[:10]} — embeds the page
        # identity, so twin images on different pages NEVER share an id
        digest = ChromaLikeBackend._sha(Path(image_path))
        vid = f"{doc_id}:{page}:{digest[:10]}"
        self.vectors[vid] = {
            "doc_id": doc_id, "page": int(page), "image_path": str(image_path),
            "vec": list(embedding or []),
        }
        self.added.append((doc_id, int(page), str(image_path)))
        return vid

    def remove_doc(self, doc_id):
        if self.fail_remove:
            raise RuntimeError("chroma offline")
        self.removed.append(doc_id)
        for vid in [v for v, e in self.vectors.items() if e["doc_id"] == doc_id]:
            del self.vectors[vid]
        return 1

    def count(self):
        return len(self.vectors)

    def query_embedding(self, embedding, k=5):
        scored = sorted(
            ((id_, _cosine_dist(e["vec"], embedding), e) for id_, e in self.vectors.items()),
            key=lambda t: t[1],
        )
        return [
            type("VH", (), {
                "doc_id": e["doc_id"], "page": e["page"],
                "image_path": e["image_path"], "id": vid, "distance": dist,
            })()
            for vid, dist, e in scored[:k]
        ]

    @staticmethod
    def _sha(path) -> str:
        import hashlib

        with Path(path).open("rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()

    # -- test-only simulation helpers -------------------------------------

    def drop_vector(self, doc_id, page):
        for vid, e in list(self.vectors.items()):
            if e["doc_id"] == doc_id and e["page"] == page:
                del self.vectors[vid]
                return vid
        return None

    def embed(self, image: "Image.Image") -> list[float]:
        self.embed_calls += 1
        # purely content-derived vector: identical pages -> identical vec
        # (distance 0), different pages -> clearly separated (distance > 0.1)
        px = image.convert("L").resize((8, 8)).tobytes()
        v = [b + 1.0 for b in px]
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]


def _cosine_dist(a, b):
    if not a or not b or len(a) != len(b):
        return 2.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return 1.0 - dot / (na * nb)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

def make_pdf(path: Path, pages: int = 2) -> Path:
    d = pymupdf.open()
    for i in range(pages):
        pg = d.new_page()
        if i == 1:
            # clearly distinct visual content (fills most of the page)
            pg.draw_rect(pymupdf.Rect(60, 60, 520, 560),
                         color=(0.1, 0.3, 0.9), fill=(0.1, 0.3, 0.9))
        pg.insert_text((72, 72),
                       f"Free cash flow = OCF - capex ({i})")
    d.save(str(path))
    d.close()
    return path


def build_env(tmp_path):
    store = DocumentStore(root=tmp_path / "data")
    src = make_pdf(tmp_path / "deck.pdf")
    rec = store.add(src)
    from app.parse import parse_file
    parse_file(store.corpus_path(rec.doc_id), rec.doc_id,
               store.artifacts_dir / rec.doc_id)
    backend = ChromaLikeBackend()
    state = VisualState(tmp_path / "vs.json")
    visual = VisualIndexer(backend, backend.embed, store, state=state)
    indexer = CorpusIndexer(store, embedder=FakeEmbedder(),
                            persist_dir=tmp_path / "db", visual=visual)
    return store, rec, indexer, backend


def manifest_pages(store, doc_id):
    m = store.artifacts_dir / doc_id / "manifest.json"
    return json.loads(m.read_text(encoding="utf-8")).get("pages", [])


# --------------------------------------------------------------------------
# A. state exists, vector lost -> detected + repaired (only the lost one)
# --------------------------------------------------------------------------

def test_lost_vector_detected_and_repaired(tmp_path):
    store, rec, indexer, backend = build_env(tmp_path)
    indexer.index_document(rec.doc_id)
    pages = manifest_pages(store, rec.doc_id)
    assert backend.count() == len(pages)

    # SIMULATE partial collection loss: page 1's vector vanishes while
    # the image + manifest + state stay unchanged
    backend.drop_vector(rec.doc_id, 1)
    assert backend.count() == len(pages) - 1

    embeds_before = backend.embed_calls
    stats = indexer.reconcile()           # normal restart flow
    assert stats.visual_repaired == 1     # exactly the lost page restored
    assert backend.count() == len(pages)
    # one verification embed for the lost page (present ones verified too:
    # page 2 embedded for verification + skipped, page 1 embedded + re-added)
    assert backend.embed_calls == embeds_before + len(pages)
    # metadata preserved: same doc/page/path as the manifest
    repaired = [e for e in backend.vectors.values()
                if e["doc_id"] == rec.doc_id and e["page"] == 1]
    assert len(repaired) == 1
    assert repaired[0]["image_path"] == pages[0]["image_path"]


def test_no_vector_loss_means_zero_embeddings(tmp_path):
    store, rec, indexer, backend = build_env(tmp_path)
    indexer.index_document(rec.doc_id)
    embeds = backend.embed_calls
    stats = indexer.reconcile()
    assert stats.visual_repaired == 0
    assert backend.embed_calls == embeds     # integrity gate passed cheaply
    assert any("already indexed" in s for s in stats.visual_skipped)


def test_repair_reports_backend_without_inspection_interface(tmp_path):
    """Documented limitation: a backend with no query_embedding/count gets
    a clear 'integrity check unavailable' report, never fake success."""
    store, rec, indexer, backend = build_env(tmp_path)
    indexer.index_document(rec.doc_id)
    backend.drop_vector(rec.doc_id, 1)

    class OpaqueBackend(ChromaLikeBackend):  # hide public inspection
        count = None
        query_embedding = None

    indexer.visual.backend = OpaqueBackend()
    indexer.visual.backend.vectors = backend.vectors
    indexer.visual.backend.embed_calls = backend.embed_calls
    stats = indexer.reconcile()
    assert any("integrity check unavailable" in f for f in stats.visual_failed)


# --------------------------------------------------------------------------
# B. failed deletion -> durable pending retry across restart
# --------------------------------------------------------------------------

def test_failed_visual_deletion_retried_after_restart(tmp_path):
    store, rec, indexer, backend = build_env(tmp_path)
    indexer.register_hooks()
    indexer.index_document(rec.doc_id)

    # initial failure: ChromaDB down while the document is removed
    backend.fail_remove = True
    store.remove(rec.doc_id)               # text purge must still succeed
    assert indexer.keyword.size == 0
    assert indexer.vector.stored_doc_ids() == set()
    assert backend.count() == 2             # stale vectors SURVIVED
    assert rec.doc_id in indexer.visual.state.pending_doc_ids()

    # restart simulation: fresh process, same persisted state file
    reloaded = VisualState(tmp_path / "vs.json")
    assert reloaded.pending_doc_ids() == [rec.doc_id]  # durable across restart
    visual2 = VisualIndexer(backend, backend.embed, store, state=reloaded)
    indexer2 = CorpusIndexer(store, embedder=FakeEmbedder(),
                             persist_dir=tmp_path / "db", visual=visual2)

    # service still down -> retry fails loudly, nothing silently dropped
    stats = indexer2.reconcile()
    assert any("pending visual purge still failing" in f
               for f in stats.visual_failed)
    assert rec.doc_id in reloaded.pending_doc_ids()
    assert backend.count() == 2

    # service recovers -> next run retries and completes the purge
    backend.fail_remove = False
    stats = indexer2.reconcile()
    assert rec.doc_id not in reloaded.pending_doc_ids()
    assert backend.count() == 0
    assert rec.doc_id not in reloaded.doc_ids()


def test_pending_purge_is_idempotent(tmp_path):
    store, rec, indexer, backend = build_env(tmp_path)
    indexer.register_hooks()
    indexer.index_document(rec.doc_id)
    backend.fail_remove = True
    store.remove(rec.doc_id)
    backend.fail_remove = False
    # two successful reconciles: second is a no-op, no crash, no re-add
    indexer.reconcile()
    indexer.reconcile()
    assert backend.count() == 0
    assert indexer.visual.state.pending_doc_ids() == []


def test_deleted_doc_never_returned_as_evidence(tmp_path):
    """Stale backend vectors for a deleted doc must never reach users."""
    store, rec, indexer, backend = build_env(tmp_path)
    indexer.register_hooks()
    indexer.index_document(rec.doc_id)
    backend.fail_remove = True
    store.remove(rec.doc_id)               # stale vectors remain in backend

    from app.retrieval import retrieve

    outcome = retrieve(
        "cash flow", keyword=KeywordIndex(), vector=None, store=store,
        visual_source=backend, text_to_visual_embedder=backend.embed,
        top_k=5,
    )
    assert all(i.doc_id != rec.doc_id for i in outcome.visual_items)
    assert all(i.doc_id != rec.doc_id for i in outcome.evidence.items)


# --------------------------------------------------------------------------
# C. interrupted indexing resumes; state corruption is surfaced
# --------------------------------------------------------------------------

def test_interrupted_visual_indexing_resumes(tmp_path):
    store, rec, indexer, backend = build_env(tmp_path)
    # fail the embed for the SECOND page only
    backend.add_fail_on_n = 1
    stats = indexer.index_document(rec.doc_id)
    assert stats.visual_pages == 1
    assert len(stats.visual_failed) == 1
    # state persisted only the successful page
    state = indexer.visual.state.get(rec.doc_id)
    assert set(state) == {"1"}

    # restart + repair: only page 2 is embedded, page 1 untouched
    reloaded = VisualState(tmp_path / "vs.json")
    indexer2 = CorpusIndexer(
        store, embedder=FakeEmbedder(), persist_dir=tmp_path / "db",
        visual=VisualIndexer(backend, backend.embed, store, state=reloaded),
    )
    backend.add_fail_on_n = None
    embeds = backend.embed_calls
    stats2 = indexer2.index_document(rec.doc_id)
    assert backend.embed_calls == embeds + 1      # exactly the missing page
    assert stats2.visual_pages == 1
    assert set(reloaded.get(rec.doc_id)) == {"1", "2"}


def test_corrupted_state_file_surfaces_hint(tmp_path):
    store, rec, indexer, backend = build_env(tmp_path)
    indexer.index_document(rec.doc_id)
    # corrupt the persisted state file
    (tmp_path / "vs.json").write_text("{not json", encoding="utf-8")
    state = VisualState(tmp_path / "vs.json")
    assert state.corrupted is True
    visual = VisualIndexer(backend, backend.embed, store, state=state)
    stats = CorpusIndexer(
        store, embedder=FakeEmbedder(), persist_dir=tmp_path / "db",
        visual=visual,
    ).reconcile()
    assert any("visual-backfill" in f for f in stats.visual_failed)


def test_legacy_state_file_still_read(tmp_path):
    store, rec, indexer, backend = build_env(tmp_path)
    indexer.index_document(rec.doc_id)
    old = json.loads((tmp_path / "vs.json").read_text(encoding="utf-8"))
    (tmp_path / "vs.json").write_text(
        json.dumps(old["docs"]), encoding="utf-8")  # legacy flat layout
    state = VisualState(tmp_path / "vs.json")
    assert state.corrupted is False
    assert set(state.get(rec.doc_id)) == {"1", "2"}


# --------------------------------------------------------------------------
# D. explicit --visual-backfill with no visual index -> honest non-zero exit
# --------------------------------------------------------------------------

def test_backfill_unavailable_exits_nonzero(tmp_path, monkeypatch, capsys):
    import app.visual as visual_mod

    monkeypatch.setattr(
        visual_mod, "production_visual_indexer",
        lambda store: (None, "app.visual_index not available (PR #11 unmerged)"),
    )
    import app.indexing as indexing_mod

    real_ds = DocumentStore

    class FakeDS(real_ds):
        def __init__(self, *a, **k):
            super().__init__(root=tmp_path / "ds")

    monkeypatch.setattr(indexing_mod, "DocumentStore", FakeDS)
    rc = indexing_main(["--visual-backfill"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "--visual-backfill" in err and "NOT performed" in err


# --------------------------------------------------------------------------
# E. IDENTITY check: twin (identical) images on different pages (Sec.2 of
#    the publication check) — distance alone must never prove presence
# --------------------------------------------------------------------------

def test_identical_pages_restored_by_identity_not_distance(tmp_path):
    """REGRESSION: two pages with IDENTICAL image bytes. Delete page 1's
    vector; the surviving twin is the nearest neighbour at distance ~0 —
    the OLD distance-based repair would call it 'present' and never
    restore page 1. The id-based check must restore page 1 WITHOUT
    damaging page 2."""
    store = DocumentStore(root=tmp_path / "data")
    # a 2-page PDF whose pages render IDENTICALLY (same bytes)
    d = pymupdf.open()
    for _ in range(2):
        pg = d.new_page()
        pg.insert_text((72, 72), "identical slide")
    d.save(str(tmp_path / "twin.pdf"))
    d.close()
    rec = store.add(tmp_path / "twin.pdf")
    from app.parse import parse_file
    parse_file(store.corpus_path(rec.doc_id), rec.doc_id,
               store.artifacts_dir / rec.doc_id)

    backend = ChromaLikeBackend()
    state = VisualState(tmp_path / "vs.json")
    visual = VisualIndexer(backend, backend.embed, store, state=state)
    indexer = CorpusIndexer(store, embedder=FakeEmbedder(),
                            persist_dir=tmp_path / "db", visual=visual)
    indexer.index_document(rec.doc_id)
    pages = manifest_pages(store, rec.doc_id)
    assert backend.count() == 2
    # both page images are byte-identical twins
    img1, img2 = (Path(pg["image_path"]) for pg in pages)
    assert img1.read_bytes() == img2.read_bytes()
    # ...yet their record ids embed the page number and differ
    vid1 = f"{rec.doc_id}:1:{ChromaLikeBackend._sha(img1)[:10]}"
    vid2 = f"{rec.doc_id}:2:{ChromaLikeBackend._sha(img2)[:10]}"
    assert vid1 != vid2
    assert vid1 in backend.vectors and vid2 in backend.vectors

    # delete page 1's vector; leave its state entry intact
    backend.drop_vector(rec.doc_id, 1)
    assert backend.count() == 1

    stats = indexer.reconcile()
    assert stats.visual_repaired == 1          # page 1 restored
    assert backend.count() == 2
    assert vid1 in backend.vectors             # the EXACT record came back
    assert vid2 in backend.vectors             # survivor untouched
    assert backend.vectors[vid2]["image_path"] == str(img2)



# --------------------------------------------------------------------------
# F. review finding 4: a CHANGED page must REPLACE its stale content-
#    addressed vector, never coexist with it — and failure must never
#    corrupt the old state (preflight -> evict -> re-add, per document)
# --------------------------------------------------------------------------

def _page_images(store, doc_id):
    m = json.loads((store.artifacts_dir / doc_id / "manifest.json")
                   .read_text(encoding="utf-8"))
    return {p["number"]: Path(p["image_path"]) for p in m["pages"]}


def test_changed_page_replaces_stale_vector(tmp_path):
    store, rec, indexer, backend = build_env(tmp_path)
    indexer.index_document(rec.doc_id)
    ids1 = {v: e["page"] for v, e in backend.vectors.items()}
    assert len(ids1) == 2 and set(ids1.values()) == {1, 2}
    page1_old = next(v for v, pg in ids1.items() if pg == 1)
    page2 = next(v for v, pg in ids1.items() if pg == 2)

    # mutate ONLY page 1's image bytes (new visual content, same path)
    img1 = _page_images(store, rec.doc_id)[1]
    d = pymupdf.open(); pg = d.new_page()
    pg.draw_rect(pymupdf.Rect(20, 20, 200, 200), color=(1, 0, 0),
                 fill=(1, 0, 0))
    pg.get_pixmap(dpi=72).save(str(img1)); d.close()

    stats = indexer.index_document(rec.doc_id)
    assert stats.visual_pages == 2
    ids2 = {v: e["page"] for v, e in backend.vectors.items()}
    assert len(ids2) == 2, "old page-1 vector must NOT survive the change"
    new_p1 = next(v for v, pg in ids2.items() if pg == 1)
    assert new_p1 != page1_old                    # stale id replaced
    assert page2 in ids2                          # unaffected page same id
    assert len(indexer.visual.state.get(rec.doc_id)) == 2


def test_changed_page_preflight_failure_keeps_old_vectors(tmp_path):
    store, rec, indexer, backend = build_env(tmp_path)
    indexer.index_document(rec.doc_id)
    ids1 = set(backend.vectors)
    img = _page_images(store, rec.doc_id)[1]
    img.write_bytes(b"not-an-image")
    stats = indexer.index_document(rec.doc_id)
    assert stats.visual_failed
    assert any("replacement aborted" in f for f in stats.visual_failed)
    assert set(backend.vectors) == ids1           # NOTHING evicted
    assert len(indexer.visual.state.get(rec.doc_id)) == 2  # state intact


def test_changed_page_eviction_failure_keeps_state_and_retries(tmp_path):
    store, rec, indexer, backend = build_env(tmp_path)
    indexer.index_document(rec.doc_id)
    ids1 = set(backend.vectors)
    img = _page_images(store, rec.doc_id)[1]
    d = pymupdf.open(); pg = d.new_page()
    pg.draw_rect(pymupdf.Rect(20, 20, 200, 200), color=(1, 0, 0),
                 fill=(1, 0, 0))
    pg.get_pixmap(dpi=72).save(str(img)); d.close()

    backend.fail_remove = True
    stats = indexer.index_document(rec.doc_id)
    assert any("eviction failed" in f for f in stats.visual_failed)
    assert set(backend.vectors) == ids1           # eviction failed: intact
    assert len(indexer.visual.state.get(rec.doc_id)) == 2

    backend.fail_remove = False                   # service back: retry works
    stats2 = indexer.index_document(rec.doc_id)
    assert not stats2.visual_failed
    assert len(backend.vectors) == 2
    assert len(indexer.visual.state.get(rec.doc_id)) == 2


# --------------------------------------------------------------------------
# G. review finding 5: equal COUNTs must not prove integrity — a missing
#    tracked page masked by an unrelated/foreign vector needs IDENTITY
#    verification (explicit audit; honest zero-network fast path)
# --------------------------------------------------------------------------

def test_equal_count_masking_detected_only_by_identity_audit(tmp_path):
    store, rec, indexer, backend = build_env(tmp_path)
    indexer.index_document(rec.doc_id)            # pages 1 and 2
    victim = backend.drop_vector(rec.doc_id, 1)   # tracked page LOST
    img = _page_images(store, rec.doc_id)[2]
    backend.add_image("unrelated-doc", 9, img,    # foreign vector fills gap
                      embedding=[1.0] + [0.0] * 63)

    # counts EQUAL (1 tracked + 1 foreign == state's expected 2) -> the
    # zero-network fast path cannot see the loss (documented limitation)
    stats = indexer.reconcile()
    assert backend.vectors.get(victim) is None

    # the explicit IDENTITY audit verifies every tracked page by its
    # expected (doc, page, digest) id and repairs only the missing one
    stats2 = indexer.reconcile(verify_ids=True)
    assert stats2.visual_repaired == 1
    assert backend.vectors.get(victim) is not None       # page 1 restored
    assert any(v["doc_id"] == "unrelated-doc"            # unrelated kept
               for v in backend.vectors.values())


def test_count_above_expected_triggers_identity_repair_not_deletion(tmp_path):
    store, rec, indexer, backend = build_env(tmp_path)
    indexer.index_document(rec.doc_id)            # pages 1,2 (state: 2)
    # reduce tracked state to page 1 only (as if page 2 were untracked)
    p1 = _page_images(store, rec.doc_id)[1]
    import app.visual as visual_mod
    st = visual_mod._file_stat(p1)
    digest = visual_mod._file_digest(p1)
    indexer.visual.state.drop_doc(rec.doc_id)
    indexer.visual.state.set_page(rec.doc_id, 1, st, digest)
    indexer.visual.state.save()
    img = _page_images(store, rec.doc_id)[2]
    backend.add_image("unrelated-doc", 9, img,
                      embedding=[1.0] + [0.0] * 63)
    # expected=1 vs actual=3 -> identity repair runs, re-adds NOTHING
    # missing (page 1 present), and never deletes unrelated entries
    stats = indexer.reconcile()
    assert stats.visual_repaired == 0
    assert any(v["doc_id"] == "unrelated-doc"
               for v in backend.vectors.values())
    assert len(backend.vectors) == 3


# --------------------------------------------------------------------------
# H. review findings 7 + 8: CLI --keyword-only --visual-backfill must fail
#    cleanly (no UnboundLocalError) and --doc --visual-backfill must be
#    SCOPED to that document only
# --------------------------------------------------------------------------

def test_keyword_only_visual_backfill_fails_cleanly(tmp_path, monkeypatch,
                                                    capsys):
    import app.indexing as indexing_mod

    class FakeDS(indexing_mod.DocumentStore):
        def __init__(self, *a, **k):
            super().__init__(root=tmp_path / "ds")

    monkeypatch.setattr(indexing_mod, "DocumentStore", FakeDS)
    rc = indexing_mod.main(["--keyword-only", "--visual-backfill"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "--visual-backfill" in err and "NOT performed" in err
    assert "UnboundLocalError" not in err


def test_scoped_doc_visual_backfill_touches_only_that_doc(tmp_path,
                                                          monkeypatch):
    store = DocumentStore(root=tmp_path / "ds")
    rec1 = store.add(make_pdf(tmp_path / "one.pdf", 2))
    from app.parse import parse_file
    parse_file(store.corpus_path(rec1.doc_id), rec1.doc_id,
               store.artifacts_dir / rec1.doc_id)
    rec2 = store.add(make_pdf(tmp_path / "two.pdf", 2))
    parse_file(store.corpus_path(rec2.doc_id), rec2.doc_id,
               store.artifacts_dir / rec2.doc_id)

    import app.visual as visual_mod
    import app.indexing as indexing_mod

    env = {}

    def fake_production(store_in):
        backend = ChromaLikeBackend()
        visual = VisualIndexer(backend, backend.embed, store_in,
                               state=VisualState(tmp_path / "vs.json"))
        env["backend"] = backend
        return visual, "ok"

    monkeypatch.setattr(visual_mod, "production_visual_indexer",
                        fake_production)

    class FakeDS(indexing_mod.DocumentStore):
        def __init__(self, *a, **k):
            super().__init__(root=tmp_path / "ds")

    monkeypatch.setattr(indexing_mod, "DocumentStore", FakeDS)

    n1 = len(manifest_pages(store, rec1.doc_id))
    n2 = len(manifest_pages(store, rec2.doc_id))

    # ingest doc1 only (text + visual): embeds doc1's pages
    rc = indexing_mod.main(["--doc", rec1.doc_id])
    assert rc == 0
    assert env["backend"].embed_calls == n1

    # SCOPED backfill: re-embeds ONLY doc1 (a fresh run, but the state
    # file from the ingest above matches, so the ingest itself re-embeds
    # nothing; doc2 must stay completely untouched)
    rc = indexing_mod.main(["--doc", rec1.doc_id, "--visual-backfill"])
    assert rc == 0
    assert env["backend"].embed_calls == n1

    # a plain --visual-backfill still covers the whole library
    rc2 = indexing_mod.main(["--visual-backfill"])
    assert rc2 == 0
    assert env["backend"].embed_calls == n1 + n2
