"""Production visual integration (Issue #4, Step 7) — offline.

Covers the application wiring WITHOUT MSpanny's unmerged branch: every
path uses injectable factories/backends, so a clean checkout exercises the
same code the production app will run once PR #11 merges. Startup must
never touch the network here (no real keys, no real services).
"""

from __future__ import annotations

import json
import io
import pymupdf
import pytest
from pathlib import Path

from app.documents import DocumentStore
from app.indexing import CorpusIndexer, IndexStats
from app.keyword import KeywordIndex
from app.vector import TextVectorIndex
from app.visual import (
    MAX_EMBED_EDGE,
    VisualIndexer,
    VisualState,
    resize_for_embedding,
)
from tests.fake_embeddings import FakeEmbedder
from PIL import Image

pytestmark = pytest.mark.smoke


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def make_pdf(path: Path, lines: list[str], pages: int = 2) -> Path:
    d = pymupdf.open()
    for p in range(pages):
        pg = d.new_page()
        y = 60
        for ln in lines:
            pg.insert_text((72, y), ln)
            y += 18
    d.save(str(path))
    d.close()
    return path


class FakeVisualBackend:
    """Duck-typed stand-in for MSpanny's VisualIndex (PR #11 API)."""

    def __init__(self):
        self.entries: dict[str, dict] = {}   # vid -> {doc_id, page, image_path}
        self.added: list[tuple] = []
        self.removed: list[str] = []
        self.fail_add = False
        self.fail_remove = False
        self.embed_calls = 0

    def add_image(self, doc_id, page, image_path, embedding=None):
        if self.fail_add:
            raise RuntimeError("embedding service down")
        vid = f"{doc_id}:{page}:{len(self.entries)}"
        self.entries[vid] = {
            "doc_id": doc_id, "page": int(page), "image_path": str(image_path),
        }
        self.added.append((doc_id, int(page), str(image_path)))
        return vid

    def remove_doc(self, doc_id):
        if self.fail_remove:
            raise RuntimeError("chroma offline")
        self.removed.append(doc_id)
        self.entries = {k: v for k, v in self.entries.items()
                        if v["doc_id"] != doc_id}
        return 1

    def count(self):
        return len(self.entries)


def fake_image_embedder(backend: FakeVisualBackend):
    def embed(image: Image.Image) -> list[float]:
        backend.embed_calls += 1
        # deterministic vector from image size + pixels (offline only)
        px = image.convert("L").resize((4, 4)).tobytes()
        return [b / 255.0 for b in px] + [0.25, 0.5, 0.75, 1.0]
    return embed


def build_env(tmp_path):
    """Store + parsed docs + text indexes + visual indexer (all offline)."""
    store = DocumentStore(root=tmp_path / "data")
    src = make_pdf(tmp_path / "deck.pdf", ["Free cash flow = OCF - capex"])
    rec = store.add(src)
    from app.parse import parse_file
    parse_file(store.corpus_path(rec.doc_id), rec.doc_id,
               store.artifacts_dir / rec.doc_id)
    kw = KeywordIndex()
    vec = TextVectorIndex(FakeEmbedder(), persist_dir=tmp_path / "db")
    backend = FakeVisualBackend()
    state = VisualState(tmp_path / "visual_state.json")
    visual = VisualIndexer(backend, fake_image_embedder(backend), store,
                           state=state)
    indexer = CorpusIndexer(store, embedder=FakeEmbedder(),
                            persist_dir=tmp_path / "db", visual=visual)
    return store, rec, indexer, backend


def count_pages(store, doc_id: str) -> int:
    manifest = store.artifacts_dir / doc_id / "manifest.json"
    return len(json.loads(manifest.read_text(encoding="utf-8")).get("pages", []))


# --------------------------------------------------------------------------
# 1+2. startup modes (text-only vs visual) — via _production_indexer DI
# --------------------------------------------------------------------------

def test_startup_with_visual_factory(tmp_path, monkeypatch, capsys):
    from app.ui import _production_indexer

    store = DocumentStore(root=tmp_path / "data")
    backend = FakeVisualBackend()

    def factory(store_):
        return (VisualIndexer(backend, fake_image_embedder(backend), store_,
                              state=VisualState(tmp_path / "vs.json")), None)

    _, indexer = _production_indexer(store, visual_factory=factory)
    assert indexer.visual is not None
    assert indexer.status()["visual"] == "enabled"
    assert "Text-only mode" not in capsys.readouterr().out


def test_startup_without_visual_module(tmp_path, monkeypatch, capsys):
    from app.ui import _production_indexer

    store = DocumentStore(root=tmp_path / "data")

    def factory(store_):
        return None, "app.visual_index not available (PR #11 unmerged)"

    _, indexer = _production_indexer(store, visual_factory=factory)
    assert indexer.visual is None
    assert indexer.status()["visual"] == "unavailable"
    out = capsys.readouterr().out
    assert "Text-only mode" in out and "PR #11" in out
    # keyword path still fully functional
    indexer.keyword = KeywordIndex()
    assert indexer.search("anything")["keyword"] == []


def test_startup_missing_visual_credentials(tmp_path, capsys):
    """ClassEmbeddingClient raises RuntimeError w/o keys -> visual disabled,
    startup completes in text-only mode."""
    import app.visual as visual_mod
    from app.ui import _production_indexer

    store = DocumentStore(root=tmp_path / "data")

    def factory(store_):
        mod = "app.visual_index"
        raise RuntimeError(
            f"Class embedding endpoint not configured for {mod}")

    _, indexer = _production_indexer(store, visual_factory=factory)
    assert indexer.visual is None
    out = capsys.readouterr().out
    assert "Text-only mode" in out and "not configured" in out


# --------------------------------------------------------------------------
# 3. upload -> three-index population
# --------------------------------------------------------------------------

def test_upload_indexes_three_indexes(tmp_path):
    store, rec, indexer, backend = build_env(tmp_path)
    stats = indexer.index_document(rec.doc_id)
    pages = count_pages(store, rec.doc_id)
    assert stats.chunks > 0 and stats.keyword_chunks > 0
    assert stats.vector_chunks == stats.keyword_chunks
    assert stats.visual_pages == pages            # every page image embedded
    assert stats.visual_failed == []
    assert len(backend.entries) == pages
    # metadata carries ORIGINAL paths + canonical doc id + parser page nums
    for entry in backend.entries.values():
        assert entry["doc_id"] == rec.doc_id
        p = Path(entry["image_path"])
        assert p.is_file() and "artifacts" in p.as_posix()
    assert indexer.status()["visual_pages"] == pages


def test_visual_failure_never_masks_text_success(tmp_path):
    store, rec, indexer, backend = build_env(tmp_path)
    backend.fail_add = True
    stats = indexer.index_document(rec.doc_id)
    # text indexes ARE populated...
    assert stats.keyword_chunks > 0 and stats.vector_chunks > 0
    # ...visual is honestly reported as failed, not claimed
    assert stats.visual_pages == 0
    assert len(stats.visual_failed) == count_pages(store, rec.doc_id)
    assert all("p" in f for f in stats.visual_failed)  # per-page record
    # backend kept working for text (no op), next attempt retriable
    backend.fail_add = False
    stats2 = indexer.index_document(rec.doc_id)
    assert stats2.visual_pages > 0


def test_upload_duplicate_not_reindexed(tmp_path):
    """Adding the same content twice (DocumentStore dedupe) must not create
    a second set of visual entries."""
    store, rec, indexer, backend = build_env(tmp_path)
    indexer.register_hooks()
    stats1 = indexer.index_document(rec.doc_id)
    before = len(backend.entries)
    # simulate the UI duplicate flow: same source -> same doc_id, no reindex
    from app.ui import add_file
    source = tmp_path / "deck.pdf"
    _, _, note = add_file(store, str(source), indexer)
    assert "already in the library" in note
    assert len(backend.entries) == before


# --------------------------------------------------------------------------
# 4. reconciliation / backfill
# --------------------------------------------------------------------------

def test_reconcile_skips_unchanged_and_backfills_missing(tmp_path):
    store, rec, indexer, backend = build_env(tmp_path)
    indexer.index_document(rec.doc_id)
    embeds = backend.embed_calls

    # second reconcile: unchanged -> ZERO embedding calls
    stats = indexer.reconcile()
    assert backend.embed_calls == embeds
    assert any("already indexed" in s for s in stats.visual_skipped)
    assert stats.visual_pages == 0

    # simulate a NEW page image added to the manifest (page 3) while 1-2 state kept
    manifest = store.artifacts_dir / rec.doc_id / "manifest.json"
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    img3 = store.artifacts_dir / rec.doc_id / "pages" / "page-003.png"
    make_pdf(tmp_path / "extra.pdf", ["x"])  # reuse renderer to create image
    d = pymupdf.open(); pg = d.new_page(); pg.draw_rect(pymupdf.Rect(10, 10, 60, 40),
                                                        color=(1, 0, 0), fill=(1, 0, 0))
    pg.get_pixmap(dpi=72).save(str(img3)); d.close()
    raw["pages"].append({"number": 3, "text": "", "image_path": str(img3),
                         "embedded_images": 0})
    manifest.write_text(json.dumps(raw), encoding="utf-8")

    embeds2 = backend.embed_calls
    stats2 = indexer.reconcile()
    assert backend.embed_calls == embeds2 + 1    # ONLY the new page embedded
    assert stats2.visual_pages == 1
    assert any(page == 3 for _, page, _ in backend.added[embeds2:])


def test_reconcile_never_backfills_fresh_docs_automatically(tmp_path):
    """Docs never seen by the visual pipeline are skipped by reconcile
    (explicit --visual-backfill only) => a restart does zero network."""
    store, rec, indexer, backend = build_env(tmp_path)
    # no index_document call: doc exists but visual state is empty
    # never backfills fresh docs automatically (explicit command only)
    stats = indexer.reconcile()
    assert backend.embed_calls == 0
    assert any("never visual-indexed" in s for s in stats.visual_skipped)
    # forced backfill indexes it
    stats2 = indexer.reconcile(visual_backfill=True)
    assert stats2.visual_pages == count_pages(store, rec.doc_id)


def test_reconcile_purges_orphan_state_and_changed_docs(tmp_path):
    store, rec, indexer, backend = build_env(tmp_path)
    indexer.index_document(rec.doc_id)
    # orphan state for a deleted-from-library doc
    indexer.visual.state.set_page("vanished-doc", 1, (123456, 99), "deadbeef")
    indexer.visual.state.save()
    stats = indexer.reconcile()
    assert "vanished-doc" in backend.removed
    # changed page set (delete page 2 from manifest) -> purge + re-add
    manifest = store.artifacts_dir / rec.doc_id / "manifest.json"
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw["pages"] = raw["pages"][:1]
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    stats2 = indexer.reconcile()
    assert stats2.visual_pages == 1
    assert rec.doc_id in backend.removed
    assert len(backend.entries) == 1


def test_reconcile_counts_corrupt_image_as_failed(tmp_path):
    store, rec, indexer, backend = build_env(tmp_path)
    indexer.index_document(rec.doc_id)
    good_pages = indexer.visual.state.get(rec.doc_id)
    assert len(good_pages) == count_pages(store, rec.doc_id)
    # corrupt ONE page image file
    manifest = store.artifacts_dir / rec.doc_id / "manifest.json"
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    img = Path(raw["pages"][0]["image_path"])
    img.write_bytes(b"not-an-image")
    # corrupting the image also changes the file -> doc is "changed",
    # so reconcile purges and re-adds; the corrupt page must FAIL honestly
    stats1 = indexer.reconcile()
    assert stats1.visual_failed, "corrupt page must be reported as failed"
    assert stats1.visual_pages == count_pages(store, rec.doc_id) - 1
    # the failed page is NOT recorded in state (retried next run)
    state_after = indexer.visual.state.get(rec.doc_id)
    assert len(state_after) == count_pages(store, rec.doc_id) - 1


# --------------------------------------------------------------------------
# 5. deletion matrix
# --------------------------------------------------------------------------

def test_deletion_purges_all_three_indexes(tmp_path):
    store, rec, indexer, backend = build_env(tmp_path)
    indexer.register_hooks()
    indexer.index_document(rec.doc_id)
    store.remove(rec.doc_id)
    assert len(backend.entries) == 0
    assert rec.doc_id in backend.removed
    assert indexer.vector.stored_doc_ids() == set()
    # the doc cannot reappear after restart: reconcile rebuild + prune
    stats = indexer.reconcile()
    assert rec.doc_id not in {
        v["doc_id"] for v in backend.entries.values()
    }


def test_deletion_when_visual_service_down(tmp_path):
    """A failed visual purge must NOT pretend success: the tracking entry
    stays and the purge is queued for retry (durable across restarts)."""
    store, rec, indexer, backend = build_env(tmp_path)
    indexer.register_hooks()
    indexer.index_document(rec.doc_id)
    backend.fail_remove = True
    store.remove(rec.doc_id)          # must not raise
    assert indexer.keyword.size == 0
    assert indexer.vector.stored_doc_ids() == set()
    # honest failure state: still tracked + queued (NOT silently dropped)
    assert rec.doc_id in indexer.visual.state.doc_ids()
    assert rec.doc_id in indexer.visual.state.pending_doc_ids()
    # recovery on a later run when ChromaDB is back
    backend.fail_remove = False
    indexer.reconcile()
    assert backend.count() == 0
    assert rec.doc_id not in indexer.visual.state.pending_doc_ids()
    assert rec.doc_id not in indexer.visual.state.doc_ids()


def test_deletion_when_visual_module_absent(tmp_path):
    store, rec, indexer, backend = build_env(tmp_path)
    indexer.visual = None             # text-only configuration
    indexer.register_hooks()
    indexer.index_document(rec.doc_id)
    store.remove(rec.doc_id)
    assert indexer.keyword.size == 0
    assert indexer.vector.stored_doc_ids() == set()


def test_deletion_when_incomplete_indexing(tmp_path):
    store, rec, indexer, backend = build_env(tmp_path)
    indexer.register_hooks()
    backend.fail_add = True
    indexer.index_document(rec.doc_id)   # text ok, visual failed
    backend.fail_add = False
    store.remove(rec.doc_id)             # partial state must not survive
    assert rec.doc_id not in indexer.visual.state.doc_ids()


def test_no_duplicate_hook_registrations(tmp_path):
    store, rec, indexer, backend = build_env(tmp_path)
    indexer.register_hooks()
    indexer.register_hooks()             # second call must be a no-op
    indexer.index_document(rec.doc_id)
    store.remove(rec.doc_id)
    assert backend.removed.count(rec.doc_id) == 1


# --------------------------------------------------------------------------
# 6. retrieval service wires everything end-to-end (offline)
# --------------------------------------------------------------------------

class StubReranker:
    def __init__(self, n=3):
        self.calls = 0
        self._n = n

    def rank(self, query, documents, top_n=None):
        self.calls += 1
        return {i: (len(documents) - i) / len(documents)
                for i in range(len(documents))}


def test_retrieval_service_end_to_end(tmp_path):
    from app.retrieval import RetrievalService
    store, rec, indexer, backend = build_env(tmp_path)
    indexer.index_document(rec.doc_id)

    service = RetrievalService(
        keyword=indexer.keyword, vector=indexer.vector, store=store,
        visual_backend=backend,
        reranker_factory=lambda: StubReranker(),
    )
    outcome = service.retrieve("free cash flow", top_k=3)
    assert outcome.stages["keyword"] == "ok"
    assert outcome.stages["rerank"] == "ok"
    assert outcome.evidence.items
    top = outcome.evidence.items[0]
    assert top.filename == "deck.pdf"


def test_retrieval_service_reranker_down_degrades(tmp_path):
    from app.retrieval import RetrievalService
    store, rec, indexer, backend = build_env(tmp_path)
    indexer.index_document(rec.doc_id)

    def broken_factory():
        raise RuntimeError("9004 unreachable")

    service = RetrievalService(
        keyword=indexer.keyword, vector=indexer.vector, store=store,
        visual_backend=backend, reranker_factory=broken_factory,
    )
    outcome = service.retrieve("free cash flow")
    assert outcome.stages.get("rerank") == "degraded"
    assert outcome.evidence.items          # text evidence still returned


def test_build_retrieval_service_from_indexer(tmp_path):
    from app.retrieval import build_retrieval_service
    store, rec, indexer, backend = build_env(tmp_path)
    indexer.index_document(rec.doc_id)
    service = build_retrieval_service(
        indexer, reranker_factory=lambda: StubReranker())
    assert service.visual_backend is backend
    outcome = service.retrieve("cash flow")
    assert outcome.evidence.items


# --------------------------------------------------------------------------
# 7. embedding-copy sizing (512px, originals untouched)
# --------------------------------------------------------------------------

def test_resize_for_embedding_caps_edge_and_keeps_original(tmp_path):
    big = Image.new("RGB", (1200, 1554), "white")
    src = tmp_path / "page.png"
    big.save(src)
    before = src.stat().st_size
    with Image.open(src) as im:
        copy = resize_for_embedding(im)
    assert max(copy.size) == MAX_EMBED_EDGE
    assert src.stat().st_size == before          # original untouched
    with Image.open(src) as im2:
        assert im2.size == (1200, 1554)          # file unmodified


def test_resize_passthrough_when_small():
    small = Image.new("RGB", (300, 200), "black")
    assert resize_for_embedding(small) is small


def test_visual_indexer_embeds_resized_copy_only(tmp_path):
    """The backend receives the ORIGINAL path; only the embedding vector
    comes from a <=512px copy (verified via a recorder embedder)."""
    store, rec, indexer, backend = build_env(tmp_path)
    sizes = []

    def spy_embed(image):
        sizes.append(max(image.size))
        return fake_image_embedder(backend)(image)

    indexer.visual.image_embedder = spy_embed
    indexer.index_document(rec.doc_id)
    assert sizes, "embedder never called"
    assert all(s <= MAX_EMBED_EDGE for s in sizes)
    assert all(Path(e).is_file() for _, _, e in backend.added)
    with Image.open(backend.added[0][2]) as orig:
        assert max(orig.size) > MAX_EMBED_EDGE  # original kept full-res
