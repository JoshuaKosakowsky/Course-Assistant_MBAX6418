"""Optional visual-index wiring for the production pipeline (Issue #4, Step 7).

Bridges OUR retrieval/indexing code to MSpanny's ``app.visual_index``
(PR #11) WITHOUT importing it until it actually exists: the module is
locate d lazily via importlib, every seam is duck-typed, and a factory is
injectable so offline tests never need the unmerged branch.

Responsibilities split (team plan):
- Issue #3 (PR #11): ``VisualIndex`` storage, ``ClassEmbeddingClient``,
  content-addressed ids, ``remove_doc()``.
- Issue #4 (this module): optional discovery, 512px embed-copy resizing,
  page-level state tracking (avoids re-embedding on restart), upload-time
  indexing, incremental backfill, deletion purging, retrieval wiring.

Production guarantees:
- Never substitutes fake embedding models.
- Resizes ONLY the working copy sent for embedding; the original
  full-resolution image file and its source path are preserved.
- When the visual module, credentials, or service are unavailable, the
  application continues in text-only mode with an explicit reason.
- Startup makes no unnecessary network requests: reconcile is incremental
  and only touches pages whose state record is missing or stale.
"""

from __future__ import annotations

import importlib.util
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from app import config
from app.indexing import IndexStats  # reuse the shared stats carrier

try:  # pillow is a project dependency (requirements.txt); import at top
    from PIL import Image
except Exception:  # pragma: no cover - requirements pin pillow
    Image = None  # type: ignore

MAX_EMBED_EDGE = 512  # px on the longest side (verified 9003 context limit)

# Reason strings surfaced in startup messages and tests.
VISUAL_UNAVAILABLE = "app.visual_index not available (PR #11 unmerged)"
EXCERPT_REQUIRED_NOTE = (
    "PR #14's Source requires a non-empty 'excerpt'; image-only pages keep "
    "their image in the evidence pack but are excluded from Source output "
    "rather than fabricating text. Future PR #14 revision should allow "
    "Source(excerpt=None) for visual-only evidence."
)


# --------------------------------------------------------------------------
# Feature detection (no hard import)
# --------------------------------------------------------------------------

_FIND_LOCK = threading.Lock()
_FOUND_MODULE: Optional[object] = None


def _find_visual_module():
    """Return MSpanny's app.visual_index module once it exists, else None.

    Cached: on a clean checkout the module simply does not exist; after
    PR #11 merges (and this branch is rebased onto it) the module becomes
    importable and production wiring activates automatically.
    """
    global _FOUND_MODULE
    with _FIND_LOCK:
        if _FOUND_MODULE is not None:
            return _FOUND_MODULE
        if importlib.util.find_spec("app.visual_index") is None:
            return None
        import app.visual_index as mod  # type: ignore

        _FOUND_MODULE = mod
        return mod


# --------------------------------------------------------------------------
# Embedding copy sizing
# --------------------------------------------------------------------------

def resize_for_embedding(image: Image.Image, max_edge: int = MAX_EMBED_EDGE) -> Image.Image:
    """Return a working copy resized so its longest edge <= max_edge.

    The ORIGINAL is never mutated; when already small enough the image is
    returned as-is (read-only use, no copy needed). Verified live on port
    9003: a full 150-dpi page render exceeds the model's 8192-token
    context (HTTP 400) while a ~512px copy embeds successfully.
    """
    if max(image.size) <= max_edge:
        return image
    width, height = image.size
    scale = max_edge / max(width, height)
    return image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        resample=Image.LANCZOS,
    )


# --------------------------------------------------------------------------
# Page-level state (persisted, so a restart cannot duplicate or re-embed)
# --------------------------------------------------------------------------

DEFAULT_STATE_PATH = config.DATA_ROOT / "indexes" / "visual_state.json"


class VisualState:
    """Persistent record of which (doc, page) images are already embedded.

    File layout (new format): {"docs": {doc_id: {page: {"m","s","d"}}},
    "pending": [doc_id, ...]}. The stat snapshot lets a restart skip
    unchanged pages with ZERO file hashing/network; a content change (size
    or mtime moved) triggers re-embedding; a corrupt/missing file fails to
    stat and is honestly reported as failed rather than treated as indexed.

    ``pending`` is the durable retry queue for document purges whose
    backend call failed (e.g. ChromaDB offline at deletion time): entries
    stay there until a remove_doc succeeds, so stale vectors can be
    retried after a restart instead of being silently abandoned.
    """

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else DEFAULT_STATE_PATH
        self._data: Dict[str, Dict[str, Dict[str, object]]] = {}
        self._pending: List[str] = []
        self._retry: Dict[str, bool] = {}
        self.corrupted = False
        self._lock = threading.Lock()
        if self.path.is_file():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(raw, dict) and "docs" in raw:
                    self._data = raw.get("docs", {}) or {}
                    self._pending = list(raw.get("pending", []) or [])
                    self._retry = dict(raw.get("retry", {}) or {})
                else:
                    # legacy layout: the whole object WAS the docs mapping
                    self._data = raw or {}
                    self._pending = []
                    self._retry = {}
            except Exception:
                # unreadable state must never block startup; flag it so a
                # clear backfill hint can be shown instead of silent trust
                self._data = {}
                self._pending = []
                self._retry = {}
                self.corrupted = True

    def get(self, doc_id: str) -> Dict[str, Dict[str, object]]:
        return dict(self._data.get(doc_id, {}))

    def has_page(self, doc_id: str, page: int, stat: object | None) -> bool:
        """True when the stored record matches the CURRENT file stat."""
        entry = self._data.get(doc_id, {}).get(str(page))
        if not entry or stat is None:
            return False
        return entry.get("m") == stat[0] and entry.get("s") == stat[1]

    def doc_ids(self) -> list[str]:
        return list(self._data)

    def set_page(self, doc_id: str, page: int, stat: object, digest: str) -> None:
        with self._lock:
            self._data.setdefault(doc_id, {})[str(page)] = {
                "m": stat[0], "s": stat[1], "d": digest,
            }

    def drop_doc(self, doc_id: str) -> None:
        with self._lock:
            self._data.pop(doc_id, None)
            self._retry.pop(doc_id, None)

    def drop_page(self, doc_id: str, page: int) -> None:
        """Remove ONE page's tracking entry (used after a post-eviction
        re-add failure: the old entry is no longer authoritative because
        the vector is gone, and a later run must re-add the page even
        though the image file's stat has not changed)."""
        with self._lock:
            data = self._data.get(doc_id)
            if data is not None:
                data.pop(str(page), None)
                if not data:
                    self._data.pop(doc_id, None)

    # -- durable pending removals (failed visual purges) -----------------

    def add_pending(self, doc_id: str) -> None:
        with self._lock:
            if doc_id not in self._pending:
                self._pending.append(doc_id)

    def drop_pending(self, doc_id: str) -> None:
        with self._lock:
            if doc_id in self._pending:
                self._pending.remove(doc_id)

    def pending_doc_ids(self) -> list[str]:
        with self._lock:
            return list(self._pending)

    # -- durable replacement-retry marker ----------------------------------
    # A doc-level replacement that EVICTED the vectors but whose re-adds
    # ALL failed ends with an empty tracking record — indistinguishable
    # from a doc that was never visual-indexed. The marker keeps such a
    # doc retryable by ordinary reconcile() (including after a restart)
    # instead of stranding it until a manual --visual-backfill.

    def mark_replacement_failed(self, doc_id: str) -> None:
        with self._lock:
            self._retry[doc_id] = True

    def clear_replacement_failed(self, doc_id: str) -> None:
        with self._lock:
            self._retry.pop(doc_id, None)

    def replacement_failed(self, doc_id: str) -> bool:
        with self._lock:
            return bool(self._retry.get(doc_id))

    def replacement_failed_docs(self) -> list[str]:
        with self._lock:
            return list(self._retry)

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(
                    {"docs": self._data, "pending": self._pending,
                     "retry": self._retry},
                    sort_keys=True, indent=1,
                ),
                encoding="utf-8",
            )
            tmp.replace(self.path)


def _file_stat(path: Path):
    """(mtime_ns, size_bytes) for a page image, or None if unreadable."""
    try:
        st = path.stat()
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def _file_digest(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Indexer adapter (ours; drives the duck-typed VisualIndex backend)
# --------------------------------------------------------------------------

class VisualIndexer:
    """Indexes parsed page images into the visual backend (Issue #4 side).

    ``backend`` is any object exposing ``add_image(doc_id, page,
    image_path, embedding=None)``, ``remove_doc(doc_id)`` and
    ``count()`` — i.e. MSpanny's VisualIndex or an offline stub.
    ``image_embedder`` is a callable PIL.Image -> vector; production uses
    the class service (never a fake), tests inject fakes.
    """

    def __init__(
        self,
        backend,
        image_embedder: Callable,
        store,
        state: VisualState | None = None,
    ):
        self.backend = backend
        self.image_embedder = image_embedder
        self.store = store
        self.state = state or VisualState()

    # -- per-document ----------------------------------------------------

    def index_document(self, doc_id: str) -> IndexStats:
        """Index every manifest page image for one document.

        - UNCHANGED page (state matches the file stat) -> skipped, zero calls.
        - NEW page -> embedded and added (unique content-addressed id).
        - CHANGED page (a tracked page whose image bytes changed) -> the
          old content-addressed vector must NOT survive alongside the new
          one. PR #11 exposes no per-page removal, so this becomes a
          controlled DOCUMENT-LEVEL replacement (``replace_document``):
          all replacement embeddings are preflighted first, and only then
          is the document evicted and re-added. Any preflight failure
          aborts with the old vectors and state untouched.
        """
        stats = IndexStats(doc_ids=1)
        try:
            doc = self.store.get(doc_id)
        except Exception as exc:
            stats.errors.append(f"{doc_id}: {exc}")
            return stats
        pages = self._manifest_pages(doc)
        problem = self._manifest_problem(doc)
        if problem is not None:
            # unreadable metadata is NOT 'zero pages': preserve vectors
            # and retryable state, never purge on a transient read failure
            stats.visual_failed.append(problem)
            return stats
        if not pages:
            stats.visual_skipped.append(f"{doc_id}: no page images in manifest")
            # never abandon searchable old vectors: purge them (durably
            # queued for retry if the backend is down) before dropping
            # the tracking state
            self._purge_doc(doc_id, stats)
            return stats
        if self._has_changed_page(doc_id, pages):
            return stats + self.replace_document(doc_id)
        prior_pages = stats.visual_pages
        prior_failed = len(stats.visual_failed)
        for page_no, image_path in pages:
            stats += self._index_one_page(doc_id, page_no, image_path)
        if stats.visual_pages > prior_pages:
            # at least one page landed: stale removal/retry markers moot
            self.state.clear_replacement_failed(doc_id)
            self.state.drop_pending(doc_id)
        elif not self.state.get(doc_id) and \
                len(stats.visual_failed) > prior_failed:
            # an indexing attempt was made and EVERY page add failed; the
            # doc has no vectors and no tracking record — keep it
            # retryable so a later reconcile re-adds it instead of
            # treating it as never visual-indexed
            self.state.mark_replacement_failed(doc_id)
        self.state.save()
        return stats

    def _has_changed_page(self, doc_id: str, pages) -> bool:
        """True when any TRACKED page's image file changed or vanished."""
        state = self.state.get(doc_id)
        if not state:
            return False
        for page_no, image_path in pages:
            entry = state.get(str(page_no))
            if entry is None:
                continue  # a NEW page is strictly additive
            if not _stat_matches(entry, Path(image_path)):
                return True
        return False

    def replace_document(self, doc_id: str) -> IndexStats:
        """Controlled document-level visual replacement (review finding 4).

        Used when a tracked page's image changed (and by the explicit
        backfill / ``--doc --visual-backfill`` workflows). Order:

        1. PREFLIGHT: embed every manifest page BEFORE any destructive
           step. Any missing/corrupt/embedding failure aborts here: the
           old vectors and the state file stay intact (retryable).
        2. EVICT only this document's old vectors (never the library).
        3. RE-ADD every page with the preflighted embeddings. On a
           non-atomic tail failure, state is recorded ONLY for pages that
           were actually added, so a later run completes the replacement
           without ever pretending success.
        """
        stats = IndexStats(doc_ids=1)
        try:
            doc = self.store.get(doc_id)
        except Exception as exc:
            stats.errors.append(f"{doc_id}: {exc}")
            return stats
        pages = self._manifest_pages(doc)
        problem = self._manifest_problem(doc)
        if problem is not None:
            # never evict/degrade based on temporarily unreadable metadata
            stats.visual_failed.append(problem)
            return stats
        if not pages:
            stats.visual_skipped.append(f"{doc_id}: no page images in manifest")
            # purge stale vectors; on backend failure the tracking entry
            # stays (durable pending queue) instead of silently leaving
            # searchable vectors behind
            self._purge_doc(doc_id, stats)
            return stats

        # 1. preflight all replacement embeddings (no mutation yet)
        ready: Dict[int, tuple] = {}
        for page_no, image_path in pages:
            stat = _file_stat(Path(image_path))
            if stat is None or not image_path:
                stats.visual_failed.append(
                    f"{doc_id} p{page_no}: image missing — replacement "
                    f"aborted, old vectors kept"
                )
                self.state.save()
                return stats
            try:
                with Image.open(image_path) as im:
                    vec = self.image_embedder(resize_for_embedding(im))
            except Exception as exc:
                stats.visual_failed.append(
                    f"{doc_id} p{page_no}: {exc} — replacement aborted, "
                    f"old vectors kept"
                )
                self.state.save()
                return stats
            ready[page_no] = (image_path, stat, vec)

        # 2. evict ONLY this document's old vectors
        try:
            self.backend.remove_doc(doc_id)
        except Exception as exc:
            stats.visual_failed.append(
                f"{doc_id}: eviction failed ({exc}) — old vectors kept"
            )
            self.state.save()
            return stats

        # 3. re-add every page; state ONLY for successes; a failed page's
        #    OLD entry is dropped (post-eviction it is no longer backed by
        #    a vector) so the next run re-adds it even though its image
        #    file's stat has not changed — retryable, never a silent skip
        added = 0
        for page_no, (image_path, stat, vec) in ready.items():
            try:
                self.backend.add_image(doc_id, page_no, image_path, embedding=vec)
                self.state.set_page(doc_id, page_no, stat,
                                    _file_digest(Path(image_path)))
                added += 1
            except Exception as exc:
                stats.visual_failed.append(f"{doc_id} p{page_no}: {exc}")
                self.state.drop_page(doc_id, page_no)
        stats.visual_pages = added
        stats.visual_added = added
        if added == len(ready):
            # full success: never retry again; a stale pending-removal
            # marker is moot (ids are content-addressed, so the vectors
            # just written ARE this document's content)
            self.state.clear_replacement_failed(doc_id)
            self.state.drop_pending(doc_id)
        elif added == 0:
            # eviction succeeded but NOTHING came back: without a durable
            # marker the doc would look 'never visual-indexed' and be
            # stranded until a manual backfill — mark it retryable and
            # never claim any page as indexed (guarantee preserved)
            self.state.mark_replacement_failed(doc_id)
        else:
            # partial: dropped entries retry additively via the normal
            # pass; any stale retry marker is no longer needed
            self.state.clear_replacement_failed(doc_id)
            self.state.drop_pending(doc_id)
        self.state.save()
        return stats

    def _index_one_page(self, doc_id: str, page_no: int, image_path: str) -> IndexStats:
        stats = IndexStats()
        stat = _file_stat(Path(image_path))
        if self.state.has_page(doc_id, page_no, stat) and stat is not None:
            stats.visual_skipped.append(f"{doc_id} p{page_no}: already indexed")
            return stats
        if not image_path or stat is None:
            stats.visual_failed.append(f"{doc_id} p{page_no}: image missing")
            return stats
        try:
            with Image.open(image_path) as im:
                vec = self.image_embedder(resize_for_embedding(im))
            # Original path is what goes into metadata; only the embedding
            # copy was resized (add_image stores image_path, we supply the
            # embedding directly, so MSpanny's code never resizes).
            self.backend.add_image(doc_id, page_no, image_path, embedding=vec)
            self.state.set_page(doc_id, page_no, stat, _file_digest(Path(image_path)))
            stats.visual_pages = 1
            stats.visual_added = 1
        except Exception as exc:
            stats.visual_failed.append(f"{doc_id} p{page_no}: {exc}")
        return stats

    # -- library-wide reconcile (incremental; safe) ----------------------

    def reconcile(self, force: bool = False, verify_ids: bool = False) -> IndexStats:
        """Bring the visual index in line with the library, page by page.

        Incremental rules (zero network for already-correct pages):
        0. RETRY failed deletions (durable pending queue) — see
           remove_document/_purge_doc: a purge that failed while ChromaDB
           was down stays queued until it succeeds, across restarts.
        1. Prune orphan state (deleted docs that predate the feature).
        2. INTEGRITY: a tracking entry is NOT proof the vector still
           exists. Every reconcile verifies membership by expected IDENTITY:
           the expected content-addressed ids are recomputed from state
           ({doc}:{page}:{digest[:10]}) and compared against a read-only,
           LOCAL enumeration of the backend's current ids — no embedding
           calls are needed for detection, and an unrelated/foreign vector
           can NEVER mask a missing tracked page (equal counts are never
           trusted). Missing pages are re-added by the same identity.
           Backends without any enumeration capability get an explicit
           limitation report instead of a pretended guarantee; for those,
           a count mismatch (either direction) still triggers the
           query-based repair, and ``verify_ids=True`` audits every page.
        3. Per-doc incremental pass (unchanged pages -> zero calls);
           changed pages and explicit forced backfill use the safe
           document-level replacement (replace_document).

        ``force=True`` ignores state and re-embeds everything (the
        explicit backfill command).
        """
        stats = IndexStats()
        known = {d.doc_id for d in self.store.list()}

        # 0. retry failed deletions (durable across restarts)
        for doc_id in self.state.pending_doc_ids():
            if doc_id in known:
                # the document is BACK in the library: because vector ids
                # are content-addressed, its current vectors ARE the
                # re-added content — the pending removal is stale and
                # must never purge freshly indexed vectors
                self.state.drop_pending(doc_id)
                continue
            try:
                self.backend.remove_doc(doc_id)
                self.state.drop_pending(doc_id)
                if doc_id not in known:
                    # orphan doc: fully forgotten once purged
                    self.state.drop_doc(doc_id)
            except Exception as exc:
                stats.visual_failed.append(
                    f"{doc_id}: pending visual purge still failing: {exc}"
                )

        # 1. prune orphan state (deleted docs that predate the feature);
        #    _purge_doc keeps failures in the pending queue instead of
        #    silently dropping the tracking entry
        for doc_id in self.state.doc_ids():
            if doc_id not in known:
                self._purge_doc(doc_id, stats)

        # 2. integrity: state entries must not be trusted blindly
        if not force:
            if verify_ids:
                # deep audit: every tracked page verified by its OWN
                # embedding and id (for backends without enumeration)
                stats += self.repair_missing_vectors()
            else:
                actual_ids = self._enumerate_backend_ids()
                if actual_ids is not None:
                    # read-only local membership check: the expected
                    # content-addressed ids recomputed from state must all
                    # be present. Equal counts are never treated as proof;
                    # unrelated/foreign ids are never deleted.
                    missing = self._expected_ids() - actual_ids
                    if missing:
                        stats += self._repair_ids(missing)
                    if self.state.corrupted and actual_ids:
                        stats.visual_failed.append(
                            "visual state file was unreadable but the "
                            "backend still has entries — run 'python -m "
                            "app.indexing --visual-backfill' to rebuild "
                            "state safely (content-addressed ids prevent "
                            "duplicates)"
                        )
                else:
                    has_count = callable(getattr(self.backend, "count", None))
                    has_query = callable(
                        getattr(self.backend, "query_embedding", None))
                    if not (has_count and has_query):
                        # documented limitation, surfaced — never invented
                        # success
                        stats.visual_failed.append(
                            "integrity check unavailable: backend exposes "
                            "no count()/query_embedding() nor id "
                            "enumeration (documented limitation)"
                        )
                    else:
                        expected = sum(
                            len(self.state.get(d))
                            for d in self.state.doc_ids()
                        )
                        try:
                            actual = int(self.backend.count())
                        except Exception:
                            actual = -1
                        if actual >= 0 and actual != expected:
                            # either direction of mismatch may hide missing
                            # or foreign/unrelated vectors — verify by
                            # identity (query-based; enumeration absent)
                            stats += self.repair_missing_vectors()
                        elif self.state.corrupted and actual > 0:
                            stats.visual_failed.append(
                                "visual state file was unreadable but the "
                                "backend still has entries — run 'python -m "
                                "app.indexing --visual-backfill' to rebuild "
                                "state safely (content-addressed ids prevent "
                                "duplicates)"
                            )
                        else:
                            stats.visual_failed.append(
                                "visual membership enumeration unavailable "
                                "on this backend: equal-count masking is "
                                "not detectable without "
                                "reconcile(verify_ids=True) or "
                                "--visual-backfill (documented limitation)"
                            )

        for doc in self.store.list():
            if force:
                stats += self.replace_document(doc.doc_id)
                continue
            if self.state.replacement_failed(doc.doc_id):
                # an earlier replacement EVICTED the vectors but every
                # re-add failed (tracking state now empty): retry the
                # doc-level replacement — this is NOT a never
                # visual-indexed document
                stats += self.replace_document(doc.doc_id)
                continue
            pages = self._manifest_pages(doc)
            problem = self._manifest_problem(doc)
            if problem is not None:
                # metadata temporarily unavailable: preserve vectors/state
                # and let a later run continue (never a destructive purge)
                stats.visual_failed.append(problem)
                continue
            state = self.state.get(doc.doc_id)
            if not state:
                stats.visual_skipped.append(
                    f"{doc.doc_id}: never visual-indexed (run --visual-backfill)"
                )
                continue
            state_pages = set(state)
            manifest_pages = {str(p) for p, _ in pages}
            # NEW pages are additive (no state entry yet) — they only
            # trigger an embed. STALE means pages REMOVED from the
            # manifest, or an EXISTING state page whose file stat moved
            # (corrupt/replaced image) -> purge + re-add that doc.
            stale = bool(state_pages - manifest_pages) or any(
                str(p) in state and not _stat_matches(state.get(str(p)), Path(img))
                for p, img in pages
            )
            if stale:
                # changed/removed pages: safe doc-level replacement (old
                # vectors stay until the replacement is fully preflighted)
                stats += self.replace_document(doc.doc_id)
            else:
                # additive change (new pages only): no purge — the new
                # pages are embedded, already-indexed pages are skipped
                # with zero embedding calls
                stats += self.index_document(doc.doc_id)
        self.state.save()
        return stats

    # -- integrity repair (state may outlive a lost vector) -------------

    def repair_missing_vectors(self) -> IndexStats:
        """Verify state-recorded pages against the backend and re-add the
        ones whose vectors are genuinely gone.

        Triggered ONLY when the backend's total entry count is below the
        state's expected total (a cheap aggregate gate that keeps normal
        restarts at zero embeddings).

        IDENTITY, not distance: two pages can share a near-identical
        image, so a low nearest-neighbour distance does NOT prove the
        specific (doc, page) record exists. MSpanny's VisualIndex returns
        a content-addressed ``VisualHit.id`` of the form
        ``{doc_id}:{page}:{sha256(image_bytes)[:10]}`` — exactly the
        digest our state records — so the expected id is recomputed from
        state and a page is 'present' ONLY when a returned hit carries
        that exact id. A deleted record's id can never be matched by a
        surviving twin page (the id embeds the page number), so the twin
        is never mistaken for proof, and re-adding the lost record never
        damages the survivor.

        If the backend returns hits without an ``id`` attribute the
        record cannot be verified by identity; the SAFEST recovery is
        then a re-add (idempotent content-addressed upsert — restores a
        missing record and leaves a present one byte-identical), which is
        what happens. Documented limitation: identical-content pages are
        only distinguishable by their ids.

        Pages whose file changed (stat mismatch) are left to the normal
        incremental pass, and verify failures are reported, never guessed.
        """
        stats = IndexStats()
        if not callable(getattr(self.backend, "query_embedding", None)) or \
                not callable(getattr(self.backend, "count", None)):
            stats.visual_failed.append(
                "integrity check unavailable: backend exposes no "
                "count()/query_embedding() (documented limitation)"
            )
            return stats
        for doc in self.store.list():
            for page_no, image_path in self._manifest_pages(doc):
                entry = self.state.get(doc.doc_id).get(str(page_no))
                if entry is None:
                    continue  # never-seen page: normal pass handles it
                stat = _file_stat(Path(image_path))
                if stat is None or (
                        entry.get("m") == stat[0] and entry.get("s") == stat[1]
                ) is False:
                    continue  # changed/missing file: normal pass handles it
                try:
                    with Image.open(image_path) as im:
                        vec = self.image_embedder(resize_for_embedding(im))
                    hits = self.backend.query_embedding(vec, k=2)
                except Exception as exc:
                    stats.visual_failed.append(
                        f"{doc.doc_id} p{page_no}: integrity verify failed: {exc}"
                    )
                    continue
                digest = str(entry.get("d", "") or "")
                expected_id = f"{doc.doc_id}:{page_no}:{digest[:10]}"
                ids = [getattr(h, "id", None) for h in hits]
                present = expected_id in ids
                if present:
                    continue  # the exact (doc, page) record is indexed
                if ids and all(i is None for i in ids):
                    # ids unavailable: cannot verify identity -> safest
                    # recovery is an idempotent re-add (never harmful)
                    pass
                # vector record lost (or identities unverifiable): restore
                # with the SAME doc/page/path metadata
                try:
                    self.backend.add_image(
                        doc.doc_id, page_no, image_path, embedding=vec
                    )
                    stats.visual_repaired += 1
                except Exception as exc:
                    stats.visual_failed.append(
                        f"{doc.doc_id} p{page_no}: repair re-add failed: {exc}"
                    )
        self.state.save()
        return stats

    # -- identity helpers (finding 5: equal counts never prove integrity) --

    def _expected_ids(self) -> set:
        """Content-addressed ids the state claims are indexed, recomputed
        offline: {doc_id}:{page}:{sha256(image bytes)[:10]} — exactly the
        format the PR #11 backend derives at add time (VisualHit.id)."""
        expected = set()
        for doc_id in self.state.doc_ids():
            for page_no, entry in self.state.get(doc_id).items():
                digest = str(entry.get("d", "") or "")
                if digest:
                    expected.add(f"{doc_id}:{page_no}:{digest[:10]}")
        return expected

    def _enumerate_backend_ids(self) -> Optional[set]:
        """Read-only LOCAL enumeration of the backend's current ids.

        Preference order: an explicit ids capability (``get_ids()`` /
        ``list_ids()`` / ``ids``), then a ChromaDB-collection-shaped
        attribute (public ``collection`` or the private ``_collection`` of
        the PR #11 class) via read-only ``Collection.get(limit=..., ...)``
        pagination. Never mutates and never calls the embedding service.
        Returns None when NO capability exists — callers must then report
        the limitation instead of trusting counts.
        """
        backend = self.backend
        for name in ("get_ids", "list_ids"):
            fn = getattr(backend, name, None)
            if callable(fn):
                try:
                    return set(str(i) for i in fn())
                except Exception:
                    return None
        raw = getattr(backend, "ids", None)
        if raw is not None:
            try:
                return set(str(i) for i in (raw() if callable(raw) else raw))
            except Exception:
                return None
        for name in ("collection", "_collection"):
            coll = getattr(backend, name, None)
            g = getattr(coll, "get", None)
            if coll is not None and callable(g):
                try:
                    ids: list = []
                    offset = 0
                    step = 512
                    while True:
                        page = g(limit=step, offset=offset)
                        pids = page.get("ids") or []
                        ids.extend(str(i) for i in pids)
                        if len(pids) < step:
                            break
                        offset += len(pids)
                    return set(ids)
                except Exception:
                    return None
        return None

    def _repair_ids(self, missing: set) -> IndexStats:
        """Re-add pages whose expected content-addressed id is absent from
        the enumerated backend ids (identity-based, no distance heuristics).
        Pages no longer present in the manifest are left to the normal
        incremental pass (they will be purged/replaced there), never
        reported as repair failures."""
        stats = IndexStats()
        for vid in sorted(missing):
            parts = vid.split(":")
            if len(parts) != 3:
                continue
            doc_id, page_no, _ = parts[0], parts[1], parts[2]
            try:
                page_no = int(page_no)
            except ValueError:
                continue
            try:
                doc = self.store.get(doc_id)
            except Exception:
                continue  # unknown doc: orphan state pruned elsewhere
            try:
                image_path = dict(self._manifest_pages(doc)).get(page_no)
            except Exception:
                continue
            if image_path is None:
                continue  # page removed from manifest: normal pass purges
            try:
                with Image.open(image_path) as im:
                    vec = self.image_embedder(resize_for_embedding(im))
                self.backend.add_image(
                    doc_id, page_no, image_path, embedding=vec
                )
                self.state.set_page(
                    doc_id, page_no, _file_stat(Path(image_path)),
                    _file_digest(Path(image_path)),
                )
                stats.visual_repaired += 1
            except Exception as exc:
                stats.visual_failed.append(
                    f"{doc_id} p{page_no}: repair re-add failed: {exc}"
                )
        self.state.save()
        return stats

    def _purge_doc(self, doc_id: str, stats: IndexStats) -> None:
        """Purge one doc from the visual backend + state (local op).

        On backend failure the tracking entry is NOT dropped: the doc is
        queued in the durable pending list so reconciliation retries the
        purge after a restart instead of abandoning stale vectors."""
        try:
            self.backend.remove_doc(doc_id)
            self.state.drop_doc(doc_id)
            self.state.drop_pending(doc_id)
        except Exception as exc:
            self.state.add_pending(doc_id)
            stats.visual_failed.append(f"{doc_id}: visual purge failed (queued for retry): {exc}")
        self.state.save()

    # -- deletion hook target -------------------------------------------

    def remove_document(self, doc_id: str) -> None:
        """Removal-hook target: purge visual entries + state. Local-only
        (ChromaDB delete), so it works while the embedding service is
        down; on failure the doc is queued in the persisted pending list
        and reconcile retries the purge on the next run — stale vectors
        can never be abandoned silently. The text indexes are purged by
        CorpusIndexer regardless."""
        try:
            self.backend.remove_doc(doc_id)
            self.state.drop_doc(doc_id)
            self.state.drop_pending(doc_id)
        except Exception as exc:
            self.state.add_pending(doc_id)
            print(f"warning: visual purge failed for {doc_id} "
                  f"(queued for retry): {exc}")
        self.state.save()

    def count(self) -> int:
        try:
            return int(self.backend.count())
        except Exception:
            return 0

    # -- helpers ----------------------------------------------------------

    def _manifest_pages(self, doc) -> List[Tuple[int, str]]:
        manifest = self.store.artifacts_dir / doc.doc_id / "manifest.json"
        if not manifest.is_file():
            return []
        try:
            raw = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            return []
        pages = raw.get("pages", []) if isinstance(raw, dict) else []
        out: List[Tuple[int, str]] = []
        for p in pages:
            try:
                out.append((int(p["number"]), str(p["image_path"])))
            except (KeyError, TypeError, ValueError):
                continue
        return out

    def _manifest_problem(self, doc) -> Optional[str]:
        """None when the manifest is valid (with any page count), else a
        reason. A MISSING or UNREADABLE manifest is NOT proof that a
        document intentionally has zero pages — callers must preserve
        existing vectors and retryable state in that case and never purge
        solely because metadata is temporarily unavailable."""
        manifest = self.store.artifacts_dir / doc.doc_id / "manifest.json"
        if not manifest.is_file():
            return (
                f"{doc.doc_id}: manifest missing — metadata temporarily "
                f"unavailable; existing visual vectors preserved (no purge)"
            )
        try:
            raw = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            return (
                f"{doc.doc_id}: manifest corrupt/unreadable — existing "
                f"visual vectors preserved (no purge)"
            )
        if not isinstance(raw, dict) or "pages" not in raw:
            return (
                f"{doc.doc_id}: manifest has no 'pages' record — existing "
                f"visual vectors preserved (no purge)"
            )
        return None


def _stat_matches(entry: object | None, path: Path) -> bool:
    """Cheap equality of stored stat snapshot vs current file (no hash)."""
    if not entry:
        return False
    cur = _file_stat(path)
    if cur is None:
        return False
    return entry.get("m") == cur[0] and entry.get("s") == cur[1]


# --------------------------------------------------------------------------
# Production factory (never fake; degrades loudly to text-only)
# --------------------------------------------------------------------------

def production_visual_indexer(store, *, state_path=None):
    """Build the production VisualIndexer, or return (None, reason).

    Requires PR #11's module (feature-detected), class credentials, and —
    only on first use — the live service. Startup itself makes no network
    calls: client construction is local; embedding happens during indexing.
    """
    mod = _find_visual_module()
    if mod is None:
        return None, VISUAL_UNAVAILABLE
    try:
        backend = mod.VisualIndex(root=str(config.DATA_ROOT / "indexes" / "visual"))
        client = mod.ClassEmbeddingClient()  # raises RuntimeError w/o creds

        def image_embedder(image: Image.Image) -> List[float]:
            return client.embed_image(resize_for_embedding(image))

        indexer = VisualIndexer(backend, image_embedder, store, state=VisualState(state_path))
        print(
            f"Visual search: page images -> {client.base_url} "
            f"(model: {config.VISUAL_EMBED_MODEL or 'unset'})"
        )
        return indexer, None
    except Exception as exc:
        return None, f"visual index unavailable: {exc}"
