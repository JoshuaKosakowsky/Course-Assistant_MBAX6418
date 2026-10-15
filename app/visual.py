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
        self.corrupted = False
        self._lock = threading.Lock()
        if self.path.is_file():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(raw, dict) and "docs" in raw:
                    self._data = raw.get("docs", {}) or {}
                    self._pending = list(raw.get("pending", []) or [])
                else:
                    # legacy layout: the whole object WAS the docs mapping
                    self._data = raw or {}
                    self._pending = []
            except Exception:
                # unreadable state must never block startup; flag it so a
                # clear backfill hint can be shown instead of silent trust
                self._data = {}
                self._pending = []
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

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(
                    {"docs": self._data, "pending": self._pending},
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
        """Index every manifest page image for one document (incremental:
        pages already recorded for the same path are skipped)."""
        stats = IndexStats(doc_ids=1)
        try:
            doc = self.store.get(doc_id)
        except Exception as exc:
            stats.errors.append(f"{doc_id}: {exc}")
            return stats
        pages = self._manifest_pages(doc)
        if not pages:
            stats.visual_skipped.append(f"{doc_id}: no page images in manifest")
            return stats
        for page_no, image_path in pages:
            stats += self._index_one_page(doc_id, page_no, image_path)
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

    def reconcile(self, force: bool = False) -> IndexStats:
        """Bring the visual index in line with the library, page by page.

        Incremental rules (zero network for already-correct pages):
        0. RETRY failed deletions (durable pending queue) — see
           remove_document/_purge_doc: a purge that failed while ChromaDB
           was down stays queued until it succeeds, across restarts.
        1. Prune orphan state (deleted docs that predate the feature).
        2. INTEGRITY: a tracking entry is NOT proof the vector still
           exists — if the backend's total count is below the state's
           expected total (e.g. partial collection loss), the affected
           pages are verified via their own embedding and only genuinely
           missing ones are re-added (see repair_missing_vectors).
        3. Per-doc incremental pass (unchanged pages -> zero calls).

        ``force=True`` ignores state and re-embeds everything (the
        explicit backfill command).
        """
        stats = IndexStats()
        known = {d.doc_id for d in self.store.list()}

        # 0. retry failed deletions (durable across restarts)
        for doc_id in self.state.pending_doc_ids():
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
            has_count = callable(getattr(self.backend, "count", None))
            has_query = callable(getattr(self.backend, "query_embedding", None))
            if not (has_count and has_query):
                # documented limitation, surfaced — never invented success
                stats.visual_failed.append(
                    "integrity check unavailable: backend exposes no "
                    "count()/query_embedding() (documented limitation)"
                )
            else:
                expected = sum(
                    len(self.state.get(d)) for d in self.state.doc_ids()
                )
                try:
                    actual = int(self.backend.count())
                except Exception:
                    actual = -1
                if 0 <= actual < expected:
                    stats += self.repair_missing_vectors()
                elif self.state.corrupted and actual > 0:
                    stats.visual_failed.append(
                        "visual state file was unreadable but the backend "
                        "still has entries — run 'python -m app.indexing "
                        "--visual-backfill' to rebuild state safely "
                        "(content-addressed ids prevent duplicates)"
                    )

        for doc in self.store.list():
            if force:
                self._purge_doc(doc.doc_id, stats)
                stats += self.index_document(doc.doc_id)
                continue
            pages = self._manifest_pages(doc)
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
                self._purge_doc(doc.doc_id, stats)
                stats += self.index_document(doc.doc_id)
            else:
                # additive change (new pages only): no purge — the new
                # pages are embedded, already-indexed pages are skipped
                # with zero embedding calls
                stats += self.index_document(doc.doc_id)
        self.state.save()
        return stats

    # -- integrity repair (state may outlive a lost vector) -------------

    SELF_MATCH_EPSILON = 0.05  # cosine distance of a page vs its own vector

    def repair_missing_vectors(self) -> IndexStats:
        """Verify state-recorded pages against the backend and re-add the
        ones whose vectors are genuinely gone.

        Triggered ONLY when the backend's total entry count is below the
        state's expected total (a cheap aggregate gate that keeps normal
        restarts at zero embeddings). Verification uses the PUBLIC
        interface (embed the page -> query_embedding k=1): a hit at
        distance ~0 proves the exact content is indexed. Pages whose file
        changed (stat mismatch) are left to the normal incremental pass —
        they will be re-added there — and pages whose verification embed
        fails are reported, never guessed.

        LIMITATION (documented, not hidden): MSpanny's VisualIndex exposes
        no per-doc/per-page inspection (only count/query/add/remove), so
        one embedding call per checked page is required; this is also
        communicated to PR #11 as a candidate API addition. If the backend
        lacks query_embedding/count, the check is skipped and reported.
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
                    hits = self.backend.query_embedding(vec, k=1)
                except Exception as exc:
                    stats.visual_failed.append(
                        f"{doc.doc_id} p{page_no}: integrity verify failed: {exc}"
                    )
                    continue
                if hits and hits[0].distance <= self.SELF_MATCH_EPSILON:
                    continue  # present — no re-embed
                # vector lost: restore with the SAME doc/page/path metadata
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

    def _purge_doc(self, doc_id: str, stats: IndexStats) -> None:
        """Purge one doc from the visual backend + state (local op).

        On backend failure the tracking entry is NOT dropped: the doc is
        queued in the durable pending list so reconciliation retries the
        purge after a restart instead of abandoning stale vectors."""
        try:
            self.backend.remove_doc(doc_id)
            self.state.drop_doc(doc_id)
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
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        return [
            (int(p.get("number", 0)), str(p.get("image_path", "") or ""))
            for p in raw.get("pages", [])
        ]


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
