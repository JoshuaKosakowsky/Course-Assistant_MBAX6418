"""Visual embedding index over page/slide images (Phase 2, issue #3).

A chromadb-backed store, deliberately SEPARATE from the text index built in
issue #2: this module owns its own persistent collection (``visual``) and
namespace. Embedders are injected so offline tests use synthetic vectors
while production uses the class endpoint client (app.visual_embeddings).

Metadata recorded per image: source document id, page number, and the
image path, so Phase 3 can build evidence packs that point back at the
original page/slide image.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .visual_embeddings import ClassEmbeddingClient, ImageEmbedder, TextEmbedder, fake_embedder
from .config import DATA_ROOT

logger = logging.getLogger(__name__)

COLLECTION_NAME = "visual"
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass(frozen=True)
class VisualHit:
    """One retrieved page/slide image with its supporting metadata."""

    doc_id: str
    page: int
    image_path: str
    distance: float
    id: str


class VisualIndex:
    """Chromadb-backed image-embedding store.

    ``root`` selects the persistent directory (gitignored); the default is
    ``DATA_ROOT/indexes/visual`` — a separate namespace from the text index.
    """

    def __init__(
        self,
        root: Path | str | None = None,
        embedder: ImageEmbedder | None = None,
        text_embedder: TextEmbedder | None = None,
    ):
        import chromadb
        from chromadb.config import Settings

        self.root = Path(root) if root is not None else DATA_ROOT / "indexes" / "visual"
        self.root.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self.root), settings=Settings(anonymized_telemetry=False)
        )
        self._collection = self._client.get_or_create_collection(name=COLLECTION_NAME)
        self._embedder = embedder
        self._text_embedder = text_embedder

    # -- write path ----------------------------------------------------------

    def add_image(
        self,
        doc_id: str,
        page: int,
        image_path: str | Path,
        embedding: list[float] | None = None,
    ) -> str:
        """Index one page/slide image — one current vector per (doc, page).

        The vector id is the stable ``(doc_id, page)`` slot (not a content
        hash), so re-indexing a changed page *replaces* that page's previous
        embedding and stale versions never remain searchable. Re-indexing
        identical content is a no-op upsert bearing the same id. The
        embedding is computed BEFORE any store write, so an embedding or
        validation failure leaves previously valid vectors untouched.

        Returns the slot id (``doc_id:page``).
        """
        image_path = Path(image_path)
        if embedding is None:
            embedding = self._get_embedder()(Image.open(image_path))
        vid = _slot_id(doc_id, page)
        self._collection.upsert(
            ids=[vid],
            embeddings=[embedding],
            metadatas=[
                {
                    "doc_id": doc_id,
                    "page": int(page),
                    # POSIX form keeps metadata portable across OSes; Path()
                    # re-parses forward slashes correctly on Windows too.
                    "image_path": image_path.as_posix(),
                }
            ],
        )
        logger.info("Indexed %s (page %s) as %s", image_path, page, vid)
        return vid

    def index_images_in_directory(
        self, image_dir: str | Path, doc_id: str | None = None, pages_from_filename: bool = True
    ) -> int:
        """Index every supported image in a directory; returns the count added."""
        directory = Path(image_dir)
        if doc_id is None:
            doc_id = directory.name
        images = sorted(p for p in directory.iterdir() if p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES)
        count = 0
        for i, image_path in enumerate(images):
            page = _page_from_filename(image_path) if pages_from_filename else i + 1
            self.add_image(doc_id, page, image_path)
            count += 1
        logger.info("Indexed %d/%d images for document %s", count, len(images), doc_id)
        return count

    # -- query path ----------------------------------------------------------

    def query_image(self, image: str | Path | Image.Image, k: int = 5) -> list[VisualHit]:
        """Image-to-image search: embed the image, then query the index."""
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        return self.query_embedding(self._get_embedder()(image), k=k)

    def query_text(self, text: str, k: int = 5, text_embedder: TextEmbedder | None = None) -> list[VisualHit]:
        """Text-to-image bridge: embed free text, then search the visual index.

        The default embedder is the SAME visual model/endpoint used for
        images (Qwen3-VL, port 9003). Mixing in the separate port-9002 text
        embedding space would return nonsense hits even at equal dimensions.
        """
        embed_text = text_embedder or self._text_embedder
        if embed_text is None:
            embed_text = ClassEmbeddingClient().embed_text_visual
        return self.query_embedding(embed_text(text), k=k)

    def query_embedding(self, embedding: list[float], k: int = 5) -> list[VisualHit]:
        if self._collection.count() == 0:
            return []
        k = max(1, min(int(k), self._collection.count()))
        res = self._collection.query(
            query_embeddings=[embedding], n_results=k, include=["metadatas", "distances"]
        )
        hits: list[VisualHit] = []
        for vid, meta, dist in zip(
            res.get("ids", [[]])[0], res.get("metadatas", [[]])[0], res.get("distances", [[]])[0]
        ):
            hits.append(
                VisualHit(
                    doc_id=meta["doc_id"],
                    page=int(meta["page"]),
                    image_path=meta["image_path"],
                    distance=float(dist),
                    id=vid,
                )
            )
        return hits

    # -- maintenance ---------------------------------------------------------

    def remove_doc(self, doc_id: str) -> int:
        """Purge every entry for one document; registers as an ingest-removal hook."""
        found = self._collection.get(where={"doc_id": doc_id})
        ids = found.get("ids", [])
        if ids:
            self._collection.delete(ids=ids)
        logger.info("Removed %d entries for document %s", len(ids), doc_id)
        return len(ids)

    def count(self) -> int:
        return self._collection.count()

    def _get_embedder(self) -> ImageEmbedder:
        if self._embedder is not None:
            return self._embedder
        return ClassEmbeddingClient().embed_image


def _slot_id(doc_id: str, page: int) -> str:
    """Stable per-(doc_id, page) slot id.

    Re-indexing a page replaces its vector (same id) instead of stacking a
    second version, so stale embeddings of changed pages are never returned.
    """
    return f"{doc_id}:{page}"


def _page_from_filename(path: Path) -> int:
    """First integer run in the filename stem, else 1 (e.g. slide_12.png -> 12)."""
    import re

    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else 1
