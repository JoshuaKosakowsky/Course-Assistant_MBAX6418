"""Text chunking for retrieval (Issue #2 / Phase 2).

Splits page-level extracted text into stable retrieval chunks that
carry source metadata, never mixing content from different pages or
documents. The input accepts the team's ingestion shapes without
importing them (duck-typed): ``app.parse.PageRecord`` (``.number``,
``.text``, ``.image_path``), ``PageText``, or plain dicts, so this
module stays usable before PR #10 merges.

Rationale for the character-based ``langchain-text-splitters``
``RecursiveCharacterTextSplitter``:

- Deterministic and offline (no tokenizer dependency).
- Default separators (blank line, newline, space, character) suit
  lecture/slide prose.
- Defaults: ``chunk_size=512`` chars (~100-150 tokens for typical
  English material) and ``chunk_overlap=64`` chars (~12-15 tokens) to
  keep sentence-level continuity across boundaries. These values are
  deliberately modest and will be compared (250/500/1000) in the Phase 6
  design comparison.

Chunk identifiers are deterministic: ``<doc_id>::pNNNN::cNNN`` where
``doc_id`` is content-addressed (sha-256 prefix from the team's
document store), so unchanged inputs yield unchanged chunk ids, and any
change of content changes the id namespace.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64


@dataclass(frozen=True)
class PageText:
    """One page/slide of extracted text.

    ``image_path`` is optional; when present it is carried onto every
    chunk so future citations can show the original page/slide image.
    """

    page_no: int
    text: str
    image_path: Optional[str] = None


@dataclass(frozen=True)
class Chunk:
    """A retrieval unit with complete source provenance."""

    chunk_id: str
    doc_id: str
    filename: str
    format: str
    page_no: int
    index_in_page: int
    text: str
    image_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "filename": self.filename,
            "format": self.format,
            "page_no": self.page_no,
            "index_in_page": self.index_in_page,
            "text": self.text,
            "image_path": self.image_path,
        }


def validate_settings(chunk_size: int, chunk_overlap: int) -> None:
    """Reject unusable chunking configuration with a clear message."""
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError(f"chunk_size must be a positive integer, got {chunk_size!r}")
    if (
        not isinstance(chunk_overlap, int)
        or isinstance(chunk_overlap, bool)
        or chunk_overlap < 0
    ):
        raise ValueError(f"chunk_overlap must be a non-negative integer, got {chunk_overlap!r}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size "
            f"({chunk_size})"
        )


def _normalize_page(page: Any) -> PageText:
    """Accept PageRecord-like objects, PageText, or dicts."""
    if isinstance(page, Mapping):
        page_no = page.get("page_no", page.get("number"))
        text = page.get("text", "")
        image_path = page.get("image_path")
    else:
        page_no = getattr(page, "page_no", None)
        if page_no is None:
            page_no = getattr(page, "number", None)
        text = getattr(page, "text", "") or ""
        image_path = getattr(page, "image_path", None)
    if page_no is None:
        raise ValueError(
            "each page input must carry a page number (page_no or number)"
        )
    return PageText(
        page_no=int(page_no),
        text=str(text),
        image_path=str(image_path) if image_path is not None else None,
    )


def chunk_id(doc_id: str, page_no: int, index_in_page: int) -> str:
    return f"{doc_id}::p{int(page_no):04d}::c{int(index_in_page):03d}"


def chunk_pages(
    doc_id: str,
    filename: str,
    pages: Sequence[Any],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    format: str = "",
) -> List[Chunk]:
    """Chunk one document's pages into retrieval chunks.

    Splitting happens per page, so a chunk never spans pages; each
    call handles exactly one document, so chunks never mix documents.
    Pages with empty/whitespace-only text produce no chunks. Short
    pages yield a single chunk when they fit within chunk_size.
    """
    validate_settings(chunk_size, chunk_overlap)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks: List[Chunk] = []
    for page in pages:
        p = _normalize_page(page)
        text = p.text.strip()
        if not text:
            continue
        index_in_page = 0
        for piece in splitter.split_text(text):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                Chunk(
                    chunk_id=chunk_id(doc_id, p.page_no, index_in_page),
                    doc_id=doc_id,
                    filename=filename,
                    format=format,
                    page_no=p.page_no,
                    index_in_page=index_in_page,
                    text=piece,
                    image_path=p.image_path,
                )
            )
            index_in_page += 1
    return chunks
