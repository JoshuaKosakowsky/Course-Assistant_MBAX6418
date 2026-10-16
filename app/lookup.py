"""Read-only, offline question lookup over the current library's PDF text.

Returns escaped HTML containing verbatim excerpts, not generated answers.
No file ingestion, deletion, semantic service, or model call happens here.
"""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING

from .documents import DocumentStore

if TYPE_CHECKING:
    from .indexing import CorpusIndexer


DEFAULT_RESULTS = 5


def lookup_question(
    store: DocumentStore,
    question: str,
    indexer: CorpusIndexer | None = None,
) -> str:
    """Find up to five BM25 passages restricted to currently registered PDFs."""
    if not isinstance(question, str) or not question.strip():
        return '<p>Enter a question to look up in the Library PDFs.</p>'
    if len(question) > 2000:
        return '<p>Please limit your question to 2,000 characters.</p>'
    try:
        documents = {doc.doc_id: doc for doc in store.list() if doc.format == '.pdf'}
        if not documents:
            return '<p>Add a PDF to the Library before using Lookup.</p>'
        if indexer is None:
            return '<p>Lookup is unavailable because the search index is not initialized. Restart the app.</p>'
        hits = indexer.keyword.search(
            question.strip(), top_k=DEFAULT_RESULTS, doc_ids=list(documents)
        )
        # Recheck membership after retrieval so concurrent removals cannot leak stale hits.
        current_ids = {doc.doc_id for doc in store.list() if doc.format == '.pdf'}
        hits = [hit for hit in hits if hit.chunk.doc_id in documents
                and hit.chunk.doc_id in current_ids]
        if not hits:
            return '<p>No matching passages found. Try specific terms from your PDFs. Scanned pages need searchable text.</p>'
        parts = ['<p><strong>Matching passages</strong> — keyword search, not a generated answer.</p>']
        for number, hit in enumerate(hits, 1):
            chunk = hit.chunk
            filename = escape(documents[chunk.doc_id].filename)
            excerpt = escape(chunk.text)
            parts.append(
                f'<article><h3>{number}. {filename} · Page {chunk.page_no}</h3>'
                f'<pre style="white-space:pre-wrap;overflow-wrap:anywhere;font-family:inherit">'
                f'{excerpt}</pre></article>'
            )
        return '\n'.join(parts)
    except Exception:
        # UI boundary: never expose exception messages, paths, keys, or PDF content.
        return '<p>Lookup is temporarily unavailable. Please try again or restart the app.</p>'
