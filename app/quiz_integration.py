"""Quiz boundary for PR #15-style evidence and Issue #7-style UI handlers.

Duck-typed on purpose: do not import unmerged retrieval or UI branches.
The adapter trusts only *already verified* image provenance; the eventual
production caller must first validate the underlying manifest/source.
"""
from __future__ import annotations

from app.quiz import EvidenceRef


def evidence_from_pack(pack: object) -> tuple[EvidenceRef, ...]:
    """Convert PR #15 EvidencePack.items into quiz evidence, fail closed.

    The unique ``doc_id`` from each item is preserved for document-scoped
    quiz selection; ``document`` keeps the filename as display metadata.
    Items without a valid ``doc_id`` convert to ``None`` and can never
    match a document selection (fails closed).

    Image-only pages are retained only when their original image is verified.
    Question generation still requires an injected generator and validates its
    selected evidence; this conversion cannot prove factual answer support.
    """
    items = getattr(pack, "items", None)
    if not isinstance(items, (list, tuple)):
        raise ValueError("evidence pack must expose items")
    refs = []
    for item in items:
        filename = getattr(item, "filename", None)
        page = getattr(item, "page_no", None)
        excerpt = getattr(item, "excerpt", None)
        image_path = getattr(item, "image_path", None)
        verified = getattr(item, "image_verified", False) is True
        doc_id = getattr(item, "doc_id", None)
        if not isinstance(filename, str) or not filename.strip():
            continue
        if type(page) is not int or page < 1:
            continue
        if not isinstance(doc_id, str) or not doc_id.strip():
            doc_id = None
        if not isinstance(excerpt, str) or not excerpt.strip():
            excerpt = None
        if not verified or not isinstance(image_path, str) or not image_path.strip():
            image_path = None
        if excerpt is None and image_path is None:
            continue
        refs.append(EvidenceRef(filename, page, excerpt, image_path, doc_id))
    return tuple(refs)
