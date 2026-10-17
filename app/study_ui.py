"""Production study-service wiring and a safe, document-scoped Q&A UI."""
from __future__ import annotations

from html import escape

from app import config
from app.qna import MISSING_INFO_FALLBACK, QnaOrchestrator
from app.qna_schema import Source
from app.quiz_integration import verified_original_image


def _normalized_excerpt(text):
    return " ".join((text or "").split()).casefold()


def production_study_services(indexer):
    """Construct clients without network calls, after index reconciliation.

    No credentials or placeholder settings produce an unavailable model,
    never a fake answer generator. Retrieval remains shared by both tabs.
    """
    from app.retrieval import build_retrieval_service
    from app.quiz_model import QuizVisionClient, VisionQuizGenerator
    from app.quiz_provider import QuizProvider
    from app.quiz_replay import QuizReplayProvider
    from app.visual_support import VisualSupportVerifier

    retrieval = build_retrieval_service(indexer)
    settings = (config.VISION_BASE_URL, config.VISION_API_KEY, config.VISION_MODEL)
    if any(not value or any(marker in value.casefold() for marker in
                           ("dummy", "replace-with", "your-key", "your-model"))
           for value in settings):
        return retrieval, None, None
    try:
        client = QuizVisionClient()
        qna = QnaOrchestrator(client, visual_verifier=VisualSupportVerifier(client))
        quiz = QuizReplayProvider(
            indexer.store, QuizProvider(retrieval, VisionQuizGenerator(client))
        )
        return retrieval, qna, quiz
    except Exception:
        return retrieval, None, None


class QnaController:
    def __init__(self, store, retrieval=None, qna=None):
        self.store = store
        self.retrieval = retrieval
        self.qna = qna

    def ask(self, materials, question):
        """Return status, escaped answer/source HTML, and verified originals."""
        if not isinstance(question, str) or not question.strip():
            return "Enter a question first.", "", []
        if len(question) > 2000:
            return "Limit your question to 2,000 characters.", "", []
        if (not isinstance(materials, (list, tuple)) or not materials or
                any(not isinstance(doc_id, str) for doc_id in materials)):
            return "Select at least one course document.", "", []
        if self.retrieval is None or self.qna is None:
            return "Q&A is unavailable. Configure the server-side vision model and restart the app.", "", []
        try:
            current = {doc.doc_id for doc in self.store.list()}
            if not set(materials).issubset(current):
                return "Selected materials are no longer available. Refresh your selection.", "", []
            outcome = self.retrieval.retrieve(
                question.strip(), doc_ids=tuple(materials), top_k=5, include_images=True
            )
            # Selection and library membership are checked again at the UI boundary.
            current = {doc.doc_id for doc in self.store.list()}
            if not set(materials).issubset(current):
                return "Materials changed while retrieving. Please ask again.", "", []
            items = [item for item in outcome.evidence.items
                     if item.doc_id in materials and item.doc_id in current]
            sources = []
            for item in items:
                image = self._image(item)
                excerpt = item.excerpt if item.excerpt and item.excerpt.strip() else None
                if excerpt or image:
                    sources.append(Source(item.filename, item.page_no, excerpt, image))
            if not sources:
                return "No supporting evidence found.", f"<p>{escape(MISSING_INFO_FALLBACK)}</p>", []
            result = self.qna.answer(question.strip(), tuple(sources))
            current = {doc.doc_id for doc in self.store.list()}
            if not set(materials).issubset(current):
                return "Materials changed while answering. Please ask again.", "", []
            if not result.grounded or not result.sources:
                return "No supported answer found.", f"<p>{escape(MISSING_INFO_FALLBACK)}</p>", []
            html = ["<h3>Answer</h3>", f"<p>{escape(result.answer).replace(chr(10), '<br>')}</p>"]
            images = []
            for number, source in enumerate(result.sources, 1):
                matches = [item for item in items
                           if item.filename == source.document and item.page_no == source.page
                           and ((source.excerpt and _normalized_excerpt(source.excerpt) in _normalized_excerpt(item.excerpt))
                                or (source.excerpt is None and source.image_path == self._image(item)
                                    and source.image_path is not None))]
                if not matches:
                    return "Unable to verify answer sources. Please try again.", "", []
                html.append(
                    f'<article id="answer-source-{number}"><h4>Source {number}: '
                    f'{escape(source.document)} · Page/slide {source.page}</h4>'
                    f'<blockquote>{escape(source.excerpt) if source.excerpt else "Visual evidence: see the original image below."}</blockquote></article>'
                )
                for item in matches:
                    image = self._image(item)
                    if image and image not in [entry[0] for entry in images]:
                        images.append((image, f"Source {number}: {item.filename} · Page/slide {item.page_no}"))
            stages = getattr(outcome, "stages", {})
            degraded = [name for name in ("semantic", "visual", "rerank")
                        if stages.get(name) in {"degraded", "skipped"}]
            status = "Answer checked against retrieved evidence."
            if degraded:
                status += " Limited retrieval: " + ", ".join(degraded) + "."
            return status, "\n".join(html), images
        except Exception:
            return "Q&A is temporarily unavailable. Please try again or check the server configuration.", "", []

    def _image(self, item):
        if item.image_verified is not True:
            return None
        return verified_original_image(self.store, {
            "doc_id": item.doc_id, "page": item.page_no, "image_path": item.image_path,
        })
