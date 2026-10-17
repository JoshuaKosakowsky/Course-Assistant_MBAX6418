"""Gradio management UI (Task 1): add, list, and remove course materials.

Run with:  python -m app.ui

- Add: upload PDF/PPTX; the store de-duplicates by content (sha256), so
  loading the same file twice never duplicates it. The file is parsed to
  text pages + page images immediately.
- Remove: select a document and remove it; its corpus copy, parsed
  artifacts, and record are deleted so later answers cannot rely on it.
- The library table shows filename, format, page count, size, and add time.

Handler functions take the store as an argument so tests can inject a
temporary data root.
"""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import pandas as pd

from .config import APP_HOST, APP_PORT
from .documents import DocumentNotFoundError, DocumentStore
from .embeddings import EmbeddingsError, TextEmbeddingClient
from .indexing import CorpusIndexer
from .lookup import lookup_question
from .ingest import add_and_parse
from .parse import ConversionError, UnsupportedFormatError
from .quiz_ui import QuizController
from .study_ui import QnaController, production_study_services

SUPPORTED_LABEL = (
    "**Supported formats:** PDF (direct) · PPTX (auto-converted — needs "
    "LibreOffice; exporting the deck to PDF manually and uploading the PDF "
    "also works). Convert anything else to PDF first."
)

TABLE_COLUMNS = ["filename", "format", "pages", "size", "added_at"]

# Scope overrides to Lookup so existing Library controls retain their styling.
LOOKUP_CSS = """
#lookup-heading h2 {
    font-family: var(--font-sans, var(--font));
    font-size: var(--block-label-text-size);
    font-weight: var(--block-label-text-weight);
    line-height: 1.4;
    color: var(--block-label-text-color);
    margin: 0;
}
#lookup-button, #lookup-button button {
    background: #2563eb;
    border-color: #2563eb;
    color: #ffffff;
}
#lookup-button:hover, #lookup-button button:hover {
    background: #1d4ed8;
    border-color: #1d4ed8;
}
#lookup-button:active, #lookup-button button:active {
    background: #1e40af;
    border-color: #1e40af;
}
#lookup-button:focus-visible, #lookup-button button:focus-visible {
    outline: 2px solid #2563eb;
    outline-offset: 3px;
}
#answer-gallery .grid-container, #quiz-gallery .grid-container {
    grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
    grid-template-rows: minmax(0, 1fr) !important;
    height: 100% !important;
}
#answer-gallery .gallery-item, #quiz-gallery .gallery-item,
#answer-gallery .thumbnail-item, #quiz-gallery .thumbnail-item {
    min-height: 0 !important;
    height: 100% !important;
    width: 100% !important;
    aspect-ratio: auto !important;
}
#answer-gallery .thumbnail-item img, #quiz-gallery .thumbnail-item img {
    width: 100%;
    height: 100%;
    object-fit: contain !important;
}
"""


def _file_path(file) -> str | None:
    """Coerce gr.File output (path string or FileData) to a path string."""
    if file is None:
        return None
    return getattr(file, "path", None) or str(file)


def _library_df(store: DocumentStore) -> pd.DataFrame:
    rows = []
    for doc in store.list():
        manifest = store.artifacts_dir / doc.doc_id / "manifest.json"
        pages = "—"
        if manifest.is_file():
            pages = len(json.loads(manifest.read_text(encoding="utf-8")).get("pages", []))
        rows.append(
            {
                "filename": doc.filename,
                "format": doc.format,
                "pages": pages,
                "size": f"{doc.size_bytes / 1024:.0f} KB",
                "added_at": doc.added_at,
            }
        )
    return pd.DataFrame(rows, columns=TABLE_COLUMNS)


def _choices(store: DocumentStore) -> list[tuple[str, str]]:
    return [(doc.filename, doc.doc_id) for doc in store.list()]


def add_file(
    store: DocumentStore, file, indexer=None
) -> tuple[pd.DataFrame, list, str]:
    """Add + parse an uploaded file; returns (library table, dropdown, status).

    ``indexer`` is optional (app.indexing.CorpusIndexer): when provided,
    a successfully parsed document is indexed immediately, and the status
    reports the chunk counts or the reason indexing was not possible.
    """
    path = _file_path(file)
    if not path:
        return _library_df(store), _choices(store), "Choose a file to upload first."
    source = Path(path)

    parsed_ok = False
    doc = None
    try:
        doc, result, already_parsed = add_and_parse(store, source)
    except UnsupportedFormatError as exc:
        note = f"Could not add '{source.name}': {exc}"
    except ConversionError as exc:
        note = f"Could not add '{source.name}': {exc}"
    except Exception as exc:  # unknown parse failure — never crash the UI
        note = f"Could not add '{source.name}', parsing failed: {exc}"
    else:
        if already_parsed:
            note = f"'{doc.filename}' is already in the library — no duplicate added."
        else:
            parsed_ok = True
            assert result is not None  # parse failure raises inside add_and_parse
            note = f"Added '{doc.filename}' · parsed {len(result.pages)} page(s)."
            if result.converted_from:
                note += " (converted from PPTX)"

    if indexer is not None and parsed_ok:
        assert doc is not None  # parsed_ok implies add_and_parse succeeded
        try:
            stats = indexer.index_document(doc.doc_id)
            note += f" · indexed {stats.chunks} chunk(s)"
            note += (
                f" · visual {stats.visual_pages} page(s)"
                if stats.visual_pages else ""
            )
            if (indexer.visual is not None and stats.visual_failed):
                note += f" · visual FAILED ({len(stats.visual_failed)} page(s))"
            if stats.skipped:
                if indexer.embedder is None:
                    note += " · vectors skipped (keyword-only mode: no embedding configured)"
                else:
                    note += " · vectors skipped (embedding service unavailable)"
            if stats.errors:
                note += " · indexing errors (see application log)"
        except Exception as exc:
            note += f" · indexing failed: {exc}"

    return _library_df(store), _choices(store), note


def remove_selected(store: DocumentStore, choice: str | None) -> tuple[pd.DataFrame, list, str]:
    """Remove the selected document (cascade delete); returns the refreshed UI."""
    if not choice:
        return _library_df(store), _choices(store), "Select a document to remove first."
    try:
        doc = store.get(choice)
        store.remove(choice)
        message = f"Removed '{doc.filename}' including all of its searchable content."
    except DocumentNotFoundError:
        message = f"Document {choice!r} is no longer in the library."
    except Exception as exc:  # pragma: no cover - upstream failure surface
        message = f"Remove failed: {exc}"
    return _library_df(store), _choices(store), message


def build_ui(store: DocumentStore | None = None, indexer=None, quiz_provider=None,
             *, retrieval=None, qna=None, connect_services=False) -> gr.Blocks:
    store = store or DocumentStore()

    if indexer is not None:
        indexer.register_hooks()
        try:
            indexer.reconcile()
        except Exception as exc:
            print(f"index reconciliation failed: {exc}")

    # Reconcile replaces the in-memory keyword index: bind retrieval AFTER it.
    if connect_services and indexer is not None:
        retrieval, qna, quiz_provider = production_study_services(indexer)
    quiz_controller = QuizController(quiz_provider, store=store, unavailable_message=(
        "Quiz generation is unavailable. Configure the server-side vision model and restart the app."
        if connect_services and quiz_provider is None else None))
    qna_controller = QnaController(store, retrieval, qna)

    with gr.Blocks(title="Course Assistant") as demo:
        gr.Markdown("# Course Assistant")
        gr.Markdown(
            "Study your course materials, ask questions, and practice with quizzes."
        )

        with gr.Tabs():
            with gr.Tab("Course Materials"):
                gr.Markdown(SUPPORTED_LABEL)

                with gr.Row():
                    file_input = gr.File(
                        label="Upload a course file",
                        file_types=[".pdf", ".pptx"],
                    )
                    add_button = gr.Button(
                        "Add to library", variant="primary"
                    )

                library_table = gr.Dataframe(
                    label="Library",
                    headers=TABLE_COLUMNS,
                    interactive=False,
                )

                with gr.Row():
                    document_dropdown = gr.Dropdown(
                        label="Select a document to remove",
                        choices=[],
                        interactive=True,
                        allow_custom_value=True,
                    )
                    remove_button = gr.Button(
                        "Remove selected", variant="stop"
                    )

                status = gr.Markdown("Ready.")

            with gr.Tab("Ask Questions"):
                gr.Markdown(
                    "Ask about text, pictures, diagrams, or charts in your selected materials. "
                    "Answers use retrieved evidence; unavailable information is acknowledged."
                )
                question_materials = gr.CheckboxGroup(
                    label="Materials for your question", choices=_choices(store),
                )
                question = gr.Textbox(label="Your question", lines=2, max_length=2000)
                ask_button = gr.Button("Ask question", variant="primary")
                answer_status = gr.Markdown()
                answer_output = gr.HTML(label="Answer and sources")
                answer_images = gr.Gallery(label="Supporting original slides and pages", columns=2, rows=1,
                                           object_fit="contain", height=420, elem_id="answer-gallery")

            with gr.Tab("Practice Quiz"):
                gr.Markdown("### Create a practice quiz")
                gr.Markdown(
                    "Choose your course materials and customize your quiz."
                )

                quiz_materials = gr.CheckboxGroup(
                    label="Course materials",
                    choices=_choices(store),
                    info="Select one or more uploaded documents.",
                )

                quiz_topic = gr.Textbox(
                    label="Topic (optional)",
                    placeholder="e.g., Retrieval-Augmented Generation",
                )

                with gr.Row():
                    quiz_count = gr.Slider(
                        minimum=1,
                        maximum=5,
                        step=1,
                        value=3,
                        label="Number of questions",
                    )
                    quiz_seed = gr.Number(
                        value=42,
                        precision=0,
                        label="Quiz seed",
                    )

                generate_button = gr.Button(
                    "Generate Practice Quiz",
                    variant="primary",
                )

                quiz_status = gr.Markdown(
                    "Select your materials to get started."
                )

                gr.Markdown("### Questions")
                quiz_answers = [
                    gr.Radio(label=f"Question {i + 1}", choices=[],
                             visible=False, interactive=True)
                    for i in range(5)
                ]
                with gr.Row():
                    submit_button = gr.Button("Submit Answers", variant="primary")
                    reveal_button = gr.Button("Reveal Answers")
                quiz_score = gr.Markdown()
                quiz_feedback = gr.Markdown()
                quiz_images = gr.Gallery(
                    label="Verified original page images",
                    visible=True,
                    columns=2,
                    rows=1,
                    height=300,
                    object_fit="contain",
                    elem_id="quiz-gallery",
                )

        for event in (ask_button.click, question.submit):
            event(qna_controller.ask, inputs=[question_materials, question],
                  outputs=[answer_status, answer_output, answer_images], api_name=False)

        generate_button.click(
            quiz_controller.generate,
            inputs=[quiz_materials, quiz_topic, quiz_count, quiz_seed],
            outputs=[quiz_status, *quiz_answers, quiz_score, quiz_feedback, quiz_images],
            api_name=False,
        )

        def submit_quiz(a1, a2, a3, a4, a5, request: gr.Request):
            return quiz_controller.finish(False, (a1, a2, a3, a4, a5), request)

        def reveal_quiz(a1, a2, a3, a4, a5, request: gr.Request):
            return quiz_controller.finish(True, (a1, a2, a3, a4, a5), request)

        submit_button.click(
            submit_quiz, inputs=quiz_answers,
            outputs=[quiz_score, quiz_feedback, quiz_images], api_name=False,
        )
        reveal_button.click(
            reveal_quiz, inputs=quiz_answers,
            outputs=[quiz_score, quiz_feedback, quiz_images], api_name=False,
        )

        gr.Markdown("## Lookup", elem_id="lookup-heading")
        gr.Markdown(
            "Ask a question to find matching passages in the Library PDFs. "
            "Results show verbatim excerpts, filenames, and page numbers—not a generated answer."
        )
        lookup_input = gr.Textbox(
            label="Question",
            placeholder="What would you like to look up in your course PDFs?",
            lines=1,
            max_lines=4,
            max_length=2000,
        )
        lookup_button = gr.Button("Look up", variant="primary", elem_id="lookup-button")
        lookup_results = gr.HTML(
            value="<p>Enter a question and select Look up, or press Enter.</p>",
            label="Lookup results",
        )
        lookup_button.click(
            lambda question: lookup_question(store, question, indexer),
            inputs=[lookup_input],
            outputs=[lookup_results],
            api_name="lookup",
        )
        lookup_input.submit(
            lambda question: lookup_question(store, question, indexer),
            inputs=[lookup_input],
            outputs=[lookup_results],
            api_name="lookup_submit",
        )

        add_event = add_button.click(
            lambda f, store=store, indexer=indexer: add_file(
                store, f, indexer
            ),
            inputs=[file_input],
            outputs=[library_table, document_dropdown, status],
            api_name="add_file",
        )

        add_event.then(
            lambda: gr.update(choices=_choices(store), value=[]),
            inputs=[], outputs=[quiz_materials], api_name=False,
        )
        add_event.then(
            lambda: gr.update(choices=_choices(store), value=[]),
            outputs=[question_materials], api_name=False,
        )

        remove_event = remove_button.click(
            lambda c, store=store: remove_selected(store, c),
            inputs=[document_dropdown],
            outputs=[library_table, document_dropdown, status],
            api_name="remove_selected",
        )

        remove_event.then(
            lambda: gr.update(choices=_choices(store), value=[]),
            inputs=[], outputs=[quiz_materials], api_name=False,
        )
        remove_event.then(
            lambda: gr.update(choices=_choices(store), value=[]),
            outputs=[question_materials], api_name=False,
        )

        demo.load(lambda: gr.update(choices=_choices(store), value=[]),
                  outputs=[question_materials], api_name=False)

        demo.load(
            lambda: gr.update(choices=_choices(store), value=[]),
            outputs=[quiz_materials], api_name=False,
        )

        demo.load(
            lambda store=store: (
                _library_df(store),
                _choices(store),
                "Ready.",
            ),
            outputs=[library_table, document_dropdown, status],
            api_name="load_library",
        )

    return demo


def _production_indexer(
    store: DocumentStore | None = None, visual_factory=None
) -> tuple[DocumentStore, CorpusIndexer]:
    """Production startup wiring for ``python -m app.ui``.

    Builds the real embedding client from the local .env; when credentials
    are missing or the service is down, the app degrades to keyword-only
    mode (BM25 still works) and never substitutes a fake embedder.

    ``visual_factory`` is an injectable ``(store) -> (VisualIndexer|None,
    reason)`` for offline tests; the default is production_visual_indexer
    (feature-detects MSpanny's app.visual_index — before PR #11 merges
    this reports a reason and keeps the app in text-only mode). Startup
    runs the visual reconcile ONLY when a visual state file already exists
    (incremental; zero embedding calls for unchanged pages), so the first
    boot after enabling the feature never triggers an automatic
    full-library visual reindex — use ``python -m app.indexing
    --visual-backfill`` for that.
    """
    if store is None:
        store = DocumentStore()
    embedder = None
    try:
        embedder = TextEmbeddingClient()
        print(f"Semantic search: using {embedder.model!r} at {embedder.base_url}")
    except EmbeddingsError as exc:
        print(f"Keyword-only mode (semantic search disabled): {exc}")
    from app.visual import production_visual_indexer

    visual = None
    factory = visual_factory or production_visual_indexer
    try:
        visual, reason = factory(store)
    except Exception as exc:  # pragma: no cover - factory fails loudly
        visual, reason = None, f"visual index unavailable: {exc}"
    if visual is None:
        print(f"Text-only mode: {reason}")
    indexer = CorpusIndexer(store, embedder=embedder, visual=visual)
    return store, indexer


if __name__ == "__main__":
    store, indexer = _production_indexer()
    build_ui(store, indexer, connect_services=True).launch(
        server_name=APP_HOST, server_port=APP_PORT, inbrowser=False, css=LOOKUP_CSS
    )
