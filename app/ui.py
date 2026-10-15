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
from .parse import ConversionError, parse_file

SUPPORTED_LABEL = (
    "**Supported formats:** PDF (direct) · PPTX (auto-converted — needs "
    "LibreOffice; exporting the deck to PDF manually and uploading the PDF "
    "also works). Convert anything else to PDF first."
)

TABLE_COLUMNS = ["filename", "format", "pages", "size", "added_at"]


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


def add_file(store: DocumentStore, file) -> tuple[pd.DataFrame, list, str]:
    """Add + parse an uploaded file; returns (library table, dropdown, status)."""
    path = _file_path(file)
    if not path:
        return _library_df(store), _choices(store), "Choose a file to upload first."
    source = Path(path)

    existing = store.find(source)
    try:
        doc = store.add(source)
    except Exception as exc:  # pragma: no cover - upstream failure surface
        return _library_df(store), _choices(store), f"Add failed: {exc}"

    if existing is not None:
        return (
            _library_df(store),
            _choices(store),
            f"'{doc.filename}' is already in the library — no duplicate added.",
        )

    try:
        result = parse_file(
            store.corpus_path(doc.doc_id), doc.doc_id, store.artifacts_dir / doc.doc_id
        )
        note = f"Added '{doc.filename}' · parsed {len(result.pages)} page(s)."
        if result.converted_from:
            note += " (converted from PPTX)"
    except ConversionError as exc:
        note = f"Added '{doc.filename}', but parsing needs LibreOffice: {exc}"
    except Exception as exc:  # pragma: no cover - upstream failure surface
        note = f"Added '{doc.filename}', but parsing failed: {exc}"

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


def build_ui(store: DocumentStore | None = None) -> gr.Blocks:
    store = store or DocumentStore()
    with gr.Blocks(title="Course Assistant — Manage Materials") as demo:
        gr.Markdown("# Course Assistant — Manage Course Materials")
        gr.Markdown(SUPPORTED_LABEL)

        with gr.Row():
            file_input = gr.File(label="Upload a course file", file_types=[".pdf", ".pptx"])
            add_button = gr.Button("Add to library", variant="primary")

        library_table = gr.Dataframe(
            label="Library", headers=TABLE_COLUMNS, interactive=False
        )

        with gr.Row():
            document_dropdown = gr.Dropdown(
                label="Select a document to remove",
                choices=[],
                interactive=True,
                allow_custom_value=True,
            )
            remove_button = gr.Button("Remove selected", variant="stop")

        status = gr.Markdown("Ready.")

        add_button.click(
            lambda f, store=store: add_file(store, f),
            inputs=[file_input],
            outputs=[library_table, document_dropdown, status],
            api_name="add_file",
        )
        remove_button.click(
            lambda c, store=store: remove_selected(store, c),
            inputs=[document_dropdown],
            outputs=[library_table, document_dropdown, status],
            api_name="remove_selected",
        )
        demo.load(
            lambda store=store: (_library_df(store), _choices(store), "Ready."),
            outputs=[library_table, document_dropdown, status],
            api_name="load_library",
        )
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name=APP_HOST, server_port=APP_PORT, inbrowser=False)
