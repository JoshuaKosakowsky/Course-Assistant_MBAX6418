"""Ingestion workflow shared by the `python -m app.ingest` CLI and the UI.

``add_and_parse()`` is the single registration path: validate the format
first, register the file in the library, and only then parse. A failed
parse rolls back the record created for this attempt (or left behind by an
earlier interrupted one) so a corrected environment can retry cleanly —
a failed import must never block a successful retry or be reported as an
already-existing duplicate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .documents import Document, DocumentNotFoundError, DocumentStore
from .parse import ConversionError, ParseResult, UnsupportedFormatError, parse_file, validate_format


def add_and_parse(store: DocumentStore, source: Path) -> tuple[Document, ParseResult | None, bool]:
    """Register and parse one file; roll back on parse failure.

    Returns ``(document, result, already_parsed)``. ``already_parsed`` is
    True only when the library already holds this exact content AND its
    manifest is present — i.e. a genuine duplicate, never an unparsed
    leftover. On any parse failure the just-created (or stale) record and
    its corpus/artifact dirs are removed so the import stays retryable;
    the original exception is re-raised.
    """
    validate_format(source)
    document = store.add(source)
    if store.is_parsed(document.doc_id):
        return document, None, True
    try:
        result = parse_file(
            store.corpus_path(document.doc_id),
            document.doc_id,
            store.artifacts_dir / document.doc_id,
        )
    except Exception:
        try:
            store.remove(document.doc_id)
        except Exception:
            pass  # keep the original failure; the stale record stays re-parseable
        raise
    return document, result, False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Add and parse one PDF or PPTX course document."
    )
    parser.add_argument("file", type=Path, help="Path to a PDF or PPTX course document")
    args = parser.parse_args(argv)

    source = args.file.expanduser()
    if not source.is_file():
        parser.error(f"File not found: {source}")

    store = DocumentStore()
    try:
        document, result, already_parsed = add_and_parse(store, source)
    except (UnsupportedFormatError, ConversionError, OSError, ValueError,
            DocumentNotFoundError) as exc:
        parser.exit(2, f"Ingestion failed: {exc}\n")

    if already_parsed:
        print(f"'{document.filename}' is already in the library — no duplicate added.")
        return 0
    assert result is not None  # parse failure raises inside add_and_parse
    print(f"Added '{document.filename}' · parsed {len(result.pages)} page(s).")
    print(f"Manifest: {result.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
