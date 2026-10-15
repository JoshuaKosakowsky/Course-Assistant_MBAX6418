"""Command-line ingestion: python -m app.ingest <file>"""

from __future__ import annotations

import argparse
from pathlib import Path

from .documents import DocumentStore
from .parse import ConversionError, UnsupportedFormatError, parse_file


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
        existing = store.find(source)
        document = store.add(source)
        if existing is not None:
            print(f"'{document.filename}' is already in the library — no duplicate added.")
            return 0
        result = parse_file(
            store.corpus_path(document.doc_id),
            document.doc_id,
            store.artifacts_dir / document.doc_id,
        )
    except (UnsupportedFormatError, ConversionError, OSError, ValueError) as exc:
        parser.exit(2, f"Ingestion failed: {exc}\n")

    print(f"Added '{document.filename}' · parsed {len(result.pages)} page(s).")
    print(f"Manifest: {result.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
