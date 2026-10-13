"""Shared test fixtures. Tiny, committed, offline — no real course files."""

import pymupdf
import pytest


@pytest.fixture
def tiny_pdf(tmp_path):
    """A 2-page PDF with searchable text, generated in-test."""
    path = tmp_path / "mini.pdf"
    doc = pymupdf.open()
    for i in range(1, 3):
        page = doc.new_page()
        page.insert_text(
            (72, 72),
            f"Page {i}: retrieval augmented generation quiz prep slides",
        )
    doc.save(str(path))
    doc.close()
    return path
