"""Unit tests for document parsing (PDF pages; PPTX conversion policy)."""

from pathlib import Path

import pytest

import app.parse as parse

pytestmark = pytest.mark.unit


def test_pdf_extracts_text_and_page_images(tmp_path, tiny_pdf):
    art = tmp_path / "artifacts"
    res = parse.parse_file(tiny_pdf, doc_id="abc123", artifacts_dir=art)
    assert len(res.pages) == 2
    assert "retrieval augmented generation" in res.pages[0].text.lower()
    for page in res.pages:
        image = Path(page.image_path)
        assert image.is_file()
        assert image.stat().st_size > 0
    assert Path(res.manifest_path).is_file()


def test_unsupported_format_raises_helpful_error(tmp_path):
    f = tmp_path / "notes.docx"
    f.write_bytes(b"docx bytes")
    with pytest.raises(parse.UnsupportedFormatError, match="Unsupported format"):
        parse.parse_file(f, doc_id="x", artifacts_dir=tmp_path / "art")


def test_pptx_without_libreoffice_raises_remediation(tmp_path, monkeypatch):
    monkeypatch.setattr(parse, "find_soffice", lambda: None)
    f = tmp_path / "deck.pptx"
    f.write_bytes(b"pptx bytes")
    with pytest.raises(parse.ConversionError, match="LibreOffice"):
        parse.parse_file(f, doc_id="x", artifacts_dir=tmp_path / "art")
