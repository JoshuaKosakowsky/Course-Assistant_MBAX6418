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


def test_pptx_conversion_prefers_console_launcher_on_windows(tmp_path, monkeypatch):
    if parse.os.name != "nt":
        pytest.skip("Windows launcher selection")

    soffice_com = tmp_path / "soffice.com"
    soffice_exe = tmp_path / "soffice.exe"
    soffice_com.write_bytes(b"console launcher")
    soffice_exe.write_bytes(b"gui launcher")
    monkeypatch.setenv("LIBREOFFICE_SOFFICE", str(soffice_exe))

    assert parse.find_soffice() == soffice_com
