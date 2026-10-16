"""Smoke tests for the offline ingestion CLI."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

import app.documents as documents
import app.ingest as ingest
import app.parse as parse
from app.ingest import main

pytestmark = pytest.mark.smoke
REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_ingest(source: Path, data_root: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "DATA_ROOT": str(data_root)}
    return subprocess.run(
        [sys.executable, "-m", "app.ingest", str(source)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_ingest_cli_parses_pdf_and_reports_saved_artifacts(tmp_path, tiny_pdf):
    result = _run_ingest(tiny_pdf, tmp_path / "data")

    assert result.returncode == 0, result.stderr
    assert "Added 'mini.pdf'" in result.stdout
    assert "parsed 2 page(s)" in result.stdout
    assert "manifest.json" in result.stdout
    assert len(list((tmp_path / "data" / "artifacts").glob("*"))) == 1


def test_ingest_cli_rejects_unsupported_format_helpfully(tmp_path):
    source = tmp_path / "notes.docx"
    source.write_bytes(b"not a supported course document")

    result = _run_ingest(source, tmp_path / "data")

    assert result.returncode == 2
    assert "Unsupported format '.docx'" in result.stderr
    assert "PDF directly" in result.stderr


# -- in-process regressions (mase7919 review findings) ----------------------
# The CLI/UI must never leave a library record that blocks a retry, and must
# report "already in the library" only for content that actually parsed.


def _store(monkeypatch, tmp_path):
    monkeypatch.setattr(documents, "DATA_ROOT", tmp_path / "data")
    return documents.DocumentStore()


def test_unsupported_file_is_not_registered(tmp_path, monkeypatch):
    store = _store(monkeypatch, tmp_path)
    source = tmp_path / "notes.docx"
    source.write_bytes(b"unsupported bytes")

    with pytest.raises(SystemExit) as exc:
        main([str(source)])

    assert exc.value.code == 2
    assert store.list() == []
    assert not list((tmp_path / "data" / "library").glob("*.json"))


def test_failed_pptx_rolls_back_and_retry_succeeds(tmp_path, monkeypatch):
    store = _store(monkeypatch, tmp_path)
    monkeypatch.setattr(parse, "find_soffice", lambda: None)
    deck = tmp_path / "deck.pptx"
    deck.write_bytes(b"not a real pptx")

    with pytest.raises(SystemExit) as exc:
        main([str(deck)])
    assert exc.value.code == 2
    assert store.list() == []  # rolled back — nothing blocks a retry

    # LibreOffice "appears"; the same file now imports cleanly.
    monkeypatch.setattr(
        ingest,
        "parse_file",
        lambda source, doc_id, artifacts_dir: parse.ParseResult(
            doc_id=doc_id, format=".pptx", pages=[],
            converted_from=deck.name, manifest_path=str(artifacts_dir / "manifest.json"),
        ),
    )
    assert main([str(deck)]) == 0
    assert len(store.list()) == 1
    assert store.list()[0].filename == "deck.pptx"


def test_stale_unparsed_record_is_reparsed_not_duplicated(tmp_path, monkeypatch, tiny_pdf, capsys):
    store = _store(monkeypatch, tmp_path)
    store.add(tiny_pdf)  # registered but never parsed (e.g. interrupted import)

    assert main([str(tiny_pdf)]) == 0
    out = capsys.readouterr().out
    assert "already in the library" not in out
    doc = store.list()[0]
    assert store.is_parsed(doc.doc_id)

    assert main([str(tiny_pdf)]) == 0
    assert "already in the library" in capsys.readouterr().out
    assert len(store.list()) == 1  # still one record, never duplicated


def test_pptx_timeout_exits_cleanly_and_rolls_back(tmp_path, monkeypatch, capsys):
    store = _store(monkeypatch, tmp_path)

    def hang(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["soffice"], timeout=300)

    monkeypatch.setattr(parse, "find_soffice", lambda: tmp_path / "soffice")
    monkeypatch.setattr(parse.subprocess, "run", hang)
    deck = tmp_path / "deck.pptx"
    deck.write_bytes(b"pptx bytes")

    with pytest.raises(SystemExit) as exc:
        main([str(deck)])

    assert exc.value.code == 2
    assert "timed out" in capsys.readouterr().err
    assert store.list() == []
