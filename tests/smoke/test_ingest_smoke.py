"""Smoke tests for the offline ingestion CLI."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

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
