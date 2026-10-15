"""Document parsing: PDF and PPTX -> text pages + page images (Task 1).

- PDF: text extracted per page (PyMuPDF), each page rendered to PNG at
  150 dpi for visual evidence. Embedded-image counts are recorded so the
  retrieval phase knows which pages carry diagrams.
- PPTX: converted to PDF with headless LibreOffice when available, then the
  PDF path runs. LibreOffice is OPTIONAL — manually exported PDFs work too.
  The converter degrades to a clear error naming the install/manual-export
  options when `soffice` is missing.

Artifacts are written under the document's artifacts directory:
    pages/page-001.png  page-001 text in manifest.json
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pymupdf

SUPPORTED_EXTENSIONS = {".pdf", ".pptx"}

_UNSUPPORTED_MESSAGE = (
    "Unsupported format {ext!r}. Supported: PDF directly; PPTX via automatic "
    "LibreOffice conversion or manual export to PDF. Convert the file to PDF "
    "and try again."
)


class UnsupportedFormatError(ValueError):
    pass


class ConversionError(RuntimeError):
    pass


@dataclass
class PageRecord:
    number: int
    text: str
    image_path: str
    embedded_images: int


@dataclass
class ParseResult:
    doc_id: str
    format: str          # original extension, e.g. ".pptx"
    pages: list[PageRecord]
    converted_from: str | None   # original filename when PPTX was converted
    manifest_path: str


def find_soffice() -> Path | None:
    """Locate LibreOffice's console launcher: env override, PATH, or
    standard Windows install paths. Prefer ``soffice.com`` on Windows so
    subprocesses can reliably capture output from headless conversions."""
    override = os.environ.get("LIBREOFFICE_SOFFICE")
    if override and Path(override).is_file():
        return _console_launcher(Path(override))
    exe = shutil.which("soffice") or shutil.which("libreoffice")
    if exe:
        return _console_launcher(Path(exe))
    for candidate in (
        "C:/Program Files/LibreOffice/program/soffice.exe",
        "C:/Program Files (x86)/LibreOffice/program/soffice.exe",
    ):
        path = Path(candidate)
        if path.is_file():
            return _console_launcher(path)
    return None


def _console_launcher(path: Path) -> Path:
    """Use the console-subsystem launcher when available on Windows."""
    if os.name == "nt" and path.suffix.lower() == ".exe":
        console_path = path.with_suffix(".com")
        if console_path.is_file():
            return console_path
    return path


def convert_pptx_to_pdf(source: Path, out_dir: Path) -> Path:
    """Convert a PPTX to PDF with headless LibreOffice. Raises
    ConversionError with remediation steps when soffice is unavailable or
    the conversion fails."""
    soffice = find_soffice()
    if soffice is None:
        raise ConversionError(
            "LibreOffice is not installed, so PPTX cannot be converted "
            "automatically. Install LibreOffice (https://www.libreoffice.org "
            "/download) or export the deck to PDF manually and upload the "
            "PDF instead."
        )
    profile_dir = out_dir / ".lo-profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(soffice),
        "--headless",
        "-env:UserInstallation=file:///" + str(profile_dir).replace("\\", "/"),
        "--convert-to",
        "pdf",
        "--outdir",
        str(out_dir),
        str(source),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    pdf_path = out_dir / (source.stem + ".pdf")
    if proc.returncode != 0 or not pdf_path.is_file():
        raise ConversionError(
            f"LibreOffice conversion failed (exit {proc.returncode}): "
            f"{proc.stderr.strip()[:400] or 'see LibreOffice output'}"
        )
    return pdf_path


def _extract_pdf(pdf_path: Path, artifacts_dir: Path) -> list[PageRecord]:
    pages_dir = artifacts_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    records: list[PageRecord] = []
    with pymupdf.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            image_path = pages_dir / f"page-{i:03d}.png"
            page.get_pixmap(dpi=150).save(image_path)
            records.append(
                PageRecord(number=i, text=text, image_path=str(image_path),
                           embedded_images=len(page.get_images()))
            )
    return records


def parse_file(source: Path, doc_id: str, artifacts_dir: Path) -> ParseResult:
    """Parse one course file into per-page text + images.

    Unsupported formats raise UnsupportedFormatError naming the accepted
    ones. PPTX conversion raises ConversionError when LibreOffice is
    missing. Writes a manifest.json alongside the pages.
    """
    source = Path(source)
    ext = source.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(_UNSUPPORTED_MESSAGE.format(ext=ext))
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    converted_from: str | None = None
    pdf_path = source
    if ext == ".pptx":
        converted_from = source.name
        pdf_path = convert_pptx_to_pdf(source, artifacts_dir)

    pages = _extract_pdf(pdf_path, artifacts_dir)

    manifest_path = artifacts_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "doc_id": doc_id,
                "format": ext,
                "converted_from": converted_from,
                "pages": [p.__dict__ for p in pages],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return ParseResult(
        doc_id=doc_id,
        format=ext,
        pages=pages,
        converted_from=converted_from,
        manifest_path=str(manifest_path),
    )
