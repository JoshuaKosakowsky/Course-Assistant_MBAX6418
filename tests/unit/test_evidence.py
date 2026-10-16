"""Unit tests for app.evidence (Issue #4 / Phase 3, Step 3).

Offline and deterministic. The manifest-authoritative image validation is
the focus: a matching ``page-001.png`` filename is NOT sufficient — the
image must live under the claiming document's artifact directory and be
recorded for the claimed page in that document's manifest.

PR #14 compatibility: EvidencePack.to_sources() must reproduce the exact
Source(document, page, excerpt, image_path) field contract.
"""

import json
from pathlib import Path

import pytest

import pymupdf

from app.chunking import Chunk
from app.documents import DocumentStore
from app.evidence import (
    EvidenceItem,
    EvidencePack,
    SourceLike,
    item_from_fused,
    item_from_visual,
    rank_metrics,
    source_position,
    validate_artifact_image,
)
from app.fusion import FusedHit, RankedEvidence
from tests.fake_embeddings import FakeEmbedder

pytestmark = pytest.mark.unit


def mk_chunk(chunk_id="c1", doc_id="doc-a", page_no=1, text="cash flow text",
             filename="finance.pdf", image_path=""):
    return Chunk(
        chunk_id=chunk_id, doc_id=doc_id, filename=filename, format=".pdf",
        page_no=page_no, index_in_page=0, text=text, image_path=image_path,
    )


def mk_fused(chunk, fused_rank=1):
    return FusedHit(
        chunk=chunk, fused_rank=fused_rank, fused_score=0.5,
        methods=("keyword", "vector"),
        evidence=(RankedEvidence("keyword", 1, 1.0),
                  RankedEvidence("vector", 1, 0.5)),
    )


def make_png(path, text="page content"):
    d = pymupdf.open()
    pg = d.new_page()
    pg.insert_text((72, 72), text)
    pg.get_pixmap(dpi=72).save(str(path))
    d.close()


def make_store(tmp_path, doc_id="doc-a", filename="finance.pdf", pages=2):
    """A DocumentStore whose artifacts mimic the parser: manifest.json +
    pages/page-001.png … with matching image_path records."""
    store = DocumentStore(root=tmp_path / "data")
    adir = store.artifacts_dir / doc_id
    (adir / "pages").mkdir(parents=True, exist_ok=True)
    pages_json = []
    for pno in range(1, pages + 1):
        img = adir / "pages" / f"page-{pno:03d}.png"
        make_png(img, f"page {pno} content")
        pages_json.append({"number": pno, "text": f"text of page {pno}",
                           "image_path": str(img), "embedded_images": 0})
    (adir / "manifest.json").write_text(
        json.dumps({"pages": pages_json}), encoding="utf-8")
    # a second document's artifacts for wrong-document tests
    bdir = store.artifacts_dir / "doc-b"
    (bdir / "pages").mkdir(parents=True, exist_ok=True)
    bimg = bdir / "pages" / "page-001.png"
    make_png(bimg, "other document's page 1")
    (bdir / "manifest.json").write_text(
        json.dumps({"pages": [{"number": 1, "text": "b",
                               "image_path": str(bimg), "embedded_images": 0}]}),
        encoding="utf-8")
    return store


# 1. evidence-pack construction ---------------------------------------------

def test_evidence_pack_construction(tmp_path):
    store = make_store(tmp_path)
    chunk = mk_chunk(page_no=1,
                     image_path=str(store.artifacts_dir / "doc-a" / "pages" / "page-001.png"))
    item = item_from_fused(mk_fused(chunk), final_rank=1,
                           rerank_score=0.9, image_included=True, store=store)
    assert isinstance(item, EvidenceItem)
    assert (item.doc_id, item.filename, item.page_no, item.chunk_id) == (
        "doc-a", "finance.pdf", 1, "c1")
    assert item.excerpt == "cash flow text"
    assert item.image_path and item.image_source == "artifact"
    assert item.fused_rank == 1 and item.fused_score == 0.5
    assert item.rerank_score == 0.9 and item.image_included is True
    assert item.methods == ("keyword", "vector")
    assert len(item.evidence_ranks) == 2
    assert item.warnings == ()

    pack = EvidencePack(query="q", items=[item], stages={"rerank": "ok"})
    assert pack.summary()["items"] == 1


# 2. PR #14 Source contract ---------------------------------------------------

def test_to_sources_matches_pr14_contract(tmp_path):
    store = make_store(tmp_path)
    chunk = mk_chunk(page_no=2,
                     image_path=str(store.artifacts_dir / "doc-a" / "pages" / "page-002.png"))
    item = item_from_fused(mk_fused(chunk), 1, 0.8, True, store=store)
    pack = EvidencePack(query="q", items=[item])
    sources = pack.to_sources()
    assert len(sources) == 1
    s = sources[0]
    assert isinstance(s, SourceLike)
    # exact field names/types of PR #14's Source(document, page, excerpt, image_path)
    assert set(s.__dataclass_fields__) == {"document", "page", "excerpt", "image_path"}
    assert s.document == "finance.pdf"
    assert s.page == 2 and isinstance(s.page, int)
    assert s.excerpt == "cash flow text"
    assert s.image_path is not None


def test_textless_items_excluded_from_sources(tmp_path):
    store = make_store(tmp_path)
    chunk = mk_chunk(text="   ")  # page without extractable text
    item = item_from_fused(mk_fused(chunk), 1, None, False, store=store)
    assert item.excerpt is None
    assert any("no extractable text" in w for w in item.warnings)
    pack = EvidencePack(query="q", items=[item])
    assert pack.to_sources() == []  # PR #14 requires a non-empty excerpt


# 3. source document/page metadata -------------------------------------------

def test_source_metadata_in_item():
    chunk = mk_chunk(doc_id="doc-x", page_no=7, filename="ops.pdf",
                     text="inventory costs")
    item = item_from_fused(mk_fused(chunk), 3, 0.5, False, store=None)
    assert (item.doc_id, item.page_no, item.filename, item.final_rank) == (
        "doc-x", 7, "ops.pdf", 3)


# 4. image association & validation -------------------------------------------

def test_valid_image_from_own_artifacts(tmp_path):
    store = make_store(tmp_path)
    chunk = mk_chunk(page_no=1,
                     image_path=str(store.artifacts_dir / "doc-a" / "pages" / "page-001.png"))
    path, note = validate_artifact_image(chunk, store)
    assert path == chunk.image_path and note == ""


def test_wrong_document_image_rejected(tmp_path):
    """REGRESSION: a correct page-001.png filename from the WRONG document
    must be rejected even when the name matches."""
    store = make_store(tmp_path)
    foreign = store.artifacts_dir / "doc-b" / "pages" / "page-001.png"
    chunk = mk_chunk(doc_id="doc-a", page_no=1, image_path=str(foreign))
    path, note = validate_artifact_image(chunk, store)
    assert path is None
    assert "wrong-document" in note


def test_wrong_page_filename_rejected(tmp_path):
    store = make_store(tmp_path)
    chunk = mk_chunk(page_no=2,
                     image_path=str(store.artifacts_dir / "doc-a" / "pages" / "page-001.png"))
    path, note = validate_artifact_image(chunk, store)
    assert path is None
    assert "page-001.png" in note  # does not match page-002.png


def test_manifest_mismatch_rejected(tmp_path):
    """The manifest is authoritative: an image NOT recorded for the
    claimed page is rejected even with a correct name/location."""
    store = make_store(tmp_path)
    manifest = store.artifacts_dir / "doc-a" / "manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    other = str(store.artifacts_dir / "doc-a" / "pages" / "page-002.png")
    data["pages"][0]["image_path"] = other  # page 1 claims page 2's image
    manifest.write_text(json.dumps(data), encoding="utf-8")
    chunk = mk_chunk(page_no=1,
                     image_path=str(store.artifacts_dir / "doc-a" / "pages" / "page-001.png"))
    path, note = validate_artifact_image(chunk, store)
    assert path is None
    assert "does not record" in note


def test_missing_image_file():
    chunk = mk_chunk(page_no=1, image_path="C:/does/not/exist/page-001.png")
    path, note = validate_artifact_image(chunk, store=None)
    assert path is None and "no longer exists" in note


def test_corrupt_image_rejected(tmp_path):
    store = make_store(tmp_path)
    img = store.artifacts_dir / "doc-a" / "pages" / "page-001.png"
    img.write_bytes(b"\x00\x01 garbage not an image")
    manifest = store.artifacts_dir / "doc-a" / "manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["pages"][0]["image_path"] = str(img)
    manifest.write_text(json.dumps(data), encoding="utf-8")
    chunk = mk_chunk(page_no=1, image_path=str(img))
    path, note = validate_artifact_image(chunk, store)
    assert path is None and "corrupt" in note


# 5. deleted documents --------------------------------------------------------

def test_deleted_document_image_rejected(tmp_path):
    store = make_store(tmp_path)
    img = store.artifacts_dir / "doc-a" / "pages" / "page-001.png"
    chunk = mk_chunk(page_no=1, image_path=str(img))
    # simulate deletion: ONLY the image removed (manifest still present)
    img.unlink()
    path, note = validate_artifact_image(chunk, store)
    assert path is None and "no longer exists" in note


# 6. visual-index adapter (distinct origin) ------------------------------------

def test_visual_item_distinct_origin(tmp_path):
    class FakeVisualHit:
        doc_id, page, id = "doc-9", 3, "vis-1"
        image_path = ""

    item = item_from_visual(FakeVisualHit())
    assert item.image_source == "none"  # no usable image
    assert item.excerpt is None  # never invent text for visual hits
    assert any("no image path" in w for w in item.warnings)


def test_visual_item_with_existing_image(tmp_path):
    img = tmp_path / "page-001.png"
    make_png(img)

    class FakeVisualHit:
        doc_id, page, id = "doc-9", 1, "vis-2"
        image_path = ""

    FakeVisualHit.image_path = str(img)
    item = item_from_visual(FakeVisualHit())
    assert item.image_source == "visual_index"
    assert item.image_path == str(img)
    assert item.page_no == 1 and item.doc_id == "doc-9"


# 7. evaluation metrics ---------------------------------------------------------

def test_rank_metrics():
    chunks = [mk_chunk(f"c{i}", page_no=1) for i in range(3)]
    chunks[1] = mk_chunk("c1", filename="marketing.pdf", page_no=2)
    ranked = [FusedHit(chunks[i], i + 1, 0.5, ("keyword",),
                       (RankedEvidence("keyword", i + 1, 0.5),))
              for i in range(3)]
    # expected = marketing.pdf page 2 which sits at position 2
    m = rank_metrics(ranked, "marketing.pdf", k=3, expected_page=2)
    assert m["position"] == 2
    assert m["top1"] is False
    assert m["recall@3"] is True


def test_metrics_missing_source():
    from app.fusion import FusedHit
    ranked = [FusedHit(mk_chunk("c0"), 1, 0.5, ("keyword",),
                       (RankedEvidence("keyword", 1, 0.5),))]
    m = rank_metrics(ranked, "nope.pdf", k=3)
    assert m["position"] is None and m["top1"] is False and m["recall@3"] is False


def test_source_position_variants():
    from app.fusion import FusedHit
    from app.rerank import RerankedCandidate

    chunk = mk_chunk("c0", filename="a.pdf", page_no=1)
    fused = FusedHit(chunk, 1, 0.5, ("keyword",),
                     (RankedEvidence("keyword", 1, 0.5),))
    rc = RerankedCandidate(original=fused, rerank_score=0.9, final_rank=1)
    item = EvidenceItem(
        doc_id="doc-a", filename="a.pdf", page_no=1, excerpt="t",
        image_path=None, chunk_id="c0", final_rank=1, rerank_score=None,
        fused_rank=1, fused_score=0.5, methods=("keyword",),
        evidence_ranks=(RankedEvidence("keyword", 1, 0.5),),
        image_included=False, image_source="none", image_verified=False)
    for ranked in ([fused], [rc], [item]):
        assert source_position(ranked, "a.pdf") == 1


# 4b. manifest is REQUIRED for artifact verification -------------------------
# (review finding 2: never label an unrecorded image image_verified=True)

def test_missing_manifest_rejected(tmp_path):
    store = make_store(tmp_path)
    (store.artifacts_dir / "doc-a" / "manifest.json").unlink()
    img = store.artifacts_dir / "doc-a" / "pages" / "page-001.png"
    chunk = mk_chunk(page_no=1, image_path=str(img))
    path, note = validate_artifact_image(chunk, store)
    assert path is None
    assert "missing" in note and "manifest" in note


def test_corrupt_manifest_rejected(tmp_path):
    store = make_store(tmp_path)
    m = store.artifacts_dir / "doc-a" / "manifest.json"
    m.write_text("{ this is not json", encoding="utf-8")
    img = store.artifacts_dir / "doc-a" / "pages" / "page-001.png"
    chunk = mk_chunk(page_no=1, image_path=str(img))
    path, note = validate_artifact_image(chunk, store)
    assert path is None and "corrupt" in note


def test_manifest_missing_page_entry_rejected(tmp_path):
    store = make_store(tmp_path)
    m = store.artifacts_dir / "doc-a" / "manifest.json"
    data = json.loads(m.read_text(encoding="utf-8"))
    data["pages"][0]["number"] = 2    # only page 2 recorded, page 1 claimed
    m.write_text(json.dumps(data), encoding="utf-8")
    img = store.artifacts_dir / "doc-a" / "pages" / "page-001.png"
    chunk = mk_chunk(page_no=1, image_path=str(img))
    path, note = validate_artifact_image(chunk, store)
    assert path is None and "no image record" in note


def test_manifest_windows_style_absolute_path_still_verifies(tmp_path):
    """Platform normalization: backslash absolute records must match the
    claim (Windows-hosted manifests) without weakening page ownership."""
    store = make_store(tmp_path)
    m = store.artifacts_dir / "doc-a" / "manifest.json"
    data = json.loads(m.read_text(encoding="utf-8"))
    data["pages"][0]["image_path"] = str(
        store.artifacts_dir / "doc-a" / "pages" / "page-001.png"
    ).replace("/", "\\")
    m.write_text(json.dumps(data), encoding="utf-8")
    img = store.artifacts_dir / "doc-a" / "pages" / "page-001.png"
    chunk = mk_chunk(page_no=1, image_path=str(img))
    path, note = validate_artifact_image(chunk, store)
    assert path is not None and note == ""


def test_valid_manifest_entry_verifies(tmp_path):
    store = make_store(tmp_path)
    img = store.artifacts_dir / "doc-a" / "pages" / "page-001.png"
    chunk = mk_chunk(page_no=1, image_path=str(img))
    path, note = validate_artifact_image(chunk, store)
    assert path == str(img) and note == ""
