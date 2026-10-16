"""Unit tests for app.chunking (Issue #2 / Phase 2 change 1).

Synthetic inputs only — no real course materials, no network, no class
endpoints. Marked unit so it runs in the default `pytest` tier.
"""

from types import SimpleNamespace

import pytest

from app.chunking import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    Chunk,
    PageText,
    chunk_id,
    chunk_pages,
    validate_settings,
)

pytestmark = pytest.mark.unit

DOC_A = "aaaa111122223333"
DOC_B = "bbbb444455556666"
FNAME = "lecture-01.pdf"


def make_pages(*texts) -> list[PageText]:
    return [PageText(page_no=i + 1, text=t) for i, t in enumerate(texts)]


# ---- basic splitting -------------------------------------------------

def test_normal_text_divided_into_chunks(tmp_path):
    words = " ".join(f"w{i}" for i in range(400))  # ~1200+ chars
    pages = make_pages(words)
    chunks = chunk_pages(DOC_A, FNAME, pages, chunk_size=200, chunk_overlap=25)
    assert len(chunks) > 3, f"long page should produce several chunks, got {len(chunks)}"
    for c in chunks:
        assert c.page_no == 1
        assert c.text in words  # a chunk never contains foreign text
    assert chunks[0].text.startswith("w0")
    assert chunks[-1].text.rstrip().endswith("w399")


def test_short_text_stays_single_chunk():
    text = "Short lecture note with a few words."
    chunks = chunk_pages(DOC_A, FNAME, make_pages(text))
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].index_in_page == 0


def test_empty_and_whitespace_pages_skipped():
    chunks = chunk_pages(
        DOC_A, FNAME, [PageText(1, "   \n\t "), PageText(3, "Real content.")]
    )
    assert [c.page_no for c in chunks] == [3]
    assert chunks[0].index_in_page == 0  # per-page counter, no phantom chunk


def test_no_pages_yields_no_chunks():
    assert chunk_pages(DOC_A, FNAME, []) == []


def test_overlap_applied_as_configured():
    text = "abcdefghij" * 10  # 100 chars, no whitespace -> char-level split
    chunks = chunk_pages(DOC_A, FNAME, make_pages(text),
                         chunk_size=12, chunk_overlap=4)
    assert len(chunks) >= 2
    assert chunks[1].text.startswith(chunks[0].text[-4:]), (
        "the configured overlap must appear at the start of the next chunk"
    )


# ---- metadata & provenance -------------------------------------------

def test_metadata_survives_on_every_chunk():
    words = " ".join(f"w{i}" for i in range(300))
    chunks = chunk_pages(DOC_A, FNAME, make_pages(words),
                         chunk_size=150, chunk_overlap=20,
                         format=".pdf")
    assert len(chunks) >= 2
    for c in chunks:
        assert c.doc_id == DOC_A
        assert c.filename == FNAME
        assert c.format == ".pdf"
        assert c.page_no == 1
        assert c.chunk_id == chunk_id(DOC_A, c.page_no, c.index_in_page)
    assert [c.index_in_page for c in chunks] == list(range(len(chunks)))


def test_no_chunk_spans_pages():
    page_texts = [
        " ".join(f"alpha{i}" for i in range(300)),
        " ".join(f"beta{i}" for i in range(300)),
    ]
    chunks = chunk_pages(DOC_A, FNAME, make_pages(*page_texts),
                         chunk_size=150, chunk_overlap=15)
    assert {c.page_no for c in chunks} == {1, 2}
    for c in chunks:
        source = page_texts[c.page_no - 1]
        assert c.text in source, "chunk must be fully contained in its own page"
    # per-page chunk counters restart
    assert any(c.page_no == 2 and c.index_in_page == 0 for c in chunks)
    assert not any(c.page_no == 1 and c.text.startswith("beta") for c in chunks)
    assert not any(c.page_no == 2 and "alpha" in c.text for c in chunks)


def test_no_chunk_mixes_documents():
    words = " ".join(f"w{i}" for i in range(200))
    chunks_a = chunk_pages(DOC_A, FNAME, make_pages(words), chunk_size=100)
    chunks_b = chunk_pages(DOC_B, "other.pdf", make_pages(words), chunk_size=100)
    assert all(c.doc_id == DOC_A for c in chunks_a)
    assert all(c.doc_id == DOC_B for c in chunks_b)
    a_ids = {c.chunk_id for c in chunks_a}
    b_ids = {c.chunk_id for c in chunks_b}
    assert a_ids.isdisjoint(b_ids), "chunk ids must be distinct across documents"


# ---- ids, stability, repeatability -----------------------------------

def test_chunk_ids_stable_for_unchanged_input():
    words = " ".join(f"w{i}" for i in range(250))
    pages = make_pages(words)
    first = chunk_pages(DOC_A, FNAME, pages, chunk_size=120, chunk_overlap=10)
    second = chunk_pages(DOC_A, FNAME, pages, chunk_size=120, chunk_overlap=10)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert first == second  # fully repeatable


def test_repeatable_across_runs():
    words = " ".join(f"w{i}" for i in range(250))
    pages = make_pages(words)
    r1 = [c.text for c in chunk_pages(DOC_A, FNAME, pages, chunk_size=120)]
    r2 = [c.text for c in chunk_pages(DOC_A, FNAME, pages, chunk_size=120)]
    assert r1 == r2


# ---- edge cases -------------------------------------------------------

def test_unusually_long_input_handled():
    text = "foo bar baz " * 8000  # ~96k chars
    chunks = chunk_pages(DOC_A, FNAME, make_pages(text))
    assert len(chunks) > 50
    for c in chunks:
        assert len(c.text) > 0
        assert len(c.text) <= DEFAULT_CHUNK_SIZE + DEFAULT_CHUNK_OVERLAP + 64
    assert chunks[-1].text.rstrip().endswith("baz")


def test_image_path_carried_onto_chunks():
    text = " ".join(f"w{i}" for i in range(200))
    page = PageText(7, text, image_path="artifacts/x/pages/page-007.png")
    chunks = chunk_pages(DOC_A, FNAME, [page], chunk_size=100)
    assert len(chunks) >= 2
    assert all(c.image_path == page.image_path for c in chunks)


# ---- input shape compatibility (PR #10 PageRecord duck-typing) --------

def test_accepts_page_record_shapes():
    record = SimpleNamespace(number=3, text="Slide content here.", image_path="slide3.png")
    chunks = chunk_pages(DOC_A, FNAME, [record])
    assert len(chunks) == 1
    assert chunks[0].page_no == 3
    assert chunks[0].image_path == "slide3.png"

    dict_page = {"number": 4, "text": "Dict page."}
    chunks2 = chunk_pages(DOC_A, FNAME, [dict_page])
    assert chunks2[0].page_no == 4


def test_missing_page_number_rejected():
    with pytest.raises(ValueError, match="page number"):
        chunk_pages(DOC_A, FNAME, [{"text": "no page number"}])


# ---- configuration validation ----------------------------------------

@pytest.mark.parametrize(
    "size, overlap",
    [
        (0, 0),
        (-5, 0),
        (True, 0),
        (512, -1),
        (100, 100),
        (100, 150),
    ],
)
def test_invalid_configuration_rejected(size, overlap):
    with pytest.raises(ValueError):
        chunk_pages(DOC_A, FNAME, make_pages("text"), chunk_size=size, chunk_overlap=overlap)


def test_validate_settings_messages():
    with pytest.raises(ValueError, match="chunk_size"):
        validate_settings(0, 0)
    with pytest.raises(ValueError, match="chunk_overlap"):
        validate_settings(100, -1)
    with pytest.raises(ValueError, match="smaller than chunk_size"):
        validate_settings(100, 100)


def test_defaults_are_reasonable():
    assert DEFAULT_CHUNK_SIZE == 512
    assert DEFAULT_CHUNK_OVERLAP == 64
    assert DEFAULT_CHUNK_OVERLAP < DEFAULT_CHUNK_SIZE


# ---- serialization ----------------------------------------------------

def test_chunk_to_dict():
    c = Chunk(
        chunk_id="abc::p0001::c000",
        doc_id="abc",
        filename="f.pdf",
        format=".pdf",
        page_no=1,
        index_in_page=0,
        text="hello",
        image_path=None,
    )
    d = c.to_dict()
    assert d["chunk_id"] == c.chunk_id and d["text"] == "hello"
    assert d["image_path"] is None
