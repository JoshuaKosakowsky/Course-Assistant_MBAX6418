"""Unit tests for the {answer, sources} schema parsing/validation."""

from __future__ import annotations

import pytest

from app.qna_schema import QnaParseError, Source, parse_answer

VALID = (
    '{"answer": "Retrieval augmented generation.", "grounded": true, '
    '"sources": [{"document": "rag-lecture.pdf", "page": 3, '
    '"excerpt": "RAG combines retrieval with generation."}]}'
)

pytestmark = pytest.mark.unit


def test_parses_minimal_valid_answer():
    parsed = parse_answer(VALID)
    assert parsed.answer == "Retrieval augmented generation."
    assert parsed.grounded is True
    assert len(parsed.sources) == 1
    src = parsed.sources[0]
    assert src.document == "rag-lecture.pdf"
    assert src.page == 3
    assert "RAG combines" in src.excerpt
    assert src.image_path is None


def test_grounded_false_with_empty_sources():
    parsed = parse_answer('{"answer": "Not covered.", "grounded": false, "sources": []}')
    assert parsed.grounded is False
    assert parsed.sources == ()


def test_grounded_string_false_is_parsed_as_false():
    # bool("false") is True — must not happen for model output.
    parsed = parse_answer('{"answer": "Not covered.", "grounded": "false", "sources": []}')
    assert parsed.grounded is False


def test_grounded_string_true_and_numeric_values():
    assert parse_answer('{"answer": "x", "grounded": "true"}').grounded is True
    assert parse_answer('{"answer": "x", "grounded": 1}').grounded is True
    assert parse_answer('{"answer": "x", "grounded": 0}').grounded is False


def test_grounded_invalid_value_raises():
    with pytest.raises(QnaParseError):
        parse_answer('{"answer": "x", "grounded": "maybe", "sources": []}')
    with pytest.raises(QnaParseError):
        parse_answer('{"answer": "x", "grounded": null, "sources": []}')


def test_tolerates_code_fences_and_surrounding_prose():
    raw = 'Here you go:\n```json\n' + VALID + '\n```\nHope that helps.'
    parsed = parse_answer(raw)
    assert parsed.answer == "Retrieval augmented generation."


def test_page_number_arriving_as_string_is_coerced():
    raw = VALID.replace('"page": 3', '"page": "3"')
    assert parse_answer(raw).sources[0].page == 3


def test_image_path_is_kept_when_present():
    raw = VALID.replace(
        '"excerpt": "RAG combines retrieval with generation."',
        '"excerpt": "RAG combines retrieval with generation.", "image_path": "data/slide_3.png"',
    )
    assert parse_answer(raw).sources[0].image_path == "data/slide_3.png"


def test_extra_unknown_keys_are_tolerated():
    raw = VALID.replace('"grounded": true', '"grounded": true, "note": "x", "meta": 1')
    assert parse_answer(raw).answer.startswith("Retrieval")


def test_missing_answer_raises():
    with pytest.raises(QnaParseError):
        parse_answer('{"sources": []}')


def test_empty_answer_raises():
    with pytest.raises(QnaParseError):
        parse_answer('{"answer": "   ", "sources": []}')


def test_missing_document_or_excerpt_raises():
    with pytest.raises(QnaParseError):
        parse_answer('{"answer": "x", "sources": [{"page": 1, "excerpt": "y"}]}')
    with pytest.raises(QnaParseError):
        parse_answer('{"answer": "x", "sources": [{"document": "d", "page": 1}]}')


def test_missing_page_raises():
    with pytest.raises(QnaParseError):
        parse_answer('{"answer": "x", "sources": [{"document": "d", "excerpt": "y"}]}')


def test_invalid_page_raises():
    with pytest.raises(QnaParseError):
        parse_answer('{"answer": "x", "sources": [{"document": "d", "page": "three", "excerpt": "y"}]}')


def test_sources_must_be_a_list():
    with pytest.raises(QnaParseError):
        parse_answer('{"answer": "x", "sources": {"document": "d"}}')


def test_not_json_raises():
    with pytest.raises(QnaParseError):
        parse_answer("sorry, I cannot answer that")


def test_unbalanced_json_raises():
    with pytest.raises(QnaParseError):
        parse_answer('{"answer": "x", "sources": []')


def test_no_object_raises():
    with pytest.raises(QnaParseError):
        parse_answer("[1, 2, 3]")
