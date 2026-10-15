"""Unit tests for prompt construction (persona, evidence, images)."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from app.prompts import MAX_IMAGES, build_messages, load_persona, system_prompt
from app.qna_schema import Source

pytestmark = pytest.mark.unit

PERSONA_TEXT = "You are the course assistant persona. Be Socratic."  # noqa: S105 (test-only)


@pytest.fixture
def persona_file(tmp_path):
    p = tmp_path / "persona.md"
    p.write_text(PERSONA_TEXT, encoding="utf-8")
    return p


def test_system_prompt_embeds_persona(persona_file):
    persona = load_persona(persona_file)
    prompt = system_prompt(persona)
    assert "You are the course assistant persona" in prompt
    assert "only from the evidence" in prompt.lower()
    assert "not in the materials" in prompt
    assert "grounded" in prompt


def test_load_persona_defaults_to_repo_file():
    persona = load_persona()
    assert "course assistant" in persona.lower()
    assert "never fabricate" in persona.lower()


def test_load_persona_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_persona(tmp_path / "missing.md")


def test_user_message_carries_question_and_numbered_evidence(persona_file):
    evidence = [
        Source(document="rag-lecture.pdf", page=3, excerpt="RAG combines retrieval with generation."),
        Source(document="syllabus.pdf", page=1, excerpt="Office hours on Tuesdays."),
    ]
    messages = build_messages("What is RAG?", evidence, persona_path=persona_file)
    user = messages[1]["content"]
    assert isinstance(user, list)
    text = user[0]["text"]
    assert "What is RAG?" in text
    assert "[1] (rag-lecture.pdf, page 3): RAG combines" in text
    assert "[2] (syllabus.pdf, page 1): Office hours" in text


def test_evidence_images_are_included_as_data_uris(persona_file, tmp_path):
    img = tmp_path / "slide.png"
    Image.new("RGB", (32, 32), (30, 120, 220)).save(img)
    evidence = [Source(document="d", page=1, excerpt="x", image_path=str(img))]
    messages = build_messages("q", evidence, persona_path=persona_file)
    user = messages[1]["content"]
    image_parts = [p for p in user if isinstance(p, dict) and p.get("type") == "image_url"]
    assert len(image_parts) == 1
    assert image_parts[0]["image_url"]["url"].startswith("data:image/")
    assert ";base64," in image_parts[0]["image_url"]["url"]


def test_max_images_cap(persona_file, tmp_path):
    evidence = []
    for i in range(5):
        img = tmp_path / f"slide_{i}.png"
        Image.new("RGB", (16, 16), (i * 10, 0, 0)).save(img)
        evidence.append(Source(document="d", page=i + 1, excerpt="x", image_path=str(img)))
    messages = build_messages("q", evidence, persona_path=persona_file, max_images=2)
    user = messages[1]["content"]
    image_parts = [p for p in user if isinstance(p, dict) and p.get("type") == "image_url"]
    assert len(image_parts) == 2
    assert MAX_IMAGES == 4  # default sanity: keep budgets explicit


def test_missing_image_file_is_skipped_not_fatal(persona_file):
    evidence = [Source(document="d", page=1, excerpt="x", image_path="does/not/exist.png")]
    messages = build_messages("q", evidence, persona_path=persona_file)
    user = messages[1]["content"]
    image_parts = [p for p in user if isinstance(p, dict) and p.get("type") == "image_url"]
    assert image_parts == []
