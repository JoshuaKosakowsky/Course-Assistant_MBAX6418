"""Prompt construction for the grounded Q&A (Phase 4).

The system prompt is generated from ``docs/PERSONA.md`` at call time — the
runtime uses the file, never a copy (PERSONA.md is the single source of
truth for the assistant's identity, per docs/PERSONA.md §Versioning).

The user message carries the question, numbered evidence excerpts with
source details, and (when available) the original page/slide images as
JPEG data URIs, capped at ``MAX_IMAGES`` so token use stays bounded.

The tiny image-encoding helper is self-contained here so this branch has
no dependency on the unmerged visual-index branch; it folds into a shared
helper once those PRs land.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Sequence

from PIL import Image

from .qna_schema import Source

PERSONA_FILE = Path(__file__).resolve().parents[1] / "docs" / "PERSONA.md"
MAX_IMAGES = 4

# Evidence-only contract appended to the persona at runtime.
_RULES = """\
## Evidence rules (non-negotiable)
- Answer ONLY from the evidence provided in this conversation. Never use
  outside knowledge to answer.
- Every claim must be supported by a cited source: document, page/slide,
  and an exact excerpt from that source. Show the excerpt itself.
- If the evidence does not contain the answer, say plainly that it is not
  in the materials — do not guess, and do not invent answers, sources,
  excerpts, page numbers, or images.
- When a diagram or chart in the provided images answers the question,
  reference that image and say what to look for in it.

## Output contract
Reply with ONLY a single JSON object, no prose outside it:
{
  "answer": "your answer to the question",
  "grounded": true,
  "sources": [
    {"document": "doc name", "page": 3,
     "excerpt": "exact text from that page",
     "image_path": "path to the original page/slide image if used"}
  ]
}
- \"grounded\": false and \"sources\": [] when the evidence does not cover
  the question. The answer must say plainly that it is not in the materials.
- Each source must reference evidence actually given to you."""


def load_persona(path: Path | str | None = None) -> str:
    """Read the persona file (docs/PERSONA.md by default)."""
    persona_path = Path(path) if path else PERSONA_FILE
    if not persona_path.is_file():
        raise FileNotFoundError(f"persona file not found: {persona_path}")
    return persona_path.read_text(encoding="utf-8")


def system_prompt(persona: str) -> str:
    return f"{persona}\n\n{_RULES}"


def _image_to_data_uri(image: Image.Image) -> str:
    """JPEG data URI for sending page/slide images to vision chat endpoints."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _evidence_text(question: str, evidence: Sequence[Source]) -> str:
    lines = [f"Question: {question}", "", "Evidence (numbered):"]
    for i, source in enumerate(evidence, start=1):
        lines.append(f"[{i}] ({source.document}, page {source.page}): {source.excerpt}")
    return "\n".join(lines)


def _image_parts(evidence: Sequence[Source], max_images: int) -> list[dict]:
    parts: list[dict] = []
    for source in evidence:
        if source.image_path and len(parts) < max_images:
            path = Path(source.image_path)
            if path.is_file():
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_to_data_uri(Image.open(path))},
                    }
                )
    return parts


def build_messages(
    question: str,
    evidence: Sequence[Source] = (),
    *,
    persona_path: Path | str | None = None,
    max_images: int = MAX_IMAGES,
) -> list[dict]:
    """Build the system + user messages for a grounded Q&A turn."""
    persona = load_persona(persona_path)
    user_content: list[dict] = [{"type": "text", "text": _evidence_text(question, evidence)}]
    user_content.extend(_image_parts(evidence, max_images))
    return [
        {"role": "system", "content": system_prompt(persona)},
        {"role": "user", "content": user_content},
    ]
