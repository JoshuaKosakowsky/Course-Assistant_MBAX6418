"""Structured Q&A output: {answer, sources} with tolerant validation.

The vision LLM returns free text; we parse it as JSON (allowing markdown
code fences and surrounding prose), coerce types defensively (e.g. page
numbers arriving as strings), and treat anything that does not validate as
a ``QnaParseError`` so callers can retry or fall back honestly — never
surface unvalidated model output as an answer.

The same :class:`Source` shape serves both the evidence pack handed to the
model (Phase 3 output) and the sources the model cites in its answer —
each carries document, page/slide, an exact excerpt, and (when available)
the original page/slide image path.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


class QnaParseError(ValueError):
    """Raised when model output cannot be parsed into a valid answer."""


@dataclass(frozen=True)
class Source:
    document: str
    page: int
    excerpt: str
    image_path: str | None = None


@dataclass(frozen=True)
class ParsedAnswer:
    answer: str
    sources: tuple[Source, ...]
    grounded: bool


def _extract_json(raw: str) -> object:
    """Return the first balanced JSON object in ``raw``, fences/prose aside."""
    text = raw.strip()
    # Peel a leading ```json / ``` fence and a trailing fence if present.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    if start == -1:
        raise QnaParseError("no JSON object found in model output")
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise QnaParseError("unbalanced JSON object in model output")


def coerce_bool(value: object) -> bool:
    """Strictly validate a JSON boolean field.

    ``bool("false")`` is ``True``, so strings must be checked explicitly.
    Missing/invalid values raise :class:`QnaParseError` so callers fail
    conservatively (retry or fall back) instead of misinterpreting input.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "1", "yes"):
            return True
        if low in ("false", "0", "no"):
            return False
        raise QnaParseError(f"invalid boolean value: {value!r}")
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    raise QnaParseError(f"invalid boolean value: {value!r}")


def _coerce_page(value: object) -> int:
    if isinstance(value, bool):
        raise QnaParseError(f"invalid page value: {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    raise QnaParseError(f"invalid page value: {value!r}")


def parse_answer(raw: str) -> ParsedAnswer:
    """Parse and validate model output into :class:`ParsedAnswer`.

    Contract: a JSON object with ``answer`` (non-empty str), ``sources``
    (list of {document, page, excerpt, image_path?}), and optional
    ``grounded`` (bool). Unknown extra keys are tolerated; anything else
    raises :class:`QnaParseError`.
    """
    try:
        obj = _extract_json(raw)
    except QnaParseError:
        raise
    except (json.JSONDecodeError, TypeError) as exc:
        raise QnaParseError(f"invalid JSON in model output: {exc}") from exc

    if not isinstance(obj, dict):
        raise QnaParseError("expected a JSON object, got %s" % type(obj).__name__)

    answer = obj.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise QnaParseError("missing non-empty 'answer'")

    sources_raw = obj.get("sources", [])
    if sources_raw is None:
        sources_raw = []
    if not isinstance(sources_raw, list):
        raise QnaParseError("'sources' must be a list")

    sources: list[Source] = []
    for entry in sources_raw:
        if not isinstance(entry, dict):
            raise QnaParseError("each source must be a JSON object")
        document = entry.get("document")
        excerpt = entry.get("excerpt")
        if not isinstance(document, str) or not document.strip():
            raise QnaParseError("source missing non-empty 'document'")
        if not isinstance(excerpt, str) or not excerpt.strip():
            raise QnaParseError("source missing non-empty 'excerpt'")
        image_path = entry.get("image_path")
        sources.append(
            Source(
                document=document.strip(),
                page=_coerce_page(entry.get("page")),
                excerpt=excerpt.strip(),
                image_path=image_path if isinstance(image_path, str) and image_path else None,
            )
        )

    grounded = obj.get("grounded", True)
    return ParsedAnswer(answer=answer.strip(), sources=tuple(sources), grounded=coerce_bool(grounded))
