"""Automated source-support verification (Phase 4 / 5b).

Each cited source must actually support the answer's claims. Default check
is a content-term overlap heuristic — fast, deterministic, offline. An
injectable LLM judge is available for stronger verification and is probed
by the e2e tests; on unparseable judge output the heuristic verdict is kept
(never fail harder than the evidence warrants).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from .qna_schema import QnaParseError, Source, coerce_bool

SUPPORTED_THRESHOLD = 0.25
WEAK_THRESHOLD = 0.10

STOPWORDS = frozenset(
    """a an and are as at be but by for from has have he her his i in is it its
    me my not of on or our she so that the their them there these they this to
    was we were what when where which who will with you your""".split()
)

_CONTENT_WORD = re.compile(r"[a-z]{3,}")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_NEGATION = re.compile(r"\b(?:no|not|never|none|without|hardly|scarcely)\b|n't\b")


def content_terms(text: str) -> set[str]:
    """Lowercased, stopword-filtered content terms of ``text``."""
    return {w for w in _CONTENT_WORD.findall(text.lower()) if w not in STOPWORDS}


def overlap_fraction(answer: str, excerpt: str) -> float:
    """Fraction of the answer's content terms present in the excerpt."""
    answer_terms = content_terms(answer)
    if not answer_terms:
        return 0.0
    excerpt_terms = content_terms(excerpt)
    return len(answer_terms & excerpt_terms) / len(answer_terms)


def _has_negation(text: str) -> bool:
    """True when the text contains an explicit negator (case-insensitive)."""
    return bool(_NEGATION.search(text.lower()))


def factual_conflict(answer: str, excerpt: str) -> bool:
    """True when the answer asserts facts the excerpt cannot support.

    The overlap heuristic ignores numbers and stopwords (e.g. ``not``), so
    "the exam is on October 20" and "the exam is on October 15" share every
    content term. Numbers and explicit negation are compared explicitly:

    - any answer number absent from the excerpt means support cannot be
      certified;
    - negation polarity is compared in BOTH directions with case
      normalization: an affirmative claim against a negated excerpt (and
      vice versa) contradicts it. "The exam is on Friday" vs "The exam is
      NOT on Friday" (either side negated) is a conflict; matching polarity
      ("...is not on Friday" vs "...is not on Friday") is not.

    Conservative by design: a negator on exactly one side of a claim the
    answer shares may downgrade an otherwise-correct answer to weak rather
    than certify uncertain support — the safe failure for a grounding layer.
    """
    answer_numbers = _NUMBER.findall(answer)
    if answer_numbers and any(n not in _NUMBER.findall(excerpt) for n in answer_numbers):
        return True
    return _has_negation(answer) != _has_negation(excerpt)


@dataclass(frozen=True)
class SourceCheck:
    index: int
    overlap: float
    verdict: str  # "supported" | "weak" | "unsupported"
    conflict: bool = False  # factual-conflict detected; verdict capped at "weak"


def _verdict(overlap: float, conflict: bool = False) -> str:
    if overlap < WEAK_THRESHOLD:
        return "unsupported"
    if conflict:
        # Content overlaps but the facts conflict (numbers, dates,
        # negation) — support can never be certified affirmatively.
        return "weak"
    if overlap >= SUPPORTED_THRESHOLD:
        return "supported"
    return "weak"


def check_sources(answer: str, sources: tuple[Source, ...]) -> tuple[SourceCheck, ...]:
    """Heuristic per-source support check, ordered by source index."""
    checks = []
    for i, source in enumerate(sources):
        overlap = overlap_fraction(answer, source.excerpt or "")
        conflict = factual_conflict(answer, source.excerpt) if source.excerpt else False
        checks.append(
            SourceCheck(index=i, overlap=overlap, verdict=_verdict(overlap, conflict), conflict=conflict)
        )
    return tuple(checks)


@dataclass(frozen=True)
class CitationCheck:
    index: int
    valid: bool
    reason: str
    image_path: str | None = None  # evidence-derived; never the model's raw path


def _normalize(text: str) -> str:
    """Case- and whitespace-collapsed text for excerpt authenticity checks."""
    return " ".join(text.split()).lower()


def _path_key(path: str | None) -> str:
    """POSIX-normalized path for comparisons (evidence paths are as_posix())."""
    if not path:
        return ""
    return str(Path(path)).replace("\\", "/")


def verify_citations(
    sources: tuple[Source, ...], evidence: Sequence[Source]
) -> tuple[CitationCheck, ...]:
    """Verify model citations against the evidence originally supplied.

    A citation is valid only when its (document, page) exists in the
    evidence AND its excerpt appears (verbatim, modulo case/whitespace)
    inside that evidence entry's excerpt. Fabricated or mismatched
    citations are flagged so the orchestrator can drop them — a model
    must never be allowed to invent sources.

    ``image_path`` is never trusted from the model: a cited path is kept
    only when it matches the matching evidence entry's own path, and the
    returned check carries the evidence-derived path (``None`` when the
    citation had none or the path was fabricated).
    """
    checks = []
    for i, src in enumerate(sources):
        matches = [e for e in evidence if e.document == src.document and e.page == src.page]
        if not matches:
            checks.append(
                CitationCheck(i, False, f"no evidence for ({src.document}, page {src.page})")
            )
        elif src.excerpt is None:
            image = next((e.image_path for e in matches
                          if e.image_path and src.image_path
                          and _path_key(e.image_path) == _path_key(src.image_path)), None)
            checks.append(CitationCheck(i, bool(image), "original image matches evidence" if image
                                        else "original image not found in evidence", image))
        elif any(e.excerpt and _normalize(src.excerpt) in _normalize(e.excerpt) for e in matches):
            if src.image_path:
                evidence_path = next(
                    (e.image_path for e in matches if e.image_path and _path_key(e.image_path) == _path_key(src.image_path)),
                    None,
                )
                if evidence_path is None:
                    checks.append(
                        CitationCheck(
                            i,
                            True,
                            f"excerpt matches evidence ({src.document}, page {src.page}); "
                            "image path discarded (not in evidence)",
                        )
                    )
                else:
                    checks.append(
                        CitationCheck(
                            i,
                            True,
                            f"excerpt matches evidence ({src.document}, page {src.page})",
                            image_path=evidence_path,
                        )
                    )
            else:
                checks.append(
                    CitationCheck(i, True, f"excerpt matches evidence ({src.document}, page {src.page})")
                )
        else:
            checks.append(
                CitationCheck(
                    i, False, f"excerpt not found in evidence ({src.document}, page {src.page})"
                )
            )
    return tuple(checks)


# -- optional LLM judge (e2e tier) ------------------------------------------

JUDGE_SYSTEM_PROMPT = (
    "You verify citations. Given a claim and one excerpt, reply with only a "
    "JSON object: {\"supported\": true|false, \"reason\": \"...\"}. "
    "Supported means the excerpt contains direct evidence for the claim."
)


def judge_sources(
    answer: str,
    sources: tuple[Source, ...],
    llm: Any,
    fallback: Callable[[str, tuple[Source, ...]], tuple[SourceCheck, ...]] = check_sources,
) -> tuple[SourceCheck, ...]:
    """LLM-judge pass over the heuristic verdicts (one model call per source)."""
    results: list[SourceCheck] = []
    for i, source in enumerate(sources):
        user_prompt = f"claim: {answer}\nexcerpt: {source.excerpt}"
        try:
            raw = llm.chat(
                [
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
            )
            verdict_obj = json.loads(_first_json_object(raw))
            supported = coerce_bool(verdict_obj.get("supported"))
        except (QnaParseError, json.JSONDecodeError, TypeError, ValueError):
            supported = None  # fall back to the heuristic below
        heuristic = overlap_fraction(answer, source.excerpt or "")
        conflict = factual_conflict(answer, source.excerpt) if source.excerpt else False
        if supported is None:
            final = _verdict(heuristic, conflict)
        elif supported:
            final = "supported" if not conflict else "weak"
        else:
            final = "unsupported"
        results.append(SourceCheck(index=i, overlap=heuristic, verdict=final, conflict=conflict))
    return tuple(results)


def _first_json_object(raw: str) -> str:
    """Minimal extractor for the judge's reply (reuses qna_schema semantics)."""
    from .qna_schema import _extract_json

    return json.dumps(_extract_json(raw))
