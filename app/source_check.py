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
from typing import Any, Callable

from .qna_schema import QnaParseError, Source

SUPPORTED_THRESHOLD = 0.25
WEAK_THRESHOLD = 0.10

STOPWORDS = frozenset(
    """a an and are as at be but by for from has have he her his i in is it its
    me my not of on or our she so that the their them there these they this to
    was we were what when where which who will with you your""".split()
)

_CONTENT_WORD = re.compile(r"[a-z]{3,}")


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


@dataclass(frozen=True)
class SourceCheck:
    index: int
    overlap: float
    verdict: str  # "supported" | "weak" | "unsupported"


def _verdict(overlap: float) -> str:
    if overlap >= SUPPORTED_THRESHOLD:
        return "supported"
    if overlap >= WEAK_THRESHOLD:
        return "weak"
    return "unsupported"


def check_sources(answer: str, sources: tuple[Source, ...]) -> tuple[SourceCheck, ...]:
    """Heuristic per-source support check, ordered by source index."""
    checks = []
    for i, source in enumerate(sources):
        overlap = overlap_fraction(answer, source.excerpt)
        checks.append(SourceCheck(index=i, overlap=overlap, verdict=_verdict(overlap)))
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
            supported = bool(verdict_obj.get("supported"))
        except (QnaParseError, json.JSONDecodeError, TypeError, ValueError):
            supported = None  # fall back to the heuristic below
        heuristic = overlap_fraction(answer, source.excerpt)
        if supported is None:
            final = _verdict(heuristic)
        elif supported:
            final = "supported"
        else:
            final = "unsupported"
        results.append(SourceCheck(index=i, overlap=heuristic, verdict=final))
    return tuple(results)


def _first_json_object(raw: str) -> str:
    """Minimal extractor for the judge's reply (reuses qna_schema semantics)."""
    from .qna_schema import _extract_json

    return json.dumps(_extract_json(raw))
