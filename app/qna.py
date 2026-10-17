"""Grounded question answering orchestrator (Phase 4, issue #5).

Pipeline: persona + evidence prompt -> vision LLM -> tolerant parse ->
grounding normalization -> automated source-support check.

On malformed output it retries (configurable, default once), then returns
an honest fallback instead of surfacing unvalidated model text. All
external dependencies (the LLM) are injected, so the whole core is
testable offline with stubs; production wires :class:`ClassVisionClient`.

The evidence pack input uses the same :class:`Source` shape Phase 3 will
produce (document, page, excerpt, image_path) — coordination flagged on
issue #5 / #4.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .prompts import build_messages
from .qna_schema import ParsedAnswer, QnaParseError, Source, parse_answer
from .source_check import CitationCheck, SourceCheck, check_sources, verify_citations

logger = logging.getLogger(__name__)

MISSING_INFO_FALLBACK = (
    "I don't have that in the course materials, so I can't answer from "
    "evidence. If you can point me at the right material or topic, I'll "
    "look it up."
)


@dataclass(frozen=True)
class QnaResult:
    answer: str
    sources: tuple[Source, ...] = ()
    grounded: bool = False
    source_checks: tuple[SourceCheck, ...] = ()
    citation_checks: tuple[CitationCheck, ...] = ()
    malformed: bool = False


@dataclass
class QnaConfig:
    retries: int = 1  # extra attempts after a malformed reply
    max_images: int = 4
    persona_path: Path | None = None


class QnaOrchestrator:
    """Answer a question from evidence with structured, verified output."""

    def __init__(self, llm: Any, config: QnaConfig | None = None, *, visual_verifier=None):
        self.llm = llm
        self.config = config or QnaConfig()
        self.visual_verifier = visual_verifier

    def answer(self, question: str, evidence: Sequence[Source] = ()) -> QnaResult:
        messages = build_messages(
            question,
            evidence,
            persona_path=self.config.persona_path,
            max_images=self.config.max_images,
        )

        parsed: ParsedAnswer | None = None
        malformed = False
        for attempt in range(self.config.retries + 1):
            raw = self.llm.chat(messages)
            try:
                parsed = parse_answer(raw)
                break
            except QnaParseError as exc:
                logger.warning(
                    "Malformed Q&A output (attempt %d/%d): %s",
                    attempt + 1,
                    self.config.retries + 1,
                    exc,
                )
                if attempt < self.config.retries:
                    continue
                malformed = True

        if parsed is None or malformed:
            return QnaResult(answer=MISSING_INFO_FALLBACK, grounded=False, malformed=True)

        answer = parsed.answer if parsed.answer.strip() else MISSING_INFO_FALLBACK

        # A grounded answer must cite sources that (a) exist in the supplied
        # evidence — fabricated document/page/excerpt/image citations are
        # dropped, with image paths always derived from evidence, never the
        # model — and (b) actually support the answer's claims. Any
        # ungrounded path returns the evidence-safe fallback rather than the
        # model's raw text, so unsupported claims can never reach the UI.
        grounded = parsed.grounded and bool(parsed.sources)
        if not grounded:
            logger.warning("Model answer not grounded; returning evidence-safe fallback")
            return QnaResult(answer=MISSING_INFO_FALLBACK, grounded=False)

        sources = tuple(parsed.sources)
        citation_checks = verify_citations(sources, tuple(evidence))
        verified = tuple(
            Source(
                document=s.document,
                page=s.page,
                excerpt=s.excerpt,
                image_path=ck.image_path,  # evidence-derived (None if fabricated/missing)
            )
            for s, ck in zip(sources, citation_checks)
            if ck.valid
        )
        if not verified:
            logger.warning("No citation verified against evidence; returning evidence-safe fallback")
            return QnaResult(
                answer=MISSING_INFO_FALLBACK, grounded=False, citation_checks=citation_checks
            )

        checks = check_sources(answer, verified)
        if self.visual_verifier is not None:
            checks = tuple(
                SourceCheck(ck.index, ck.overlap, "supported", False)
                if (source.image_path and ck.verdict != "supported"
                    and self.visual_verifier(answer, source) is True) else ck
                for source, ck in zip(verified, checks)
            )
        supported = tuple(s for s, ck in zip(verified, checks) if ck.verdict == "supported")
        if not supported:
            logger.warning("No source confidently supports the answer; returning evidence-safe fallback")
            return QnaResult(
                answer=MISSING_INFO_FALLBACK,
                grounded=False,
                citation_checks=citation_checks,
                source_checks=checks,
            )
        presented = tuple(s for s, ck in zip(verified, checks) if ck.verdict != "unsupported")

        return QnaResult(
            answer=answer,
            sources=presented,
            grounded=True,
            source_checks=checks,
            citation_checks=citation_checks,
        )
