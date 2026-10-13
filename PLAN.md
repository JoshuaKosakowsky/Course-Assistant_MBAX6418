# Plan — Course Assistant (Assignment 2)

High-level plan for the course-assistant build. Each phase lists its
**completion checks**; substantial phases become GitHub issues so the team
can choose ownership. Phases are ordered to build incrementally — every
phase ends with something runnable and tested.

Status legend: `pending` (not started) · `in progress` · `done`.

---

## Phase 0 — Repo & collaboration setup (this scaffold)
- [x] Git repo initialized (`main` branch), `.gitignore`, README, `.env.example` (dummy values only)
- [ ] GitHub remote created; branch policy agreed (working branches + PRs; `main` protected or reviewed)
- [ ] Team fills `.env` with class endpoint details (dobolyi.com, ports 9001+) — **blocker for Phase 1+**
- [ ] Substantial tasks filed as GitHub issues with completion checks; ownership chosen
- **Definition of done:** any teammate can clone, install, and run the empty app; issues exist with checks.

## Phase 1 — Materials ingestion
- [ ] Decide and document accepted formats (PDF first; others TBD);
      reject clearly with a helpful message otherwise
- [ ] Parse documents: extract text preserving source metadata (doc, page/slide, section); render/keep original page and slide images
- [ ] Sample corpus in `data/` (gitignored): syllabus + at least one slide deck from Canvas
- [ ] Small CLI/script: `python -m app.ingest <file>` -> verified text + images on disk
- **Definition of done:** a sample PDF and deck produce text + page/slide images; spot-check shows fidelity (no garbled text, images intact).

## Phase 2 — Indexing (chunking + separate stores)
- [ ] Chunk text with `langchain-text-splitters`, carrying source metadata on every chunk
- [ ] Keyword index (`bm25s`) over chunks
- [ ] Text embedding index (`chromadb`) — separate from visual
- [ ] Visual embedding index over page/slide images — separate from text
- [ ] Ingest script indexes the sample corpus; record chunk/index counts
- **Definition of done:** each of the three indexes is independently queryable; counts logged; one sample query returns plausible hits per index.

## Phase 3 — Retrieval, fusion, rerank
- [ ] Combine keyword + text results (e.g. RRF or score fusion); fetch top candidates
- [ ] Candidate images retrieved via visual index
- [ ] Multimodal reranker (class endpoint) reranks candidates; log and save a comparison (top-k before/after)
- [ ] Evidence pack: text excerpts + matching page/slide images, each with source details
- **Definition of done:** end-to-end retrieval on 5+ sample questions returns evidence with images; rerank measurably improves top-1 for at least one demo query.

## Phase 4 — Q&A with structured output
- [ ] Prompt template with evidence and "answer from sources only" instructions
- [ ] Vision-capable LLM (class endpoint) answers questions incl. diagrams/charts
- [ ] Structured output `{answer, sources}`; schema-validate; check each source actually supports the answer (automated + manual spot-check)
- [ ] Missing-info behavior: explicit "not in the materials" — never invent answers or citations
- [ ] Gradio tab 1: pick material(s), ask, see answer + clickable evidence (excerpt + image)
- **Definition of done:** test suite (pytest) covers: answer with sources, image-grounded answer, out-of-scope question handled honestly, malformed output caught.

## Phase 5 — Quiz generation
- [ ] Choose material + optional topic filter -> N multiple-choice questions with explanations
- [ ] Fixed answer key per generation; solutions hidden until answered or requested
- [ ] Scoring + per-question feedback linking to document/page/section/excerpt/image
- [ ] Gradio tab 2: quiz flow (generate → answer → score → review)
- **Definition of done:** full quiz flow works; answer key stable across runs of the same generation; solutions provably hidden pre-answer; feedback shows original evidence.

## Phase 6 — Polish, testing, comparison
- [ ] One design comparison (proposal: rerank vs. no-rerank on answer quality; alternatives: chunk size 250/500/1000, hybrid vs. keyword-only) — scripted, results saved in `results/`
- [ ] Repeatable automated checks for the whole pipeline (`pytest`, one command)
- [ ] Error handling: missing keys, network down, unsupported file, empty index
- **Definition of done:** `pytest` green; comparison results + screenshots saved; README documents what was tested and what remains unchecked.

## Phase 7 — README, findings, limitations
- [ ] Setup guide verified from a clean clone
- [ ] Screenshots of Q&A + quiz flows and retrieval evidence
- [ ] Findings, design decisions, and known limitations stated plainly
- **Definition of done:** a stranger can set up and run the app from README; reviewer can see proven tests and honest limitations.

---

## Security policy (from assignment)
- Class API keys: server-side env vars or local `.env` only.
- Never expose keys in the interface, browser code, logs, errors, screenshots, docs, test results, or GitHub.
- Committed configuration examples contain dummy values (`cp .env.example .env`).

## Open questions for the team
1. Class endpoint details (ports, keys, per-service formats) — who has them?
2. GitHub repo location/visibility (CU org vs. personal; private vs. public)?
3. Accepted input formats beyond PDF (PPTX? DOCX? images with text?).
4. The one design comparison (proposal: rerank on/off) — agree or swap.
