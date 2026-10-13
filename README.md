# Course Assistant — MBAX 6418-801 (Assignment 2)

![CI](https://github.com/JoshuaKosakowsky/Course-Assistant_MBAX6418/actions/workflows/ci.yml/badge.svg)

A course assistant that answers questions and generates practice quizzes from
course materials (slides, syllabus, and other files downloaded from Canvas).
Built on hybrid RAG: keyword search + text embeddings + visual embeddings,
with source-grounded evidence — including original page/slide images.

**Status:** Phase 0 — repo & collaboration scaffold. No end-to-end pipeline yet.
Work breakdown, ownership, and completion checks live in
[PLAN.md](PLAN.md) and the GitHub issues.

---

## Features (planned)

- **Ingest** course files (PDF and other formats — exact list TBD and
  documented once decided). Preserve text *and* original page/slide images,
  keeping source metadata (document, page/slide, section).
- **Q&A** grounded in the materials, including diagrams and charts, with
  structured `{answer, sources}` output. Each source shows the document and
  page/slide or section with an actual excerpt or screenshot. Missing
  information is acknowledged — never invented, never fabricated citations.
- **Quiz generation** by course material, with an optional topic filter.
  Multiple-choice questions with scores and explanations. The answer key is
  fixed; solutions are hidden until the student answers or requests them.
  Feedback shows the document, page/slide, and source excerpt/image.

## Architecture (target)

```
course files ──> parse (text + page/slide images)
                    ├──> text chunks ──> keyword index (bm25s)  ─┐
                    │                 └─> text embedding index ──┼─> fuse + rerank ─> evidence pack ─> LLM ─> {answer, sources}
                    └──> page images ─> visual embedding index ──┘                     (incl. images)          Gradio UI
```

Text and visual indexes stay **separate**; results are combined and
reranked before the best evidence (text + images) is passed to the LLM.

## Setup

```bash
git clone <repo-url> && cd Course-Assistant_MBAX6418
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then fill in real values (see below)
```

### Class services

The LLM, embeddings, reranking, and document parsing run on the class
endpoints at `dobolyi.com`, ports **9001 and above**. Connection details
(keyed, per-service) come from the project team — see `.env.example` for the
template. Fill `.env` locally; the file is gitignored.

## Security (class policy)

- Keys live **only** in server-side environment variables or the local
  `.env` — never in the UI, browser code, logs, errors, screenshots,
  documentation, test output, or GitHub.
- Committed config examples contain **dummy values only** (`.env.example`).

## Testing

Repeatable automated checks (pytest) run per phase — see PLAN.md. Test
results, screenshots, and findings are saved in the repo so the team can
review what was verified and what remains unchecked.

## Team

- Ownership of substantial tasks via GitHub issues (with completion checks).
- Work flows through branches, commits, pull requests, and reviews.

## Contributing

- **Agents:** read [AGENTS.md](AGENTS.md) first — it governs AI coding agents.
- **Contributors:** [CONTRIBUTING.md](CONTRIBUTING.md) — workflow, issue
  claiming, bug reports, review expectations.
- Commit conventions: [docs/COMMITS.md](docs/COMMITS.md)
- Test tiers: [docs/TESTING.md](docs/TESTING.md) — `pytest` = unit + smoke;
  `pytest -m e2e` needs class keys.
