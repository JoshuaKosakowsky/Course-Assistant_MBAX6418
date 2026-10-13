# AGENTS.md — Operating Guide for AI Coding Agents

Read this file first; it governs all work in this repo. Humans should read
CONTRIBUTING.md. Detail lives in docs/ (COMMITS.md, TESTING.md) and PLAN.md.

## Project

Course assistant for MBAX 6418-801 (Assignment 2): hybrid-RAG Q&A and quiz
generation over course materials. Python + Gradio; bm25s, chromadb,
langchain-text-splitters. Class services (vision LLM, embeddings, reranker,
document parser) run at dobolyi.com ports 9001+ — connection details live in
the local `.env`, never in the repository.

Status: Phase 0 (scaffold). Work is tracked as GitHub issues #1–#9 with
completion checks (see PLAN.md). This is a collaborative team repo: claim an
issue before starting, and follow one issue = one branch = one PR.

## Hard rules

1. **Secrets.** Never write a real API key into any tracked file, log,
   error, screenshot, issue, PR, or chat message. Keys exist only in the
   gitignored `.env`. Committed examples carry dummy placeholders
   (`.env.example`).
2. **Honesty.** The app answers only from retrieved evidence; it must
   acknowledge missing information and never fabricate answers, sources, or
   citations.
3. **Shared infrastructure.** Class endpoints are keyed, shared services.
   Only opt-in e2e tests (marked) call them — treated as deliberate,
   documented usage.

## Commands

```bash
pip install -r requirements.txt   # setup
cp .env.example .env              # then fill real values (never commit)
pytest                            # unit + smoke (fast, offline)
pytest -m e2e                     # real class endpoints; needs .env + network
```

## Workflow

1. Comment on the issue to claim it (team chooses ownership).
2. Branch from `main`: `type/slug` matching the issue (e.g. `feat/ingest`).
3. Commit messages per docs/COMMITS.md (type, ≤50-char subject, imperative
   mood, body wrapped at 72).
4. Add or update tests with your change (docs/TESTING.md) — unit for logic,
   smoke for wiring, e2e only for real-endpoint behavior.
5. Run `pytest` locally and re-scan your diff for secrets before pushing.
6. Open the PR using the template; link the issue (`Closes #N`).

## Definition of done (every contribution)

- Issue linked · tests green (correct tier) · no secrets in the diff
- Docs updated where behavior changed
- PR description states plainly what was tested and what was not

## Reading list

README.md (overview/architecture) · PLAN.md (phases) · docs/COMMITS.md ·
docs/TESTING.md · CONTRIBUTING.md (human workflow)
