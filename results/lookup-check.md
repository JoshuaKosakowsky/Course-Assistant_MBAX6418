# Lookup verification

## Scope

Added a read-only Lookup section beneath the Library table. It accepts a question
and returns up to five existing BM25 text passages from currently registered PDFs,
with filename and physical page number. It does not generate an LLM answer or call
class endpoints. No new runtime dependencies were added.

Files: `app/lookup.py`, `app/ui.py`, `tests/smoke/test_lookup.py`, and README usage.

## Tests actually run

- Baseline `.venv/Scripts/python.exe -m pytest -q`: **154 passed, 2 deselected**.
- Observed failing tests before implementing the lookup path, validation/error
  handling, and UI wiring; reran the targeted tests after each implementation step.
- Final `.venv/Scripts/python.exe -m pytest -q`: **173 passed, 2 deselected**.
- One pre-existing ChromaDB `asyncio.iscoroutinefunction` deprecation warning is
  present in both baseline and final runs. No class-service e2e tests were run.

The new tests exercise real generated PDFs, parsing, DocumentStore, and BM25.
They cover citations, empty/invalid/oversized questions, empty Library, missing
indexer, no lexical matches, stopword-only questions, retrieval failures without
internal error disclosure, stale/deleted documents, deletion during retrieval,
PDF-only filtering, HTML escaping, unchanged on-disk Library content, the five-hit
limit, and both Gradio click/submit event bindings.

## Browser verification

Started a separate local Gradio instance on port 7861 backed by a temporary,
synthetic-only library. No test fixtures were inserted into the user's Library.

- Lookup is below the Library table.
- Clicking Look up for “How is free cash flow calculated?” returned the synthetic
  source `lookup-test-finance.pdf`, Page 2, and its exact expected excerpt.
- Enter submission for an unmatched query displayed the no-matching-passages
  message (the first CDP key event omitted a native keycode; repeating with a real
  Enter event succeeded).
- `lookup-demo.png` captures only this generated synthetic PDF.
- The temporary verification server was stopped after verification.

Restarted the actual Course Assistant on port 7860 and checked that the live Gradio
configuration exposes both `lookup` and `lookup_submit`. Opened the updated app in
the desktop preview.

## Preserved constraints

AST comparison against git HEAD confirmed that `add_file` and `remove_selected`
are unchanged. Existing add/remove/load callback expressions are also unchanged.
The diff in `app/ui.py` only adds an import and the new Lookup components/events.
No document-store, ingestion, parsing, or removal code was modified.

## Limits / workflow status

Lookup is keyword-based rather than semantic Q&A. It needs extracted searchable
PDF text; it does not perform OCR or search original PPTX records. Previously
rendered results are a snapshot and must be refreshed by a new lookup after a
Library change. Real course-text lookup quality was not evaluated; browser
result verification used synthetic text.

Changes are local on `feat/library-lookup`; no commit, push, issue claim, or PR was
performed. GitHub CLI was unavailable on PATH and at its standard Windows install
location, so the remote issue-claim workflow remains outstanding.
