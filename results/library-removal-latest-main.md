# Library-row removal fix on latest main

Base: `75f39be` (origin/main). Local work branch: `fix/library-row-removal`.

## Changes

Reapplied stable document-ID row selection, proper Dropdown choices/value updates,
invalid-ID rejection, and clearing the selection after refresh/removal. Kept the
current tabbed UI and existing quiz-material `.then(...)` refresh callbacks.

No visible UI definitions were changed: comparison with the base version matched
all 48 non-State component definitions. Only invisible session State and event
wiring were added. The add-file, document-store, and indexing implementations
were not changed.

## Executed verification

- Baseline offline suite: **831 passed, 1 skipped, 14 deselected**.
- Regression tests observed failing before the selection/wiring/validation fixes.
- Latest full offline suite: **861 passed, 1 skipped, 14 deselected**.
- Existing ChromaDB warning and additional Gradio coroutine-function deprecation
  warnings are dependency warnings, not failing tests.
- Actual Gradio process_api integration checks session State, row selection,
  removal, dropdown clearing, surviving file bytes, target-only keyword/vector
  purging, and the existing quiz-material refresh retaining all remaining docs.
- Regressions also cover duplicate filenames, stale/deleted rows, invalid indices,
  deselection, malformed IDs, and mismatched table/choice snapshots.
- `git diff --check` passes.

## Browser verification

An isolated loopback Gradio UI on port 7862 used three generated PDFs under a
scratch directory. Clicking beta populated the existing removal dropdown;
Remove selected deleted only beta, cleared the field, and retained alpha/gamma.
Lookup then found no beta-only biology text while still retrieving alpha's
astronomy passage. No real Library documents were deleted during these checks.

No live class-endpoint tests were run.

## Single-snapshot identity follow-up

The filename-only alignment guard has been removed. `_library_updates(store,
message)` captures `tuple(store.list())` once and passes that same tuple to table
and dropdown construction, deriving row IDs from those exact document records.
Load/add/remove refreshes all use this path. Independently rendered legacy handler
views are not reused in the displayed update; only the operation status is reused.

The regression creates two distinct PDFs named `same.pdf`, with different page
counts, and makes any second library listing return the IDs in reverse order.
Before the fix, the actual Gradio load callback displayed the first snapshot's
metadata but stored the second snapshot's IDs. After the fix it performs one read,
keeps metadata/options/IDs aligned, and removes the selected identity rather than
its same-named neighbor. Additional tests cover empty snapshots and stale views
returned by add/remove handlers. The 48 visible component definitions still match
HEAD; the UI and existing quiz refresh callbacks are unchanged.
