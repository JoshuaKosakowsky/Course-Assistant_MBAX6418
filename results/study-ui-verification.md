# Connected study UI verification

Snapshot: Issue #7 integration branch, updated through main `faec5b5`
(including PR #29's single-snapshot library selection fix).

- Final full offline suite: **884 passed, 1 skipped, 14 deselected**.
  Live class-service tests were excluded. Tracked-file secret checks passed.
- The normal `python -m app.ui` entry point also booted from the clean review
  copy without a `.env` or course materials: HTTP 200, keyword-only mode,
  model generation safely unavailable. Dependencies were installed in a
  separate Python 3.12 environment; this is not a second teammate's full
  installation walkthrough.
- Browser fixture: `python -m scripts.preview_study_ui --port 7862`.
  It constructs a synthetic PDF, real document store/parser/BM25 retrieval,
  real Q&A and quiz adapters, and an explicit deterministic injected model.
  No real course materials, keys, or remote services are used.
- Q&A: selected the document, asked how BM25 retrieves documents, observed
  the checked answer, filename, physical page number, excerpt, and original
  page image. Verified light and dark readability. Corrected gallery sizing
  after observing that Gradio initially cropped the original image.
- Quiz: selected the same document, topic BM25, one question, seed 42.
  Before answering the browser exposed the prompt and four options without
  explanation or answer key. Selected BM25 and submitted: score 1/1, correct
  answer, explanation, original excerpt, and verified page image appeared.
- Offline integration tests also cover missing evidence (no model call),
  unavailable services, selection/removal, escaped HTML, production service
  binding after index reconciliation, quiz provider/replay, image-only
  citations, and strict fail-closed visual support verdicts.
- After merging the library-selection change, the single document snapshot,
  duplicate-filename identity regressions, Gradio removal callbacks and both
  study-material pickers pass together in the full suite.

Screenshots: `study-qna-offline.png`, `study-qna-light-offline.png`, and
`study-quiz-offline.png`. Every displayed document is labelled synthetic.

Not established: live course-answer correctness, the Week 2 meme retrieval,
representative chart reasoning, actual endpoint configuration, performance
under simultaneous users, or the assignment's complete design comparison.
The synthetic model's factual correctness is not evidence for a real model.
