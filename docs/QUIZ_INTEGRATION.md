# Quiz integration contract (Issues #6, #7; PR #15 dependency)

This contract is **offline and provisional**. It does not import the unmerged
retrieval or UI branches, contact a class endpoint, or prove model grounding.
It belongs on Issue #6's `feat/quiz-engine` branch; Issue #7 owns the Gradio UI.

## Evidence from PR #15

Pass a vetted `EvidencePack`-like object to `evidence_from_pack(pack)` in
`app/quiz_integration.py`. It reads `pack.items`: each entry exposes
`doc_id`, `filename`, `page_no`, `excerpt`, `image_path`, and
`image_verified` (the `EvidenceItem` fields in PR #15). The result is a
tuple of `EvidenceRef`:
`document` (filename, display-only), `page`, `excerpt`, `image_path`,
`doc_id` (unique document identity).

- Carry text excerpts as supplied. Include an image path **only** when
  `image_verified is True`; never promote an unverified visual hit to a cited
  original. A verified image-only page may be used as evidence; a page with
  neither text nor verified image is omitted. Missing identity/page is omitted.
- `doc_id` is preserved from each item. Items without a valid `doc_id`
  convert to `None` and can never match a document selection (fails closed).
- This adapter does **not** validate file ownership, manifest membership, or
  whether a generated answer is factually supported. PR #15 must fix its
  pending manifest/provenance findings and supply validated evidence before
  production use. Synthetic unit tests validate only the shape and gates.
- `QuizRequest.materials` contains **unique document IDs** (`EvidenceRef.doc_id` /
  `EvidenceItem.doc_id`), never filenames. The document store permits
  different files with identical filenames (records are keyed by content
  hash), so filename scoping would leak evidence across selections.
  `EvidenceRef.document` keeps the filename only as display metadata for
  student-facing feedback. Issue #7's picker must pass document IDs directly;
- The current topic filter is a case-insensitive literal substring of the
  excerpt. This is not semantic retrieval. Image-only evidence is excluded
  when a topic is supplied, unless a future retriever supplies topic-matched
  text separately. Revisit this after #15 integration.
- `generate_quiz(request, evidence, generator)` validates the number of
  questions and exact evidence identity. The future generator must return
  `Question` records with four options, a correct index and explanation; the
  adapter does not itself establish factual support.

## Issue #7 UI event boundary

Keep `Quiz` and `QuizSession` **on the server**. Never place them, their repr,
answer keys, explanations, or feedback in Gradio components or browser state
before submission. `gr.State` is not a secrecy boundary for an answer key;
use a server-side session registry with per-user isolation, opaque session IDs,
expiration, and cleanup before enabling a multi-user UI.

1. **Generate:** Resolve the selected materials; retrieve/validate evidence;
   call `generate_quiz(QuizRequest(materials, topic, count, seed), evidence,
   generator)` and create a server-side `QuizSession`. Send only
   `session.student_view()` to the browser: `completed` and an ordered list
   of `{prompt, options}`. Reset prior answers/session when regenerating.
2. **Submit:** Receive one zero-based option index for every question; call
   `session.submit(indices)` server-side. Invalid count/type/index must not
   reveal answers. A completed session rejects second submission. Only then
   display score, correct choices, explanations and source feedback.
3. **Reveal solutions:** Offer a deliberate separate action calling
   `session.reveal()`. It returns solutions and evidence with `score=None`
   and locks subsequent submission; it must not trigger on page load.

`tests/smoke/test_quiz_ui_contract.py` exercises this sequence with plain
handler functions and synthetic evidence. It is **not** a Gradio/browser test,
not concurrent-session proof, and not a live endpoint test. Before merging
with #15 and #7, add a real combined integration test (including visual-only
source, malformed evidence, image ownership and multi-user isolation), rerun
Windows offline tests, and run opt-in live endpoint checks with gitignored
credentials. Issue #6 remains open until the full quiz flow is verified.
