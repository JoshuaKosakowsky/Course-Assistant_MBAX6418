# Quiz integration contract (Issues #6, #7; PR #15 dependency)

This contract separates the offline quiz core and model/provider slice from
remaining live-model and UI work. Retrieval from PR #15 is now merged and its
real classes are exercised offline with synthetic PDFs and a fake model.
Issue #6 owns the callable provider; Issue #7 owns the Gradio UI. These tests
do not prove live grounding.

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
  whether a generated answer is factually supported. The production retriever
  must supply manifest-validated evidence before use. Synthetic unit tests
  validate the adapter's shape and gates. A separate synthetic-PDF smoke test
  exercises the merged retrieval implementation; neither proves live grounding.
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

## Callable provider and vision-model adapter

`QuizProvider(retriever, generator)` in `app/quiz_provider.py` implements the
callable expected by Issue #7: `provider(QuizRequest) -> Quiz`. Construction
makes no network calls. A request:

1. Rejects invalid configuration (one to five questions, document IDs, topic).
2. Calls `retriever.retrieve(query, doc_ids=request.materials, top_k=8,
   include_images=True)`; the outcome must expose `.evidence` as above.
3. Rechecks selected document IDs and the existing literal topic filter.
4. From that filtered, ranked pool, keeps the first four image-bearing
   references and **all** text-only references, preserving their original
   relative order. Excess image-bearing references are omitted entirely,
   even if they also have text; retained references keep their original image
   paths and provenance. The model adapter's four-image limit is unchanged.
5. Passes this identical subset to `generate_quiz` for both model input and
   exact evidence-identity validation, so a question cannot cite an omitted
   reference. Count validation precedes returning a server-held `Quiz`.

With no topic, retrieval uses the generic query `key concepts definitions
methods examples`. The retrieval pool remains `top_k=8`; the four-image
budget may further reduce the evidence available for generation. Selection
uses retrieval rank deterministically, not per-document quotas. This is a
bounded sample, **not** comprehensive coverage of all selected documents,
and does not guarantee every selected document contributes a question.
Empty/unsupported evidence stops before model usage.

`VisionQuizGenerator(QuizVisionClient())` in `app/quiz_model.py` implements
question generation. The client reads `BASE_URL_VISION`, `CLASS_API_KEY`, and
`VISION_MODEL` from existing server-side config. The URL must be HTTP(S) with
no embedded credentials, query, or fragment; reserved port 9000 is refused.
No keys, source paths, or raw server/transport errors belong in browser output.

The model receives evidence records labeled `E1`, `E2`, etc., and local image
content when present. Evidence is untrusted input, not instructions. It must
return only JSON with a `questions` list, each item containing:

```json
{"prompt":"...","options":["...","...","...","..."],
 "correct_index":0,"explanation":"...","evidence_id":"E1"}
```

Source document/page/excerpt/image fields come from the server's original
reference, never model-provided paths. Invalid output is rejected; there is
no fabricated-question fallback. A structurally valid citation **does not
prove** the answer or explanation is factually supported. Adversarial model
output and factual correctness still require separate evaluation.

The image boundary trusts the production retriever to supply validated local
artifact paths. Do not expose these APIs to arbitrary client-created packs
or paths. Images are encoded in memory for the model; original paths remain
unchanged for feedback. The UI must validate/serve original evidence safely.

Integration sketch **after** retrieval and UI prerequisites land (these imports
are deliberately not made by this branch):

```python
from app.quiz_model import QuizVisionClient, VisionQuizGenerator
from app.quiz_provider import QuizProvider
# retrieval_service = build_retrieval_service(initialized_indexer)
provider = QuizProvider(retrieval_service, VisionQuizGenerator(QuizVisionClient()))
# build_ui(store, indexer, quiz_provider=provider)
```

A seed fixes answer placement only for identical generated questions/options.
Temperature zero is not a guarantee of identical remote generations. Retain the
actual generated quiz/key server-side; cross-generation reproducibility needs
explicit caching/replay or a separately verified service guarantee. This slice
does not claim that acceptance criterion has been fully established.

Offline checks:

```bash
python -m pytest tests/unit/test_quiz_model.py tests/unit/test_quiz_provider.py tests/smoke/test_quiz_provider_flow.py -q
```

These use injected fake responses and synthetic artifacts. They exercise image
encoding, selected-material scope, schema validation, error concealment, and
session scoring/reveal, **not** live endpoint correctness or a real Gradio flow.

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
