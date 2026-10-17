# Quiz integration contract (Issues #6, #7; PR #15 dependency)

This contract separates the offline quiz core and model/provider slice from
remaining live-model and UI work. Retrieval from PR #15 is now merged and its
real classes are exercised offline with synthetic PDFs and a fake model.
A separate opt-in suite exercises the real quiz model using synthetic PDFs.
Issue #6 owns the callable provider; Issue #7 owns the Gradio UI. Structural
validation and a small synthetic live sample do not prove general grounding.

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
Chat requests explicitly set `response_format={"type":"json_object"}`. The
class vision service supports this mode; prompt instructions alone produced
Markdown-fenced JSON in live tests. The parser still rejects fences, extra
fields, invalid options, and invented evidence IDs. Unsupported server modes
fail safely; there is no retry using weaker output validation.

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

Integration sketch: retrieval is merged; the UI still needs explicit wiring:

```python
from app.quiz_model import QuizVisionClient, VisionQuizGenerator
from app.quiz_provider import QuizProvider
# retrieval_service = build_retrieval_service(initialized_indexer)
provider = QuizProvider(retrieval_service, VisionQuizGenerator(QuizVisionClient()))
# build_ui(store, indexer, quiz_provider=provider)
```

A seed fixes answer placement only for identical generated questions/options.
Temperature zero is not a guarantee of identical remote generations. The
optional server-side replay wrapper below retains an actual generated quiz/key;
it does not make the remote model deterministic.

### Bounded server-side replay

Wrap the existing provider once per server process, not once per request:

```python
from app.quiz_replay import QuizReplayProvider
replay_provider = QuizReplayProvider(store, provider, max_entries=128)
# build_ui(store, indexer, quiz_provider=replay_provider)
```

This backend slice does not change production startup or Johana's UI branch.
Keep the wrapper and returned `Quiz` on the server. Create a new `QuizSession`
for each student/attempt; never cache a completed session, score, or browser
state. Sharing an immutable quiz must not share completion state.

Replay is limited to identical requests and unchanged selected materials.
Fingerprint checks cover selected library records, corpus bytes, and parsed
artifacts, including source images, before reuse and around generation.
Removed/unavailable materials fail closed rather than returning old answers.
Failed or invalid generations are not retained. The cache is bounded (default
128 quizzes) and process-local: eviction, explicit invalidation, and server
restart end replay availability. A later generation can differ. This is not
permanent reproducibility, disk persistence, or cross-worker coordination.

The wrapper serializes calls, including generation, to coalesce simultaneous
identical requests. This trades throughput for a small, auditable concurrency
boundary. Fingerprints do not establish factual support or replace the
retriever's evidence validation; before/after checks cannot make external
filesystem mutations atomic with returning a quiz. Retrieval/model/config
changes require a fresh wrapper or explicit cache invalidation. The server's
trusted document store remains the authority, not browser-supplied paths.

Offline checks:

```bash
python -m pytest tests/unit/test_quiz_model.py tests/unit/test_quiz_provider.py tests/smoke/test_quiz_provider_flow.py -q
```

These use injected fake responses and synthetic artifacts. They exercise image
encoding, selected-material scope, schema validation, error concealment, and
session scoring/reveal, **not** live endpoint correctness or a real Gradio flow.

### Opt-in live quiz verification

Set the real server-side `BASE_URL_VISION`, `CLASS_API_KEY`, and `VISION_MODEL`
in your local gitignored `.env` or process environment. Then run deliberately:

```bash
RUN_LIVE_QUIZ=1 python -m pytest tests/e2e/test_quiz_live.py -m e2e -q
```

The assignment requires a deliberate opt-in for shared-service usage. The file
is marked `e2e`, excluded from the default/CI selection, and skips without
`RUN_LIVE_QUIZ=1` or complete non-placeholder vision settings. It does not
print or persist credentials, headers, raw responses, or screenshots. Do not
use pytest's `--showlocals` with real credentials. It generates its own temporary
PDFs; no course files or committed fixture content are needed.

Four cases cover real PDF parsing/BM25 retrieval into the real provider/model
and quiz session: original-image generation, missing-image text fallback,
unsupported-topic rejection before a model call, and safe handling of one
intentionally invalid synthetic credential. The valid cases use same-name
PDFs with different identities, check the known correct duration, preserve
original source feedback, hide the key until submit/reveal, reject invalid
answers, and verify correct/incorrect scoring. Semantic retrieval, visual
indexing, reranking, browser events, and image-only discovery are **not**
exercised by this file.

On Windows, the new original-image case failed against unchanged merged main
`f2b7001`; with server JSON mode, the four cases passed. The request-payload
unit regression separately failed before the fix. Broader exploratory live
checks also found source-supported synthetic questions, but independent
regeneration changed wording/options and answer positions with the same seed.
A live pass does not certify all future generated answers or establish replay.

The exploratory run also exposed separate setup/visual-index limitations:
nonempty `.env` model names and the rerank key may still be template values;
compare model names with authenticated `/models` output without logging keys.
A rendered 396x512 JPEG sent as a plain data-URI `input` to the visual embedder
was rejected with HTTP 400; the same image in the endpoint's documented
multimodal `messages` input succeeded. That visual-client change, stable
regeneration/replay, and general factual-answer evaluation are outside this
JSON-mode fix. No local `.env` values or teammate branches are changed here.

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
not concurrent-session proof, and not a live endpoint test. Retrieval integration
is tested separately through real merged classes. Before claiming the full
student-facing flow, add combined browser verification (including visual-only
source, malformed evidence, image ownership and multi-user isolation), rerun
Windows offline tests, and verify any additional live service stages with
gitignored credentials. Issue state alone does not establish acceptance.
