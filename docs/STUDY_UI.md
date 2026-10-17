# Q&A and quiz integration (Issue #7)

Launch with `python -m app.ui`. Production startup reconciles the library,
then constructs one retrieval service shared by Ask Questions and Practice
Quiz. Creating clients does not send model requests. Index reconciliation
may repair missing text vectors, as in the existing indexing workflow.

Set server-side `BASE_URL_VISION`, `VISION_MODEL`, and `CLASS_API_KEY` in
the gitignored `.env`. Existing text, visual, and reranker configuration is
used by the retrieval service. Missing or template vision settings disable
generation with a safe message. No test model is used in production.

In Ask Questions, select materials, enter a question, and choose Ask
question. The UI shows the checked answer, document/page excerpts, and
manifest-verified original page images. An empty retrieval result does not
call the model. Retrieval stages that were skipped or degraded are shown
without exposing raw service errors. Images in the gallery can be opened
to inspect the original source page.

Image-only evidence preserves `excerpt=None`; no placeholder excerpt is
treated as extracted text. Such citations must match an original image in
the supplied evidence and pass an additional vision-model support check.
Malformed verdicts and service failures fail closed. This model judgement
is not proof of factual correctness: evaluate answers against original
course slides, especially diagrams, charts and the Week 2 meme example.

In Practice Quiz, select materials, optionally enter a literal topic
substring, and generate one to five questions. Answer keys remain on the
server until Submit Answers or Reveal Answers. Feedback includes source
excerpts and verified original images. The production provider uses the
bounded instance-local replay cache, so identical requests against unchanged
materials reuse the quiz; restarting or changing materials invalidates that
guarantee. A seed alone does not make remote model output deterministic.

Offline checks use real parsing, indexing, retrieval, provider and UI
controllers with injected model responses. They cover document scoping,
missing evidence/services, escaped output, source images, image-only support,
quiz generation/replay and answer concealment. Browser screenshots use
synthetic materials and explicitly injected models; they are interface
verification, not live course-answer evaluation.

Run `pytest` for unit and smoke tests. Deliberate live checks remain opt-in;
this change does not establish the required course-material evaluation,
class parser integration, or fresh-copy setup acceptance. See Issue #8 and
the assignment's evaluation requirements before declaring submission-ready.
