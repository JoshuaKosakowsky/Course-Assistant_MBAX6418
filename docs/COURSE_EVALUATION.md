# Course-material acceptance and comparison (Issue #8)

The committed eight-question set in `course-evaluation.json` covers syllabus
rules, slide text, two visual questions and one deliberately unanswerable
question. Its expected source pages were checked against the local syllabus
and Week 2 PDF. The original meme (physical PDF page 33) and comparison
diagram (page 30) were inspected locally. Course PDFs and page images are
not committed. This is a prepared evaluation set, not completed results.

## Deliberate live run

After configuring the gitignored `.env`, import the named syllabus and Week 2
PDF into the library. The filenames in the specification must resolve to one
document each; duplicate display names fail closed. Adjust those filenames
for your authorized local copies if necessary. Keep the same material IDs,
questions and model settings for both approaches.

Use `python -m app.indexing --visual-backfill` deliberately to prepare missing
visual vectors when needed. The evaluation does not fabricate visual indexes
or silently treat skipped visual stages as verified visual performance.

Run:

```bash
python -m scripts.evaluate_course --live
```

Without `--live` the command exits before creating clients or making requests.
The live run deliberately uses the configured class vision, embedding and
reranking services. Startup reconciliation may repair missing text vectors.
The two approaches use the same indexes and vision model; reranking is off
for one and on for the other. Question order is fixed and approach order is
counterbalanced. Time includes retrieval, generation and any visual support
check; it excludes startup/indexing. One pass is not a latency benchmark.

Results default to gitignored `results/raw/course-evaluation.json`, because
answers and excerpts can contain course content. Service exceptions are
recorded as generic failures; no headers or credentials are serialized.
Do not publish raw results without reviewing course-content permissions.

## Human review

For every approach and question, inspect the answer and original source:

1. Set `correct` to a boolean after checking all factual claims.
2. Set `sources_support_answer` to a boolean after matching cited pages and
   excerpts to the original. A valid source ID alone does not establish support.
3. For visual questions, set `original_image_matches` after checking the
   actual image and the answer's interpretation, including diagram labels.
4. Explain failures or uncertainty in `review_notes`. Leave fields null when
   unreviewed; do not turn unreviewed answers into zeroes or successes.
5. For the missing-information question, require refusal without an invented
   code or citation. A grounded flag by itself is not a correctness score.

Summarize the reviewed local results without any network requests:

```bash
python -m scripts.evaluate_course --summarize results/raw/course-evaluation.json
```

Summary metrics use only fully reviewed question pairs for which the intended
rerank stage actually ran (on) or was skipped (off). The report discloses
unreviewed/invalid pairs and leaves scores null when none qualify. Record
failed/degraded runs as limitations, rather than claiming the approaches were
compared successfully. Discuss the missing-information case separately if
no reranking candidates exist.

## Submission record

In README, report all eight questions, reviewed answer accuracy, source
support, elapsed time and image correctness, including failures. Explain
which approach you would retain and why, with the quality/latency tradeoff.
Save a sanitized aggregate summary and your interpretation for team review.
Keep the unreviewed/raw distinction explicit. Existing synthetic BM25 and
reranking demonstrations do not replace this course-answer evaluation.

The runner's offline tests use injected services/models, exercise comparison
controls and failure handling, and make no class requests. They do not claim
that any real course question has been answered correctly by the app.
