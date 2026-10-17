# Phase 6/7 — Chunk Size Comparison

## Objective

Compare three chunk sizes for BM25 keyword retrieval while holding the chunk overlap constant.

The goal is to examine retrieval effectiveness and index size using a reproducible, offline experiment.

## Experimental Setup

- Retrieval method: BM25 keyword search
- Chunk sizes: 250, 500, and 1,000 characters
- Chunk overlap: 64 characters
- Retrieval depth: top 3 results
- Dataset: Four synthetic course-document pages
- Evaluation queries: Four, each with a known relevant page
- External API calls: None
- Randomness: None

All configurations used identical source content and queries.

## Results

| Chunk size | Chunks | Accuracy@1 | Hit Rate@3 | MRR@3 |
|---|---:|---:|---:|---:|
| 250 | 25 | 100% | 100% | 1.000 |
| 500 | 13 | 100% | 100% | 1.000 |
| 1,000 | 7 | 100% | 100% | 1.000 |

Detailed results are available in:

- `results/chunk-size-comparison.csv`
- `results/chunk-size-details.json`

## Findings

All three configurations retrieved the expected source page as the first result for every evaluation query.

Increasing chunk size reduced the number of indexed chunks:

- 250 characters: 25 chunks
- 500 characters: 13 chunks
- 1,000 characters: 7 chunks

The 1,000-character configuration produced 72% fewer chunks than the 250-character configuration.

For this dataset, the larger configuration reduced index size without decreasing the measured retrieval effectiveness.

## Interpretation

The experiment demonstrates a tradeoff between chunk granularity and the number of indexed records.

Larger chunks may reduce index-management overhead, while smaller chunks may provide more focused evidence passages.

The results do not establish a universally optimal chunk size because the evaluation dataset is small and synthetic.

No measurable retrieval-quality difference was observed across the three configurations.

## Automated Testing

The new test module covers:

- Metric validity
- Deterministic repeatability
- Invalid chunk settings
- Empty indexes
- Empty queries
- Document filtering
- Document removal

Validation results:

- New comparison tests: 14 passed
- Full test suite: 237 passed, 6 skipped
- Warnings: 5 deprecation warnings

## Limitations

This experiment evaluates BM25 keyword retrieval only.

It does not evaluate semantic embeddings, multimodal reranking, generated answers, or real student questions.

The synthetic pages contain distinct vocabulary, making retrieval relatively straightforward.

Indexing latency, memory consumption, and retrieval latency were not measured.

The six skipped tests were not investigated as part of this experiment.

## Reproduction

Run the comparison:

    python -m scripts.compare_chunk_sizes

Run the complete test suite:

    python -m pytest -o addopts="" -q

Both commands run locally without requiring external API credentials.

## Conclusion

The 1,000-character configuration generated the fewest chunks while maintaining identical retrieval scores on the synthetic evaluation.

Further evaluation using more diverse questions and real course material would be necessary before recommending a production default.
