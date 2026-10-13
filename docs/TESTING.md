# Testing — Course Assistant (MBAX 6418-801)

Three tiers, all pytest. Markers are defined in pyproject.toml; the default
run is unit + smoke (fast and offline). E2E is opt-in because it hits the
shared class endpoints.

| Tier   | Marker        | Offline? | How to run            | What it covers |
|--------|---------------|----------|-----------------------|----------------|
| Unit   | `unit`        | Yes      | `pytest -m unit`      | Pure logic: chunking preserves source metadata, fusion math, `{answer, sources}` schema validation, source-support checks, quiz answer-key stability (same seed → same key), solution-hiding rules, `.env` parsing, repo hygiene (no secrets in tracked files) |
| Smoke  | `smoke`       | Yes      | `pytest -m smoke`     | Cross-component wiring on tiny committed fixtures: ingest → index → retrieve without class endpoints (stubbed/synthetic embeddings), CLI entrypoints run, Gradio app builds and launches headless |
| E2E    | `e2e`         | No       | `pytest -m e2e`       | Real class endpoints (needs `.env` keys + network): question → verified `{answer, sources}` with real evidence image, full quiz flow, screenshots saved to `results/` |

## Commands

```bash
pytest                 # unit + smoke (default)
pytest -m unit         # unit only
pytest -m smoke        # smoke only
pytest -m e2e          # real endpoints — deliberate, needs .env
```

E2E tests must auto-skip when `.env` is missing or incomplete (pytest.skip),
so a teammate without keys can still run the default suite.

## Rules

- **New logic ships with its unit test**, in the same PR — including the
  negative case (`test.*` for the thing that must NOT happen).
- **Wiring changes get a smoke test** on fixtures, not on real data.
- **Anything that calls a class endpoint is an e2e test**, marked `e2e`,
  run locally with keys, and documented. Never let unit/smoke tests hit the
  network.
- **Fixtures** live in `tests/fixtures/` — tiny and committed (one page-
  sized PDF, one slide image). Real course files stay in `data/`
  (gitignored) and are never required for the default suite.
- **Outputs**: screenshots, comparison results, and logs go in `results/`
  (committed for team review) — except anything containing keys or
  sensitive course content.
- **Design comparison** (assignment requirement): script it, save
  before/after metrics and screenshots under `results/`, and state the
  conclusion in the PR and README.
- **Clean runs**: the default suite must pass from a fresh clone after
  `pip install -r requirements.txt`, no keys needed.

## CI

`.github/workflows/ci.yml` runs `pytest -m "unit or smoke"` plus a secret
scan on every push/PR. E2E stays out of CI (keys are never stored there);
a teammate with keys runs it locally and saves evidence to `results/`.
