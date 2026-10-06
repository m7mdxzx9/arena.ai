# Micro-benchmark

Measured locally on **2026-10-07** in the delivered workspace. These are engineering sanity measurements, not cross-machine performance claims.

## Python top-five labelled scores

Implementation: `backend/neural_forge/ranking.py::top_five_scores`

Contract:

- accepts `(label, score)` records;
- ignores empty labels and non-finite/unparseable scores;
- keeps the maximum score per exact label;
- sorts descending by score with deterministic case-insensitive/exact label tie-breaks;
- returns at most five records.

Test coverage: `backend/tests/test_workspace.py::test_top_five_scores_filters_groups_and_sorts`.

Command:

```bash
cd backend
PYTHONPATH=. .venv/bin/python scripts/micro_benchmark.py
```

Observed result with Python 3.11.2:

| Input | Iterations | Median | p95 |
|---:|---:|---:|---:|
| 25,000 records (997 repeating labels) | 25 | 2.807 ms | 3.651 ms |

The resulting first label was `label-0367` with score `115.99953820236937`. The script prints all five winners so a rerun can verify behavior as well as timing.

## TypeScript unknown-error formatter

Implementation: `frontend/src/api.ts::formatApiError`

The formatter safely handles `ApiError`, ordinary `Error`, strings, primitives, application `{error|message|detail}` objects, FastAPI validation arrays, `null`, malformed values and objects/Proxies whose getters throw. It truncates exposed messages and returns a neutral fallback instead of serializing arbitrary object internals.

Test coverage: `frontend/tests/api.test.ts` (2 tests, including a throwing `Proxy`). Focused Vitest execution passed together with localization tests; exact final suite results are recorded in `CHANGE_MANIFEST.md`.

The formatter is intentionally tested for safety/contract rather than advertised with a nanosecond throughput number: UI error handling is network- and render-bound, and a synthetic timing would not be decision-useful.
