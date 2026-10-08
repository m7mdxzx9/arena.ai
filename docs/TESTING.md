# Testing

## Commands

### Backend

```bash
cd backend
python -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-torch.txt
PYTHONPATH=. .venv/bin/python -m pytest -q
```

### Frontend

```bash
cd frontend
npm ci
npm run lint
npm run build
```

`npm run test:ui` requires a running API at `http://127.0.0.1:8000` by default; it does not start the backend itself. Override with `NF_API=http://host:port`. For an isolated local run, start the API in another shell with temporary DB/document/checkpoint directories, then run:

```bash
npm run test:ui
npx vitest run tests/api.test.ts tests/localization.test.tsx  # unit tests only; no API server
```

### Docker smoke

```bash
docker compose build
docker compose up -d
curl --fail http://localhost:8000/api/health
curl --fail http://localhost:8000/api/system/capabilities
```

## Coverage by subsystem

| Subsystem | Meaningful checks |
|---|---|
| Existing campaign | players/meta, progression, mastery, missions, bosses, labs and API contracts |
| Dataset uploads | formats, limits, traversal-resistant storage, ownership and uploaded-data training |
| PyTorch/CNN | real CPU optimizer/backprop, curves, checkpoint files, convolution output, augmentation |
| Local models | explicit unavailable behavior and model-discovery contracts |
| Advanced Personal RAG | extraction, page/chunk metadata, ownership, cached provider vectors, BM25/hybrid ranking, honest statistical fallback, heuristic labels, citations and injection boundaries |
| Personalized Tutor | actual mastery/run/mistake/review context, bounded no-answer/progressive hints, Arabic curated fallback, feedback not changing BKT, raw prompt non-persistence |
| Learning missions | same-question/source/different-method evidence, citation membership, foreign-player rejection, idempotent completion and same-concept contextual hint ladder |
| Retrieval boss | existing educational phases preserved and final personal-RAG phase requires owned-document/retrieved-citation evidence |
| Agent | validated calculator, allow/deny permissions and observable fake-provider trace |
| Prompt/evaluation | versions, variable errors, deterministic evaluator persistence and unsafe regex rejection |
| Backup/portfolio | provenance, escaped HTML, secret redaction, merge restore and atomic rollback |
| Localization/capability metadata | resource parity, persistence, DOM RTL and technical LTR; API-error localization; runtime coverage count/partial flag and honest statistical/neural RAG labels |
| API formatter | application errors, validation details, primitives and throwing Proxy objects |
| Routing | every preserved and newly added SPA route rendered against a live API |

## Real versus test doubles

PyTorch tests execute real CPU tensor operations. Classical ML tests execute real scikit-learn. Upload and RAG tests use real parsers and algorithms on small in-memory/local fixtures. Agent and optional model-boundary tests use a deterministic fake **provider only** so permission, provenance and error handling can be asserted; tool execution, retrieval, persistence and validation remain real. No test claims Ollama or CUDA was available.

## Latest verified results on the current working tree

- Backend: `PYTHONPATH=. .venv/bin/python -m pytest -q` → **138 passed, 4 skipped**, one Starlette `TestClient` deprecation warning.
- Frontend UI: `npm run test:ui` against the temporary local API → **4 test files passed, 13 tests passed** (routes/widgets and gameplay flows included).
- Localization/API focused tests are included in that UI run: **6 tests passed** across those two files.
- Lint: `npm run lint` → **0 errors, 22 warnings**. Warnings are reported, not described as a warning-free lint run.
- Build: `npm run build` → TypeScript passed; Vite built successfully, 65 modules transformed.

Re-run these commands after further source edits. A historical result never automatically certifies later changes.

## Warning policy

The backend warning is Starlette's compatibility deprecation for importing `TestClient` through the installed FastAPI stack; it does not indicate a failed behavior test. Lint warnings are reported separately from errors and must not be described as a clean lint run unless they reach zero.
