# Testing

## Commands

### Backend

```bash
cd backend
python -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-torch.txt
.venv/bin/python -m pytest -q
```

### Frontend

```bash
cd frontend
npm ci
npm run lint
npm run build
npm run test:ui
```

`test:ui` starts a real backend on port 8765 and runs Vitest route/API smoke tests against it. Focused unit tests can run without a server:

```bash
npx vitest run tests/api.test.ts tests/localization.test.tsx
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
| Local models | explicit unreachable behavior and model discovery contracts |
| Personal RAG | extraction/query, citations, ownership and document prompt-injection flags |
| Agent | validated calculator, allow/deny permissions and observable fake-provider trace |
| Prompt/evaluation | versions, variable errors, deterministic evaluator persistence and unsafe regex rejection |
| Backup/portfolio | provenance, escaped HTML, secret redaction, merge restore and atomic rollback |
| Localization | resource parity, persistence, DOM RTL and technical LTR |
| API formatter | application errors, validation details, primitives and throwing Proxy objects |
| Routing | every preserved and newly added SPA route rendered against a live API |

## Real versus test doubles

PyTorch tests execute real CPU tensor operations. Classical ML tests execute real scikit-learn. Upload and RAG tests use real parsers on small in-memory fixtures. Agent-loop unit tests use a deterministic fake **provider only** so permission/tool behavior can be asserted; the tool execution and persistence are real. No test claims an Ollama model was available.

## Latest verified results

See the delivery report or `CHANGE_MANIFEST.md` for the final run performed on the delivered tree. Do not infer that a historical count covers later edits; rerun the commands after modifying source.

## Warning policy

The known test warning is Starlette’s compatibility deprecation for importing `TestClient` through the installed FastAPI stack. It does not indicate a failed behavior test. Lint warnings are reported separately from errors and must not be described as a clean lint run unless they reach zero.
