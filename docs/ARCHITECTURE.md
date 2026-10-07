# NEURAL FORGE Architecture

## Purpose

NEURAL FORGE is a local-first **AI learning game and personal AI laboratory**. The upgrade keeps the original campaign, progression, adaptive learning, scikit-learn/NumPy labs, educational RAG and agent simulators intact while adding separate real-workspace capabilities.

## Runtime topology

```text
React 19 + TypeScript + Vite (hash-routed SPA)
                 │ relative /api requests
                 ▼
FastAPI application (`neural_forge.app`)
  ├─ learning/game services
  ├─ scikit-learn + NumPy experiment engines
  ├─ optional PyTorch engine
  ├─ guarded upload/document stores
  ├─ provider abstraction → optional local Ollama
  └─ SQLite persistence + UUID-owned files
```

FastAPI serves both `/api/*` and the built SPA. Browser code uses relative URLs, so the same deployment origin works locally, in Docker and behind a reverse proxy. The core campaign and deterministic labs do not need network access or a paid API.

## Major boundaries

| Boundary | Main modules | Responsibility |
|---|---|---|
| API | `app.py` | Pydantic request validation, route ownership, structured errors, static delivery |
| Persistence | `db.py` | Idempotent SQLite schema and compatibility-safe additions |
| Learning game | `game.py`, `mastery.py`, `missions.py`, `bosses.py`, `achievements.py` | XP/ranks, equipment, BKT, Leitner review, missions and bosses |
| Classical ML | `ml.py`, `datalab.py`, `datasets.py` | Real preprocessing, splitting, fitting, metrics and diagnostics |
| Educational neural engine | `nn.py` | Transparent NumPy forward/backward training |
| Professional neural engine | `torch_engine.py` | Optional real PyTorch MLP/CNN training and local checkpoints |
| Personal data | `user_datasets.py` | Validated CSV/TSV/JSON/XLSX ingestion and player-owned normalized storage |
| Local models | `llm.py` | Provider protocol and bounded Ollama HTTP implementation |
| Personal RAG | `document_rag.py` | Guarded extraction, chunking, hybrid retrieval, citations and grounded optional generation |
| Real agents | `real_agent.py` | Structured local-model loop, validated tools, permissions, timeouts, bounded visible memory and traces |
| Prompt/evaluation | `prompts.py`, `evaluation.py` | Prompt versions and centralized deterministic evaluators |
| Learning support | `tutor.py`, `mistakes.py` | Offline/local tutor and experiment-derived review journal |
| Portability/career | `backup.py`, `portfolio.py` | Atomic merge restore and experiment-grounded case studies |
| Localization | `frontend/src/i18n.tsx`, `frontend/src/locales/*` | Persisted locale, interpolation, RTL document state and LTR technical isolation |

## Data and ownership model

A player ID scopes all mutable data. SQL queries for user datasets, documents, prompts, agent configurations, evaluation sets, portfolios and runs include `player_id`. Uploaded filenames are never used as storage paths: UUID filenames live under configured roots. Dataset and document deletion resolves and verifies paths beneath those roots.

SQLite is intentionally used for a single-user/local deployment. The API currently has **no authentication or multi-tenant authorization boundary**; exposing it directly to untrusted networks is unsupported. See [SECURITY.md](SECURITY.md).

## Persistence and compatibility

`DB.__init__` runs additive, idempotent `CREATE TABLE IF NOT EXISTS` statements. Existing tables and payload formats remain intact. New player settings (`language`, `default_model`) are merged with older settings at read/update boundaries rather than requiring destructive migration. Existing run APIs remain compatible.

Large/private binaries are separated from SQLite:

- datasets: normalized CSV under `NEURAL_FORGE_UPLOAD_DIR`;
- documents: source files under `NEURAL_FORGE_DOCUMENT_DIR`, extracted chunks in SQLite;
- checkpoints: local `state_dict` plus JSON metadata beside the database.

Portable backups deliberately exclude these binaries and declare exclusions in their manifest.

## Frontend structure

`App.tsx` retains the existing shell and hash router. New laboratory pages use the same cards, controls, charts and game context. `OpenLab.tsx` is the non-linear entry point. `I18nProvider` owns language state and updates `<html lang dir>`. `rtl-and-labs.css` supplies logical-direction layout and forces code, model identifiers, equations, charts and JSON back to LTR.

## Failure model

Domain errors become structured JSON with an HTTP status and code. Provider unavailability remains explicit; there is no fabricated model response. Uploads fail closed on format, size, shape or extraction limits. Training returns configuration, seed, device, warnings and measured outputs. Optional capability detection is available at `GET /api/system/capabilities`.

## Deployment

- Development: `./run.sh`
- Container: `docker compose up --build`
- Persistent container data: `/data`
- Optional host Ollama: `OLLAMA_BASE_URL`, defaulting in Compose to `host.docker.internal:11434`

The Docker build supports `--build-arg INSTALL_TORCH=0` for a smaller image; in that image the PyTorch capability and lab correctly report unavailable.
