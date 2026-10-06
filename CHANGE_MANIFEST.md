# Change Manifest

## Delivery identity

- Date: **2026-10-07**
- Repository: `m7mdxzx9/arena.ai`
- Branch: `arena/d8508d0d-arena-ai`
- Preserved base commit: `e26537851e5e4450807dacab014d79ea58095b4f`
- Strategy: additive in-place upgrade; no repository rewrite or progress reset

## Filesystem delta

Counted with Git (`git ls-tree` for the base and `git ls-files --cached --others --exclude-standard` for delivery):

| Measure | Before | After | Delta |
|---|---:|---:|---:|
| Non-ignored repository files | 85 | 137 | +52 |
| Text lines across those files | 14,430 | 20,706 | +6,276 |

The working tree contained 13 modified tracked files and 52 new non-ignored files before the delivery commit. Generated `.venv`, `node_modules`, `dist`, SQLite data and caches are ignored and excluded.

## Preserved systems

Profiles/settings, campaign areas, 104-concept tree, procedural questions, XP/ranks/equipment, achievements, BKT, Leitner review, missions, bosses, prediction lab, Data Lab, scikit-learn workbench/history, NumPy NN, Code Dojo, research challenges, NLP widgets, educational RAG and deterministic agent-security simulator remain connected.

## Added backend systems

- secure player-owned CSV/TSV/JSON/XLSX dataset store and uploaded-data ML path;
- deterministic top-five labelled-score utility;
- real optional PyTorch MLP and Conv2d engines, CPU/CUDA reporting, curves, generated checkpoints, augmentation, confusion/prediction/feature-map outputs;
- bounded provider protocol and local Ollama discovery/chat/structured output;
- offline/local tutor and experiment-derived Mistake Journal;
- PDF/TXT/Markdown/DOCX personal RAG with bounded extraction, hybrid retrieval, citations, injection flags and optional grounded generation;
- controlled real agent loop with fixed schema tools, permissions, timeouts, bounded visible memory and action traces;
- agent-vs-agent arena with identical tasks, deterministic phrase/completion scoring and measured step/latency comparison;
- persistent prompt versioning and central deterministic evaluation datasets;
- experiment-grounded portfolio and escaped Markdown/HTML/JSON exports;
- versioned, recursively redacted, atomic merge backup/restore;
- runtime capabilities endpoint and structured request logging/error responses;
- additive SQLite tables/settings compatible with existing records.

## Added frontend systems

- structured English/Arabic provider and resources, persisted language, document RTL and technical LTR isolation;
- localized shell/onboarding/campus/settings and all new workspace surfaces;
- Open Lab, Dataset Workspace, PyTorch/CNN, Model Hub, Tutor, Mistakes, Personal RAG, Real Agent/Arena, Prompt Lab, Evaluation Lab, Portfolio and Backup/Restore routes;
- responsive lab/RTL styling;
- safe unknown-error formatter and typed application errors;
- lazy-loaded new labs, reducing the primary production JS chunk from the warning threshold to about 393.90 kB before gzip.

## Engineering and deployment artifacts

- required architecture, learning, adaptive, ML, PyTorch, RAG, agent, security, localization and testing documentation;
- engineering fingerprint, signature, micro-benchmark and capability matrix;
- non-root multi-stage `Dockerfile`, `compose.yaml`, persistent data volume and healthcheck;
- focused backend integration/security/real-training tests and frontend API/localization tests.

## Final verification on delivered source

| Command/check | Result |
|---|---|
| `backend/.venv/bin/python -m pytest -q` | **117 passed**, 0 failed, 1 Starlette deprecation warning, 31.26 s |
| `frontend/npm run test:ui` | **4 files passed, 11 tests passed**, 0 failed, 23.90 s |
| `frontend/npm run lint` | **0 errors, 22 warnings** (same count as baseline; Fast Refresh/effect warnings in preserved files) |
| `frontend/npm run build` | passed, 63 modules transformed; primary JS 393.90 kB / 116.88 kB gzip plus lazy chunks |
| focused API/localization Vitest run | 2 files passed, 4 tests passed |
| backend `pip check` | no broken requirements |
| frontend `npm audit --audit-level=low` | 0 vulnerabilities |
| `git diff --check` | passed |
| app construction | 84 routes |
| live `GET /api/health` | `{"ok":true}` |
| live capability smoke | PyTorch 2.14.1+cu130 installed; CUDA false/0 devices; Ollama explicitly unavailable |
| micro-benchmark | 25,000 records × 25; 2.807 ms median, 3.651 ms p95 |

Docker syntax/artifacts were inspected, but a Docker daemon/CLI was unavailable in the workspace, so no image-build result is claimed.

## Disclosed limitations

- Some preserved legacy campaign/widget prose remains English. New surfaces, shell/settings and RTL infrastructure are bilingual; the matrix labels overall Arabic content coverage **partial**, not complete.
- Authentication and hostile multi-tenant deployment are absent by design; this remains a local single-user application.
- Ollama was not reachable in the verification environment. Provider-dependent paths are tested with deterministic fake-provider boundaries and fail explicitly at runtime.
- CUDA was unavailable; genuine PyTorch tests executed on CPU.
- Portable JSON backup excludes dataset/document binaries and checkpoints and records this in its manifest.
- LLM-as-judge is not presented; evaluation and arena scoring are deterministic and explicitly labelled.
