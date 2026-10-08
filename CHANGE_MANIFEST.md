# Change Manifest

**Status date:** 2026-10-08 (Asia/Riyadh)
**Repository:** `m7mdxzx9/arena.ai`
**Working branch:** `arena/667d59d2-arena-ai`
**Branch base:** `75bf761ef5a318259b674086ca1a20fee7d45482`
**Upgrade strategy:** additive changes in the existing repository; the campaign, stored player progression, APIs and existing educational labs were preserved.

## Scope delivered in this workspace

- Added **Personalized Tutoring** under Modern AI, after Structured Output and before RAG. Its curriculum explanation and three questions cover actual saved history, mastery estimates, and progressive hints.
- Expanded the bilingual experience with reviewed Arabic learning material for 21 of 105 concepts, bilingual RAG/Tutor learning missions and retrieval-boss phases, Arabic RAG simulator controls/diagnoses/benchmark questions and gold answers, structured API error localization, and explicit RTL/LTR handling. Overall legacy Arabic coverage remains partial.
- Kept the **RAG Learning Simulator** and **Advanced Personal RAG Lab** distinct. The simulator uses a fixed fictional English corpus and transparently labelled TF-IDF/LSA, 32-feature hashing, BM25 and lexical-overlap heuristic reranking. The Advanced Lab works on owned player documents and supports inspectable local retrieval, citations, saved experiments, and optional local neural providers.
- Added cached local Sentence Transformers/Ollama embedding providers and an optional local cross-encoder path for Advanced RAG; statistical LSA and lexical retrieval remain clearly labelled fallbacks. No automatic model download or hosted document processing is used.
- Personalized the offline/optional-local Tutor using the player's real mastery estimates, run metrics/configuration, mistake patterns and due-review state. No-answer/progressive-hint behavior is preserved; feedback does not change BKT; raw question/answer transcripts are not persisted.
- Added evidence-checked side missions for comparing retrieval, citation grounding and contextual Tutor hints. Completion checks persisted experiments/interactions, rejects cross-player experiments and awards XP idempotently.
- Extended the Retrieval Warden without replacing the training phases: its final phase requires a saved personal RAG experiment with a player-owned document, retrieved evidence, a citation matching a retrieved chunk and a documented strategy.
- Updated documentation and capability metadata to state current functionality and remaining limitations. Added schema v5 backfills for pre-existing personal-document indexes and preserves older Tutor interaction rows; backend regression tests exercise both migrations and the new flows, with localized API-error unit tests.

## Preserved systems

Existing profiles, campaign areas and the original 104-concept progression remain in place; the tree now has 105 concepts after the additive Personalized Tutoring entry. XP/ranks/equipment, achievements, BKT, spaced review, missions, boss fights, datasets, ML workbench, NumPy/PyTorch labs, existing APIs and saved progression remain in place. The new evidence-based missions and boss phase are additive.

## Verification on the current source tree

| Check | Result |
|---|---|
| Backend: `PYTHONPATH=. .venv/bin/python -m pytest -q` | **138 passed, 4 skipped**, one Starlette `TestClient` deprecation warning |
| Frontend: `npm run test:ui` against a temporary local API | **4 files passed, 13 tests passed**, including route/widget smoke and gameplay flows |
| Localization/API unit assertions in that run | **6 passed** across `localization.test.tsx` and `api.test.ts` |
| Frontend lint: `npm run lint` | **0 errors, 22 warnings**; not warning-free |
| Frontend build: `npm run build` | TypeScript passed; Vite built successfully, 65 modules transformed |
| Backend dependency consistency: `.venv/bin/pip check` | No broken requirements |
| `git diff --check` | Passed |
| Live Ollama / cached model / CUDA / Docker image verification | Not available or not run; no live-provider, CUDA or Docker-build success is claimed |

The UI tests used a temporary backend database and asset directories, not the user's normal profile store. Run the commands in `docs/TESTING.md` after any further source edits.

## Deliberately disclosed limitations

- Arabic learning content is broader than menu localization but incomplete: 21 of 105 concepts have reviewed Arabic teaching/questions, the simulator's fixed source passages and free-form bigram outputs stay English, and legacy campaign/widgets still contain English text.
- The simulator is educational and statistically implemented; it is not neural RAG. Advanced Lab neural embeddings/cross-encoder and local-LLM answering require a suitable local dependency/model/Ollama installation.
- The local environment did not verify live Ollama, CUDA or Docker image execution. Local-only provider behavior has deterministic tests, not a live model claim.
- Authentication and hostile multi-tenant deployment are absent; this remains a local single-user learning application.
- Scanned PDFs have no OCR and complex document layout is flattened.

## Project archive

`neural-forge-phase1-complete.zip` contains the 149 non-ignored project files from this working tree, including modified and newly added source/tests/docs. `.git`, Python virtual environments, `node_modules`, compiled `dist` output and local databases are excluded. The archive is **519,098 bytes**. ZIP CRC/integrity and required-file membership checks passed for the packaged tree.

See `CAPABILITY_MATRIX.md`, `docs/RAG.md`, `docs/TUTOR.md`, `docs/LOCALIZATION.md` and `docs/TESTING.md` for details.
