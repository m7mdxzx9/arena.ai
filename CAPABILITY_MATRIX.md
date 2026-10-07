# Capability Matrix

Status rules are deliberately strict:

- **Complete** — usable UI and logic, persistence/validation where applicable, and meaningful tests.
- **Partial** — useful implementation exists, but at least one important requested dimension is absent.
- **Optional** — behavior is real when a declared local dependency is installed; an unavailable dependency is reported, never simulated.
- **Simulator** — intentionally educational and explicitly labelled.
- **Absent** — not implemented.

This matrix describes the repository at this revision. It is not a roadmap and does not count a backend-only endpoint as a complete product feature.

| Capability | Status | Evidence / limitation |
|---|---|---|
| Profiles and settings | Complete | SQLite player records, validated settings, profile UI |
| XP, ranks, achievements, equipment | Complete (preserved) | Existing game services and UI remain connected |
| Missions and boss battles | Complete mechanics; localization partial | Durable progression and real lab-backed actions; portions of legacy narrative remain English |
| BKT mastery | Complete (preserved) | Per-attempt probability updates and tests |
| Leitner review | Complete (preserved) | Due scheduling and review UI |
| Experiment history/comparison | Partial | Saved runs, notes and comparisons work; lifecycle management and richer comparison workflows remain limited |
| Classical scikit-learn ML | Complete for current task catalogue | Real fitting, splits, diagnostics and player-owned uploaded data; bounded tabular tasks only |
| NumPy neural network | Complete, educational | Real transparent forward/backprop, explicitly separated from PyTorch |
| Code Dojo | Complete (preserved) | Bounded exercise evaluator; agents cannot access it as a shell |
| Educational RAG lab | Simulator (preserved) | Deterministic bundled-corpus learning surface, explicitly labelled |
| Educational agent security lab | Simulator (preserved) | Deterministic state machine, explicitly labelled |
| Structured English/Arabic resources | **Partial** | The shell and newer personal-lab pages use structured locale files; significant legacy campaign, workbench and widget prose still remains English |
| Persisted language and document RTL | Complete | Player/browser persistence and `<html lang dir>` updates are tested |
| RTL layout and technical LTR isolation | Substantially complete | Global RTL layout/font rules and LTR code/chart/JSON isolation; untranslated legacy surfaces still need review |
| Secure personal dataset workspace | Complete for supported formats | CSV/TSV/JSON/XLSX limits, normalized server storage, ownership, inspect/delete/train UI and tests |
| Open Lab launch surface | Complete | Routed non-linear launcher for implemented labs |
| Real PyTorch MLP | Optional, partial scope | Real CPU/CUDA training, curves and generated checkpoints; built-ins plus uploaded mixed-type classification, but not general regression/multilabel/time-series training |
| Real CNN | Optional, partial scope | Real Conv2d digit training, curves, confusion, predictions, feature maps and bounded augmentation; not a general image-dataset workspace |
| CUDA reporting | Complete | Runtime/device reporting and explicit CPU fallback; CUDA execution cannot be verified on CPU-only CI |
| Generated-checkpoint manager | Complete | Ownership, hash/size metadata, rename/download/delete UI and tests; arbitrary checkpoint upload is intentionally unsupported to avoid unsafe deserialization |
| Additive database migrations | Complete for schema v2 | Migration ledger plus `PRAGMA user_version`; existing databases are upgraded additively |
| Provider abstraction | Complete for Ollama | Typed bounded protocol and one concrete local provider; no second provider implementation yet |
| Model Hub / Ollama lifecycle | Optional, substantially complete | Health, metadata, pull and confirmed delete UI; pull is synchronous and has no resumable/background progress stream |
| Offline tutor | Complete for curated catalogue | Local curated answers with player/run context; not a generative model |
| Local-LLM tutor | Optional | Real selected Ollama model with bounded context and explicit provider errors |
| Mistake Journal | Complete for current sources | Persistent experiment-derived records, filtering and spaced review |
| Personal document RAG | Complete for supported documents | Bounded PDF/TXT/MD/DOCX extraction, normalized text storage, hybrid retrieval, citations and extractive answer mode |
| Grounded local RAG generation | Optional | Real Ollama generation with cited context and prompt-injection warnings; no fabricated fallback |
| Controlled real Agent Lab | Optional, substantially complete | Fixed schemas/tools, explicit permissions, timeouts, visible memory, traces and configuration CRUD; no shell tool |
| Agent-vs-agent arena | Optional, partial evaluation depth | Same bounded tasks across 2–4 configurations with deterministic scoring and saved summaries; no human rubric or calibrated judge |
| Prompt Lab | Optional, partial | Persistent prompts/immutable versions, variables, CRUD and real Ollama execution; richer side-by-side A/B workflow is still limited |
| Central Evaluation Lab | Complete for deterministic evaluators | Persistent dataset CRUD, nine deterministic evaluator types and saved result runs |
| LLM-as-judge | Absent by design | Not represented as implemented; deterministic evaluation is used instead |
| Portfolio | Complete for case studies | Run-grounded editable CRUD plus Markdown/HTML/JSON export |
| Compact JSON backup | Complete for declared records | Versioned, redacted, validated transactional profile merge; exclusions are explicit |
| Full archive backup | Complete for guarded local assets | `.nfbackup` UI/API includes normalized datasets/documents and opaque generated checkpoints; validates paths/counts/sizes/formats/SHA-256 and never deserializes checkpoints. Agent execution traces remain intentionally excluded |
| CV expansion | **Partial** | Educational convolution visual plus digit CNN; no owned image-folder ingestion, transfer learning or detection/segmentation workflow |
| NLP expansion | **Partial** | Tokenization, embeddings, attention, n-gram generation and tabular sentiment examples; no general transformer fine-tuning/evaluation workspace |
| Docker packaging | Implemented, unverified here | Non-root multi-stage image, persistent volume, healthcheck and Compose exist; this environment has no Docker executable, so no image-build claim is made |
| Authentication / network multi-tenancy | Absent | Local profile separation is not authentication; the app must not be exposed as a hostile multi-user service |
| Documentation | Partial | Architecture/security/testing/localization documents exist, but capability and legacy-content documentation still need continuing reconciliation |

## Audit conclusion

The repository is a substantial working learning game with a growing personal laboratory. It is not yet accurate to call the entire requested product complete. In particular, full legacy Arabic localization, broader CV/NLP workflows, deeper experiment/prompt evaluation workflows, and Docker/CUDA/Ollama runtime verification remain open. Educational simulators are retained and labelled separately from real PyTorch, retrieval and local-model execution.
