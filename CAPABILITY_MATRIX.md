# Capability Matrix

Statuses mean: **complete** = usable UI + real logic + persistence/validation where applicable + meaningful tests; **partial** = useful implementation exists but one or more requested dimensions remain; **optional** = complete when a declared local dependency is present; **simulator** = intentionally educational and labelled; **planned** = not implemented.

| Capability | Status | Evidence / limitation |
|---|---|---|
| Profiles and settings | Complete | SQLite player records, mode/settings updates, profile UI |
| XP, ranks, achievements, equipment | Complete (preserved) | Existing game services and UI remain connected |
| Missions and boss battles | Complete (preserved) | Durable progress and real lab-backed challenges |
| BKT mastery | Complete (preserved) | Per-attempt probability state and tests |
| Leitner review | Complete (preserved) | Due scheduling and profile review UI |
| Experiment history/comparison | Complete (preserved) | Reproducible run records, notes and comparisons |
| Classical scikit-learn ML | Complete (preserved/extended) | Real fitting, metrics, splits, diagnostics, uploaded datasets |
| NumPy neural network | Complete, educational | Real transparent forward/backprop; labelled separately |
| Code Dojo | Complete (preserved) | Sandboxed exercise evaluator; no agent access |
| Educational RAG lab | Simulator (preserved) | Deterministic bundled-corpus learning surface |
| Educational agent security lab | Simulator (preserved) | Deterministic state machine, clearly labelled |
| Structured English/Arabic resources | Partial | Shell/settings/new labs translated; some legacy campaign/widget prose remains English |
| Persisted language and document RTL | Complete | Player/browser persistence, `<html lang dir>`, unit tests |
| RTL layout and technical LTR isolation | Complete | Global RTL stylesheet, Arabic font stack, LTR code/charts/JSON |
| Secure personal dataset workspace | Complete | CSV/TSV/JSON/XLSX, strict limits, UUID ownership, profile/delete/train UI |
| Open Lab | Complete | Non-linear routed launch surface |
| Real PyTorch MLP | Optional complete | Real training/curves/device/checkpoints; needs PyTorch install |
| Real CNN | Optional complete | Conv2d training, curves, confusion, predictions, feature maps, augmentation |
| CUDA reporting | Complete | Runtime capability/device count and explicit CPU fallback warning |
| Local checkpoint management | Complete for generated checkpoints | Local state_dict + metadata; arbitrary checkpoint upload intentionally unsupported |
| Provider abstraction | Complete | Protocol and bounded Ollama implementation |
| Model Hub/Ollama discovery | Optional complete | Real health/model metadata; explicit offline state |
| Offline tutor | Complete | Curated adaptive context, always local |
| Local-LLM tutor | Optional complete | Ollama generation with selected model and bounded context |
| Mistake Journal | Complete | Experiment-derived records, persistence, filters and spaced review |
| Personal document RAG | Complete | PDF/TXT/MD/DOCX, bounded extraction, hybrid retrieval, citations, extractive mode |
| Grounded local RAG generation | Optional complete | Ollama path, injection labelling and explicit failures |
| Controlled real Agent Lab | Optional complete | Structured model decisions, fixed tools, permissions, timeouts, memory and trace |
| Agent-vs-agent arena | Optional complete | Same tasks across 2–4 configs; deterministic phrase/completion scoring, step/latency metrics and saved run |
| Prompt Lab | Optional complete | Persistent versions/variables/comparison and real Ollama execution |
| Central Evaluation Lab | Complete | Persistent sets, nine deterministic evaluator types and result runs |
| LLM-as-judge | Planned by design | Not added without calibration; deterministic evaluators are labelled honestly |
| Portfolio | Complete | Run-grounded editable case studies and Markdown/HTML/JSON export |
| Backup/restore | Complete for portable records | Versioned, redacted, atomic merge; binary datasets/documents/checkpoints excluded by manifest |
| CV expansion | Complete for bounded scope | CNN, training augmentation, feature maps, confusion and mistakes |
| NLP expansion | Complete (preserved) | Tokenization, embeddings, attention, n-gram LM, sentiment workbench |
| Docker packaging | Complete | Non-root multi-stage image, persistent volume, healthcheck, Compose |
| Authentication/multi-tenancy | Not in scope / absent | Local single-user architecture; unsafe to expose directly |
| Comprehensive documentation | Complete for implemented systems | Required architecture/learning/ML/RAG/agent/security/localization/testing docs |

## Audit conclusion

The original repository was already a substantial working game, not a mockup. The upgrade therefore preserved its architecture and added guarded modules/routes/pages rather than replacing it. Claims above intentionally distinguish optional local-model/PyTorch behavior, educational simulators and planned work.
