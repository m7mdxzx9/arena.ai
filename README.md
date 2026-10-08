# NEURAL FORGE

**An offline-first AI learning game and personal AI laboratory that takes you from “what is AI?” to reproducible local experiments.**

You start as an *AI Beginner* on an AI campus. You learn each idea before you are tested on it. You predict what an experiment will do, then run it on real data. You diagnose what went wrong, fix it, and beat bosses that each stand for a classic ML failure. The top rank is *AI Research Scientist*.

Metrics, curves, confusion matrices, decision boundaries and retrieval scores are computed by the running code. The RAG Learning Simulator is deliberately separate: it uses a fixed fictional campus corpus with transparent statistical TF-IDF/LSA or feature hashing, BM25, and a lexical-overlap heuristic—not neural embeddings. The Advanced RAG Lab works on the player's own uploaded documents and can use an explicitly available local neural model. Neither lab silently substitutes a hosted service or claims a fallback is a neural model.

```
AI Beginner → AI Student → Junior ML Engineer → ML Engineer → AI Engineer → Research Engineer → AI Research Scientist
```

---

## Quick start

Requirements: Python ≥ 3.10 and Node ≥ 18. No GPU, no API keys, no paid services, no internet access at runtime.

```bash
./run.sh            # creates/updates the venv, installs deps (including PyTorch), builds the UI
# or: INSTALL_TORCH=0 ./run.sh  # smaller install; PyTorch/CNN lab reports unavailable
# open http://localhost:8000
```

To start the pieces by hand:

```bash
# backend (serves the API and the built UI)
python3 -m venv backend/.venv
backend/.venv/bin/pip install -r backend/requirements.txt
# optional real PyTorch/CNN engine:
backend/.venv/bin/pip install -r backend/requirements-torch.txt
(cd frontend && npm ci && npm run build)
cd backend && .venv/bin/python -m uvicorn neural_forge.app:app --host 0.0.0.0 --port 8000

# optional: hot-reloading UI dev server on :5173 (proxies /api to :8000)
cd frontend && npm run dev
```

Progress is saved in `backend/neural_forge_data/neural_forge.sqlite3`. Set `NEURAL_FORGE_DB=/path/to.sqlite3` to use a different file.

Docker is also supported: `docker compose up --build` serves the app on port 8000 and stores state in the `neural-forge-data` volume. The image runs as a non-root user.

---

## What you can do

| Place | What happens there |
|---|---|
| **Campus** (11 areas) | Foundation Academy, Data District, ML Workshop, Evaluation Chamber, Neural Network Tower, Computer Vision Lab, Language Intelligence Center, Generative AI Facility, RAG Archives, Agent Arena, Research Institute. Each area levels up visually as you become proficient in its concepts. |
| **Open Lab** | Non-linear entry to personal datasets, real PyTorch/CNN, local models, tutor, mistake review, personal-document RAG, controlled real agents, prompt versions and deterministic evaluation. |
| **Personal Dataset Workspace** | Guarded CSV/TSV/JSON/XLSX uploads with profiling, ownership, preprocessing and real saved scikit-learn runs. No pickle or executable formats. |
| **PyTorch & CNN Lab** | Optional real optimizer/backprop training, CPU/CUDA reporting, learning curves, checkpoints, training augmentation, confusion matrix, image mistakes and feature maps. |
| **Model/Tutor/RAG/Agent labs** | The Tutor adapts from this player's saved mastery, run metrics, mistake patterns and due reviews; offline explanations and progressive hints remain useful without an LLM. Ollama generation is optional and local. Personal RAG keeps source citations inspectable; agent tools have schemas, permissions, timeouts and observable action traces—never a shell. |
| **Prompt & Evaluation labs** | Persistent prompt versions and variables plus reusable deterministic evaluators for text, numbers, schemas, citations, tool choice and retrieval. |
| **Portfolio & Backup** | Experiment-grounded editable case studies with Markdown/HTML/JSON export, and versioned secret-redacted atomic merge backups. |
| **Knowledge Tree** | 105 concepts in 10 branches: Foundations, Mathematics, Data Science, Machine Learning, Evaluation, Deep Learning, Modern AI, RAG, AI Agents, Responsible AI. Modern AI includes Personalized Tutoring after Structured Output and before RAG; concepts remain gated by their prerequisites. |
| **Lessons** | TEACH → VISUAL → EXAMPLE → (CODE) → GUIDED → PREDICT → PRACTICE → CHALLENGE → REFLECT. Questions are procedurally generated (18 generators), so a retry is a new question, not a memorised one. |
| **Prediction Lab** | 12 "predict, then run" experiments, e.g. *what happens at learning rate 5?* or *does removing the leaky column lower accuracy?* Your guess is compared with the real result and explained. |
| **Data Lab** | Column types, missing values, statistics, histograms, a correlation matrix, target/class balance, duplicates, and an interactive scatter plot. |
| **ML Workbench** | Choose a dataset, target, features, one of 15 models, preprocessing, hyperparameters, split, CV and threshold. You get real metrics, train vs test results, a confusion matrix you can click into, predictions, a feature importance chart, a decision boundary, and the **equivalent scikit-learn code**. |
| **Experiment History** | Every run stores its dataset, features, target, preprocessing, algorithm, hyperparameters, seed, metrics, timestamp and your notes. You can compare runs side by side on accuracy, latency, model size, interpretability and generalisation gap. |
| **10 Dataset Missions** | Student performance, house prices, spam, churn, digit images, animals, synthetic medical triage, customer segmentation, sentiment, imbalanced fraud. Each follows the full loop: inspect the data → hypothesis → configure → predict → real run → diagnose → reflect. |
| **8 Boss Battles** | The Overfitter, The Leak, The Imbalance Titan, The Data Chaos, The Learning Rate Beast, Hallucination, Retrieval Failure, Prompt Injection. Each is a multi-phase fight won by real experiments. The obvious-but-wrong fix does **not** work. |
| **Neural Network Lab** | Set layers, neurons, activation, optimiser, learning rate, batch size, epochs, dropout and L2. Train a from-scratch NumPy MLP (gradient-checked) and watch train/val curves and the decision boundary. Bad settings are allowed and diagnosed. You also get a matching PyTorch snippet. |
| **Language / GenAI labs** | Tokenizer (BPE trained live), attention heatmaps, a bigram language model with temperature, and 2D projections of statistical LSA/TF-IDF vectors (not neural embeddings). |
| **RAG Learning Simulator** | A fixed fictional campus corpus for controlled chunking, retrieval and abstention experiments. Uses statistical TF-IDF/LSA or 32-feature hashing, BM25 and an explicitly heuristic lexical-overlap reranker; its benchmark questions and gold answers are localized, while retrieved passages and free-form bigram outputs remain English. It is not the personal Advanced RAG Lab and does not use neural embeddings. |
| **Advanced RAG Lab** | Upload and inspect owned PDF/TXT/Markdown/DOCX sources; tune chunking and BM25/dense/hybrid retrieval; inspect saved rankings, citations and evaluation. Statistical LSA is the offline fallback; actual local Sentence Transformers/Ollama embeddings and a local cross-encoder are optional, cached/local-only providers. Extractive answering works offline; grounded Ollama generation is opt-in. |
| **Evidence-based RAG & Tutor missions** | Compare saved runs over the same question/documents, verify citations against retrieved chunks, and complete a three-step Tutor hint ladder using actual saved run context. The retrieval boss retains its original educational phases and adds a final personal-RAG evidence check. |
| **Agent Arena** | A deterministic tool-using agent attacked by prompt injection. You switch on code-level defences: channel separation, delimiters, permissions and confirmation. Every message in the trace is labelled **trusted** or **untrusted**. |
| **Code Dojo** | 19 graded Python exercises, from variables to k-NN, vector search and a tool-permission gateway. They run in a resource-limited sandbox, plus a step-by-step variable tracer. |
| **Research Institute** | Open-ended challenges scored on a real metric *and* on engineering quality: fraud cost, a house-price model under a latency budget, and stable cross-validated student prediction. |

### Five difficulty modes

| Mode | Behaviour |
|---|---|
| 1 Beginner | Everything explained, a recommended setup, hints shown automatically after a miss, no penalties |
| 2 Guided | You choose between reasonable options; free hints on request |
| 3 Practice | Mostly independent; hints cost half the XP |
| 4 Engineer | Generated code shown everywhere, code cards in lessons, harder items earlier |
| 5 Research Challenge | No hints, stricter boss thresholds, open-ended challenges |

### Adaptive learning

This is not a black box. Each concept carries a **Bayesian Knowledge Tracing** estimate of P(known), updated by every answer, prediction, mission, boss phase and code exercise. **Leitner spaced review** schedules revisits. The adaptation rules are explicit: struggle → mini-lesson, visual and easier items, then a retry; mastery → quick checks and harder items. See [docs/ADAPTIVE_LEARNING.md](docs/ADAPTIVE_LEARNING.md).

### English / العربية

Language is selected in Profile & Settings and persists. Arabic enables document-level RTL while code, equations, commands, model identifiers, JSON and charts remain LTR. Learning content includes reviewed Arabic text/questions for 21 of 105 concepts, the new Personalized Tutoring concept, RAG/Tutor missions, retrieval-boss phases, and the simulator's benchmark questions and answer keys. This is still partial coverage: unlisted legacy curriculum/widget prose, simulator source passages and free-form bigram outputs remain English. See [docs/LOCALIZATION.md](docs/LOCALIZATION.md) and the [capability matrix](CAPABILITY_MATRIX.md) for exact coverage and limits.

---

## Tests

```bash
# Backend API/unit/integration/regression suite
cd backend && PYTHONPATH=. .venv/bin/python -m pytest -q

# UI route/gameplay tests target a running backend (NF_API overrides the default :8000)
# Start it in another shell: cd backend && .venv/bin/python -m uvicorn neural_forge.app:app --host 0.0.0.0 --port 8000
cd frontend && npm run test:ui

# Static checks and production build
npm run lint           # 0 errors; 22 warnings (lint is not warning-free)
npm run build
```

Latest verified on this tree: backend **138 passed, 4 skipped**; frontend **4 files / 13 tests passed**; lint **0 errors, 22 warnings**; production build passed (65 modules).

---

## Repository layout

```
backend/
  neural_forge/
    app.py            FastAPI app: /api routes + serves frontend/dist
    game.py           game logic: progress, XP, ranks, lessons, missions, bosses, challenges
    mastery.py        BKT + Leitner + adaptation rules (pure functions)
    curriculum/       105 concepts (teach/visual/example/reflect content) + question generators
    datasets.py       19 generated, reproducible datasets
    datalab.py        data profiling
    ml.py             experiment runner (sklearn), pipeline errors that teach, code generation
    nn.py             NumPy MLP with backprop, optimisers, dropout, diagnosis
    rag.py, corpus.py retrieval + grounded answering over a built-in document corpus
    agent.py          prompt-injection sandbox agent with switchable defences
    predictions.py    predict-then-run experiments
    missions.py       dataset missions + research challenges
    bosses.py         8 boss battles
    playgrounds.py    data for interactive visuals (gradient descent, k-means, attention, …)
    sandbox.py        subprocess runner for learner Python
    user_datasets.py, torch_engine.py, document_rag.py, real_agent.py
    prompts.py, evaluation.py, tutor.py, mistakes.py, portfolio.py, backup.py
  tests/              pytest suite
frontend/
  src/pages, src/labs, src/widgets   React + TypeScript UI (no UI framework, hand-drawn SVG charts)
  src/locales, src/i18n.tsx          structured bilingual resources and RTL state
  tests/                             vitest UI smoke + gameplay flow + unit tests
docs/                               architecture, learning, Tutor, engines, RAG, agents, security, localization and testing
```

## Data and licensing

All datasets are **generated in code** with fixed seeds (`backend/neural_forge/datasets.py`), with documented causal structure such as the deliberate leak, deliberate noise and deliberate imbalance. Digit images are drawn procedurally, and the RAG corpus is written for the game. The medical dataset is synthetic and for education only. No third-party data is downloaded.
