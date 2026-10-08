# Retrieval-Augmented Generation

NEURAL FORGE has two intentionally separate RAG surfaces. The first teaches fixed-corpus concepts; the second works with documents owned by the current player. Do not describe the simulator's TF-IDF, LSA, feature hashing or heuristic reranking as neural models.

## 1. RAG Learning Simulator

The preserved RAG Archives game uses a fixed, fictional English campus handbook and a bundled evaluation set. It computes chunking, TF-IDF, TruncatedSVD/LSA, deliberately weak 32-feature hashing, BM25, reciprocal-rank fusion, a small corpus-trained bigram model, and an extractive grounded answerer at runtime. Its optional reranker is a lexical-overlap heuristic. This is useful for controlled comparisons, but it is not a general-purpose or neural personal-document RAG system.

The simulator's controls, metric names, diagnosis messages, benchmark questions and displayed gold answers have English/Arabic resources. Retrieved source passages and free-form bigram outputs remain English so learners can inspect the original evidence and computation; that limitation is shown in the UI. Personal files are not sent to or inserted into this fixed corpus.

## 2. Advanced Personal RAG Lab

### Accepted documents and ownership

PDF, TXT, Markdown and DOCX are accepted up to 10 MiB. Extraction enforces character/page/paragraph/chunk limits. Macro-enabled Office files, executables, archives and arbitrary serialized objects are rejected. PDF and DOCX extraction uses bounded parser libraries; no embedded script or macro is executed.

Each source receives a UUID beneath a configured document root. Every list, inspect, re-index, query, experiment and delete path checks the current player's ownership. Extracted text, normalized chunk metadata, optional model vectors and experiment evidence are stored locally. Source passages are treated as untrusted data.

### Chunking and retrieval

Students can inspect and re-index chunks and configure bounded word chunk size/overlap. Available retrieval strategies are dense, BM25 lexical, and weighted Reciprocal Rank Fusion hybrid retrieval; scores, initial ranks and selected evidence are inspectable. Saved experiments keep the query, source IDs, configuration, retrieved chunk IDs, answer, citations, metrics and timestamp so comparisons and learning checks can use actual runs.

### Representation providers (honest labels)

- **Statistical LSA** (`TF-IDF + TruncatedSVD`) is the reliable offline fallback. It is a statistical latent-space representation, not a neural embedding.
- **Sentence Transformers** computes real local neural vectors on CPU when the optional dependency and selected model are already available in the local cache. The application never initiates an automatic model download.
- **Ollama embeddings** use the configured local Ollama embedding endpoint and an installed local embedding model; documents are not sent to a hosted API.
- Lexical-only BM25 is also available when no vector provider is used; dense/hybrid modes reject the missing vector representation rather than ranking zero vectors.

Cached chunk vectors are reused when the provider/model and document chunk are unchanged. If a model is missing, incompatible, or fails, the API reports an explicit error; a model failure is not silently relabelled as neural or hidden behind another result.

Optional Python dependency:

```bash
cd backend
.venv/bin/pip install -r requirements-rag.txt
```

Install/cache a Sentence Transformers model separately if wanted. The runtime loads it with local-only settings. Ollama embedding use requires Ollama running locally and the embedding model already installed there.

### Reranking

The user may disable reranking, use a transparent lexical-overlap heuristic, or select a cached local Sentence Transformers cross-encoder. The heuristic is explicitly labelled *not neural*. The cross-encoder reads the query and each bounded shortlist passage jointly; it is optional, CPU-capable and local-only. Neither reranker can recover a passage omitted from the initial candidate shortlist.

### Answer modes and evidence

- **Offline extractive** mode is always available. It returns evidence selected from retrieved passages with citations; it is not presented as a generative LLM.
- **Grounded local Ollama** mode is explicitly selected. Only the question and retrieved evidence that fits the context budget are sent to the local model. Evidence is delimited and labelled untrusted; instructions embedded in a document cannot authorize tools or override system instructions. The answer records local provider/model metadata and exact chunk citation IDs. Provider failures are explicit rather than disguised as fallback generation.

A citation is traceability evidence, not independent proof that a source is true or that every answer claim is supported.

## Evidence-based missions and boss progression

The side missions use persisted player-owned records rather than self-reported checkboxes:

- retrieval comparison requires the same normalized question, the same nonempty source set, two different retrieval methods and evidence in both saved runs;
- grounding requires an actual retrieved passage and at least one saved citation whose chunk ID occurs in that run's retrieved set;
- the Tutor hint ladder requires levels 1, 2 and 3 for the same concept, each with an actual saved-run context;
- cross-player experiment IDs are rejected and completion rewards are idempotent.

The Retrieval Warden retains its existing training-simulator and safety phases. A final distinct phase validates a saved Advanced RAG experiment, confirms one of its sources belongs to the player's workspace, checks retrieved evidence, validates citation-to-retrieved-chunk membership and accepts only a documented retrieval strategy.

## Known limits

- Scanned/image-only PDFs have no OCR.
- Tables and complex layout are flattened to text.
- Local model availability depends on optional packages, cache contents or the learner's Ollama installation; none was reachable during this verification run.
- The fixed-corpus simulator's source documents remain English despite localized labels, questions and gold answers.
- No hosted embedding/generation service is used by these flows.
- Portable JSON backups exclude document binaries and associated model weights; local full-archive backup rules are documented separately.
