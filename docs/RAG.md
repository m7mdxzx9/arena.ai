# Retrieval-Augmented Generation

## Two distinct systems

1. **RAG Learning Simulator** — preserved bundled-corpus educational lab for controlled experiments with chunking, retrieval and answer settings.
2. **Personal RAG** — real player-owned document ingestion, retrieval, citations and optional grounded local generation.

The UI labels them separately.

## Accepted personal documents

PDF, TXT, Markdown and DOCX are accepted up to 10 MiB. The extractor enforces character/page/paragraph/chunk limits. Macro-enabled Office files, executables, archives and arbitrary serialized objects are rejected. PDF and DOCX extraction uses bounded parser libraries; no embedded script or macro is executed.

Each source receives a UUID path beneath a configured document root. Ownership is checked for list, query and delete operations. Extracted text and chunks are stored in SQLite; the original is retained only in guarded local storage.

## Retrieval

Personal RAG supports:

- lexical TF-IDF retrieval;
- deterministic local hash embedding/cosine retrieval;
- hybrid rank/score combination;
- configurable bounded top-k;
- per-chunk source, rank, score and metadata.

The local hash representation is honestly identified as deterministic local retrieval—not a neural embedding API.

## Answer modes

### Extractive fallback

Always available offline. It composes an answer from selected passages and returns citations. It is labelled `extractive`, not as an LLM response.

### Grounded Ollama

When explicitly selected, the provider receives the question and retrieved passages with instructions to treat document text as untrusted evidence. The result records provider/model metadata and citations. Provider failure is returned explicitly; no fallback is disguised as generated output.

## Prompt-injection handling

Ingestion flags common instruction-like patterns. Retrieved text is wrapped and labelled as untrusted data. Agent knowledge search similarly marks observations untrusted. Flags are visible to the learner; detection is heuristic and is not represented as a complete security guarantee.

## Citation behavior

Citations reference exact chunk IDs and source names included in the retrieval response. Citation presence proves which local evidence was selected; it does not independently prove the source is correct.

## Current limits

- Scanned/image-only PDFs have no OCR.
- Tables/layout are flattened to text.
- There is no neural cross-encoder reranker.
- Personal document binaries are excluded from portable JSON backup.
