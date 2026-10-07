# Security

## Threat model

NEURAL FORGE treats datasets, documents, backup JSON, prompt variables, evaluation cases and local-model output as untrusted. It is designed for a local single-user deployment. It is **not** an authenticated internet multi-tenant service.

## Upload controls

### Datasets

- accepted: CSV, TSV, JSON records, XLSX;
- rejected: pickle/joblib, macro formats, executables and unknown extensions;
- source, expanded workbook, row, column and cell-length limits;
- one bounded worksheet; formulas retained only as inert text/warnings and never evaluated;
- UUID storage beneath a configured root;
- normalized CSV used for later training;
- player ownership on read/delete/train.

### Documents

- accepted: PDF, TXT, Markdown, DOCX;
- no macro execution, archive expansion, arbitrary object loading or OCR subprocess;
- byte, extraction, page/paragraph, character and chunk limits;
- UUID storage and player ownership;
- prompt-like text is flagged and always passed as untrusted evidence.

### Checkpoints

Only server-produced PyTorch state dictionaries are saved. Uploaded checkpoint/pickle loading is not supported.

## Path handling

Client filenames are metadata only. Stored names are UUIDs. Delete/read operations resolve paths and require them to remain under their configured root. The application never joins a client filename into an executable path.

## Local model transport

The Ollama base URL is validated to HTTP(S), has bounded connect/read timeouts and bounded response size. No paid or remote provider is required. Administrators control `OLLAMA_BASE_URL`; do not point it at untrusted network services.

## Agent controls

Real agents have a fixed registry, Pydantic argument validation, enabled-tool checks, explicit permissions, step/time limits and bounded observations. No unrestricted shell, arbitrary file/network access or dynamic code execution exists. The calculator uses an AST allowlist rather than `eval`.

## Backup/restore

Portable backup is JSON, versioned and capped at 5 MiB. Secret-like keys are recursively redacted. Restore validates top-level format, section types/counts and performs a merge inside one SQLite transaction; validation/database failure rolls back. It never restores executable objects or file paths. Binary uploads are excluded.

## HTML and UI

React escapes normal content. Portfolio HTML export escapes every learner-provided field. JSON/code views render as text. The app does not use user-provided `dangerouslySetInnerHTML`.

## Resource limits

Limits exist for uploads, extracted text, rows/columns, training parameters, prompt/document lengths, agent steps, agent tools, evaluation cases, regex patterns and backup size. Regex evaluation rejects common catastrophic nested quantifiers and truncates evaluated text.

## Secrets

The repository requires no API key. Do not put credentials in prompt variables, documents, experiments or visible agent memory. Docker configuration contains no embedded secrets.

## Deployment guidance

1. Keep the service on localhost or behind authentication and TLS.
2. Mount `/data` on a filesystem with appropriate OS permissions.
3. Run as the non-root container user (the provided Dockerfile does).
4. Back up `/data` separately if binary uploads/checkpoints matter.
5. Restrict Ollama exposure and available models.
6. Apply reverse-proxy request limits and rate limits for any shared deployment.
7. Run dependency audits and tests before upgrades.

## Known limitations

- There is no login/session/CSRF boundary; player IDs are local application selectors, not authorization credentials.
- Synchronous training can consume CPU; local request concurrency should be constrained.
- Parser libraries process complex PDFs/XLSX/DOCX and must be kept patched.
- Prompt-injection detection is heuristic; code-enforced tool permissions remain the actual boundary.
- SQLite is unsuitable for hostile multi-tenant concurrency.
