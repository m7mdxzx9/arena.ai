# Engineering Signature

This upgrade is characterized by the following verifiable decisions:

1. **Extension, not replacement.** Existing campaign, BKT/Leitner state, missions, experiments, scikit-learn, NumPy NN, Code Dojo and educational simulators remain in their original architecture.
2. **Truthful execution boundaries.** Real PyTorch, Ollama, personal RAG and controlled agents live beside—not behind—the preserved simulators. Optional dependencies and provider outages are visible.
3. **Untrusted-input discipline.** Dataset/document/backup/evaluation inputs have format, ownership, path and resource boundaries. No arbitrary pickle/checkpoint or agent shell execution is accepted.
4. **Evidence over decoration.** Curves, confusion matrices, mistakes, checkpoints, evaluation results and portfolio projects derive from saved execution outputs.
5. **Code-enforced agent safety.** Tool schemas, allowlists, separate permissions, timeouts, bounded memory and action/observation traces are implementation constraints rather than prompt-only promises.
6. **Offline-first core.** Paid APIs are neither configured nor required. Ollama is local and optional.
7. **Structured localization foundation.** Locale resources, persisted language, document direction and technical LTR isolation are centralized and tested. Remaining legacy English prose is disclosed rather than called complete.
8. **Portable but explicit backup.** Restore is a validated atomic merge. Binary/private file exclusions are carried in the manifest.

## Reproduce the signature

```bash
cd backend && .venv/bin/python -m pytest -q
cd ../frontend && npm run lint && npm run build && npm run test:ui
curl http://localhost:8000/api/system/capabilities
```

Compare implementation claims with `CAPABILITY_MATRIX.md`, security boundaries with `docs/SECURITY.md`, and exact delivered changes/results with `CHANGE_MANIFEST.md`.

This is a project engineering signature, not a cryptographic author identity or a claim about any hidden runtime model.
