#!/usr/bin/env bash
# One-command local start: installs deps on first run, builds the UI, serves everything on http://localhost:8000
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x backend/.venv/bin/python ]; then
  python3 -m venv backend/.venv
  backend/.venv/bin/pip install -q -r backend/requirements.txt
fi
if [ ! -f frontend/dist/index.html ] || [ "${REBUILD:-0}" = "1" ]; then
  (cd frontend && { [ -d node_modules ] || npm install --silent; } && npm run build)
fi
cd backend
exec .venv/bin/python -m uvicorn neural_forge.app:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}"
