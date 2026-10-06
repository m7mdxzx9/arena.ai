#!/usr/bin/env bash
# One-command local start: installs changed deps, builds the UI, serves http://localhost:8000
set -euo pipefail
cd "$(dirname "$0")"

INSTALL_TORCH="${INSTALL_TORCH:-1}"
if [ ! -x backend/.venv/bin/python ]; then
  python3 -m venv backend/.venv
fi
REQ_INPUT="$(cat backend/requirements.txt)"
if [ "$INSTALL_TORCH" = "1" ]; then
  REQ_INPUT+="$(cat backend/requirements-torch.txt)"
fi
REQ_HASH="$(printf '%s' "$REQ_INPUT" | sha256sum | cut -d' ' -f1)"
STAMP="backend/.venv/.neural-forge-requirements"
if [ ! -f "$STAMP" ] || [ "$(cat "$STAMP")" != "$REQ_HASH" ]; then
  backend/.venv/bin/pip install -q -r backend/requirements.txt
  if [ "$INSTALL_TORCH" = "1" ]; then
    backend/.venv/bin/pip install -q -r backend/requirements-torch.txt
  fi
  printf '%s' "$REQ_HASH" > "$STAMP"
fi

if [ ! -f frontend/dist/index.html ] || [ "${REBUILD:-0}" = "1" ]; then
  (cd frontend && { [ -d node_modules ] || npm ci --silent; } && npm run build)
fi
cd backend
exec .venv/bin/python -m uvicorn neural_forge.app:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}"
