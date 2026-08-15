#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# --reload is dev-only; Replit sets REPLIT_DEPLOYMENT=1 in production
if [ -n "${REPLIT_DEPLOYMENT}" ]; then
  exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8080}" --workers 1
else
  exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8080}" --reload --reload-dir app
fi
