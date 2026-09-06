#!/usr/bin/env bash
# ProdRag — launch both ends (backend via docker + frontend via streamlit)
# Usage: ./launch.sh [--local] [--no-frontend] [--no-docker]
#   --local       run api/worker locally (uvicorn+arq) instead of docker
#   --no-frontend skip frontend
#   --no-docker   skip docker (use with --local if infra already running)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
BACKEND="$ROOT/backend"
FRONTEND="$ROOT/frontend"
COMPOSE="deploy/docker-compose.yml"
ENV_FILE="$ROOT/.env"
API_URL="http://localhost:8000"
FRONTEND_PORT=8501

LOCAL=0; NO_FRONTEND=0; NO_DOCKER=0
for arg in "$@"; do case $arg in --local) LOCAL=1;; --no-frontend) NO_FRONTEND=1;; --no-docker) NO_DOCKER=1;; *) echo "unknown arg: $arg"; exit 1;; esac; done

need() { command -v "$1" >/dev/null 2>&1 || { echo "missing: $1"; exit 1; }; }
need uv

if [[ ! -f "$ENV_FILE" ]]; then
  echo "no $ENV_FILE — copying from .env.example"
  cp "$ROOT/.env.example" "$ENV_FILE"
  echo "edit $ENV_FILE (PRODRAG_SUPABASE_URL/_SECRET_KEY, PRODRAG_CHAT_API_KEY, PRODRAG_API_TOKEN) then re-run"
  exit 0
fi

# ensure schema is noted
if ! grep -q "SUPABASE" "$ENV_FILE" 2>/dev/null; then
  echo "warn: check .env.example for required PRODRAG_* vars"
fi

cleanup() {
  echo; echo "shutting down..."
  jobs -p | xargs -r kill 2>/dev/null || true
  if [[ $LOCAL -eq 0 && $NO_DOCKER -eq 0 ]]; then
    docker compose -f "$COMPOSE" down 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

wait_api() {
  echo -n "waiting for api $API_URL/health"
  for i in {1..60}; do
    if curl -sf "$API_URL/health" >/dev/null 2>&1; then echo " OK"; return 0; fi
    echo -n "."; sleep 2
  done
  echo " timeout"; return 1
}

# ---- backend ----
if [[ $LOCAL -eq 1 ]]; then
  echo "== backend (local) =="
  # local qdrant/redis still needed unless NO_DOCKER
  if [[ $NO_DOCKER -eq 0 ]]; then
    docker compose -f "$COMPOSE" up -d qdrant redis
    echo "qdrant/redis up (docker)"
    sleep 3
  fi
  # ensure backend deps
  uv sync --extra dev 2>&1 | tail -3
  # apply schema if DATABASE_URL set (best-effort)
  if grep -q "PRODRAG_DATABASE_URL" "$ENV_FILE" && grep -q "postgresql://" "$ENV_FILE"; then
    DB_URL=$(grep PRODRAG_DATABASE_URL "$ENV_FILE" | cut -d= -f2- | tr -d ' "' | tail -1)
    if [[ -n "$DB_URL" ]]; then
      echo "applying schema.sql to $DB_URL"
      psql "$DB_URL" -f "$BACKEND/app/core/schema.sql" 2>&1 | tail -5 || echo "psql failed — apply manually via Supabase SQL editor"
    fi
  fi
  echo "starting api (uvicorn) + worker (arq)..."
  uv run uvicorn app.main:app --app-dir "$BACKEND" --host 0.0.0.0 --port 8000 --reload &
  uv run arq app.worker.WorkerSettings --custom-settings "$BACKEND/app/worker.py" 2>&1 | sed 's/^/[worker] /' &
  # arq alt: if custom-settings not supported, fallback
  if ! wait_api; then
    # fallback worker launch
    echo "retry worker..."
    uv run python -m arq app.worker.WorkerSettings 2>&1 | sed 's/^/[worker] /' &
    wait_api || { echo "api failed to start"; exit 1; }
  fi
else
  echo "== backend (docker) =="
  need docker
  docker compose -f "$COMPOSE" down 2>/dev/null || true
  docker compose -f "$COMPOSE" up --build -d
  docker compose -f "$COMPOSE" logs -f api worker 2>&1 | sed 's/^/[docker] /' &
  wait_api || { echo "docker api not ready — check: docker compose -f $COMPOSE logs api"; exit 1; }
  echo "api + worker + qdrant + redis running (docker)"
  # schema auto-applied? No — remind
  echo "if first run: psql \"\$PRODRAG_DATABASE_URL\" -f backend/app/core/schema.sql"
fi

# ---- frontend ----
if [[ $NO_FRONTEND -eq 1 ]]; then
  echo "== frontend skipped =="
  echo "backend at $API_URL"
  wait
else
  echo "== frontend (streamlit) =="
  uv sync --extra frontend 2>&1 | tail -3
  echo "starting frontend at http://localhost:$FRONTEND_PORT"
  uv run streamlit run "$FRONTEND/app.py" --server.port "$FRONTEND_PORT" --server.headless true &
  # also hint simple test UI
  echo "simple test UI: open $FRONTEND/simple_test.html directly (file://) — CORS allowed"
  echo
  echo "both ends up:"
  echo "  api:      $API_URL/health"
  echo "  docs:     $API_URL/docs"
  echo "  frontend: http://localhost:$FRONTEND_PORT"
  echo "  test ui:  $FRONTEND/simple_test.html"
  echo "  health:   curl -H \"Authorization: Bearer \$(grep PRODRAG_API_TOKEN $ENV_FILE | cut -d= -f2)\" $API_URL/health"
  echo
  echo "logs: docker compose -f $COMPOSE logs -f  (docker mode) or see [worker]/[docker] prefixes"
  echo "stop: Ctrl+C (or ./launch.sh --no-frontend --no-docker to keep infra)"
  wait
fi
