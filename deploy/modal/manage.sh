#!/usr/bin/env bash
# Manage the Modal GPU services. Both services are configured to scale to zero
# (min_containers=0), so an idle/offline server costs nothing automatically.
# Use this script for explicit control.
#
#   ./manage.sh status   # are the apps deployed / any containers running?
#   ./manage.sh stop     # fully undeploy both apps (releases GPU immediately)
#   ./manage.sh start    # redeploy both apps (cached image -> seconds)
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

APPS=(prodrag-embed prodrag-clip prodrag-rerank)
FILES=(deploy/modal/embed_service.py deploy/modal/clip_service.py deploy/modal/rerank_service.py)

status() {
  modal app list 2>/dev/null | grep -i prodrag || echo "no prodrag apps deployed"
  echo
  modal container list 2>/dev/null | grep -i prodrag || echo "no active containers (scaled to zero)"
}

stop() {
  for app in "${APPS[@]}"; do
    echo "== stopping $app =="
    modal app stop -y "$app" 2>&1 | tail -1
  done
}

start() {
  for file in "${FILES[@]}"; do
    echo "== deploying $file =="
    modal deploy "$file" 2>&1 | grep -E "deployed|URL|rror" | tail -2
  done
}

case "${1:-}" in
  status) status ;;
  stop) stop ;;
  start) start ;;
  *) echo "usage: $0 {status|stop|start}"; exit 1 ;;
esac
