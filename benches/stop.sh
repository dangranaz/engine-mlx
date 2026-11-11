#!/usr/bin/env bash
#
# stop.sh — stop the engine server started by start.sh.
#
# For engine-mlx it delegates to server.sh stop (graceful). For custom
# BENCH_START_CMD launches it kills the tracked launcher process group.
#
# Config (env): ENGINE (default "engine-mlx"), matching start.sh.

set -euo pipefail

ENGINE="${ENGINE:-engine-mlx}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$HERE/.run"
PID_FILE="$RUN_DIR/${ENGINE}.pid"

if [[ "$ENGINE" == "engine-mlx" && -z "${BENCH_START_CMD:-}" ]]; then
  SERVER_SH="$HERE/../engine-mlx/scripts/server.sh"
  if [[ -x "$SERVER_SH" ]]; then
    echo "stopping engine-mlx via $SERVER_SH"
    "$SERVER_SH" stop || true
  fi
fi

if [[ -f "$PID_FILE" ]]; then
  pid="$(cat "$PID_FILE")"
  if kill -0 "$pid" 2>/dev/null; then
    echo "stopping launcher pid $pid"
    kill "$pid" 2>/dev/null || true
    sleep 1
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
fi

echo "stopped '$ENGINE'"
