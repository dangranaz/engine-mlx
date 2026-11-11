#!/usr/bin/env bash
#
# stop-server.sh — stop the engine-mlx server started by start-server.sh.
#
# Usage:
#   ./stop-server.sh
set -euo pipefail

info() { printf '\033[0;36m%s\033[0m\n' "$*"; }

RUN_DIR="${TMPDIR:-/tmp}/engine-mlx"
PID_FILE="$RUN_DIR/server.pid"

if [[ ! -f "$PID_FILE" ]]; then
  info "no pid file ($PID_FILE) — server not running (or started another way)."
  exit 0
fi

PID="$(cat "$PID_FILE")"
if kill -0 "$PID" 2>/dev/null; then
  info "stopping engine-mlx (pid $PID)"
  kill "$PID" 2>/dev/null || true
  # Give it a moment, then force if still alive.
  for _ in 1 2 3 4 5; do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
  kill -0 "$PID" 2>/dev/null && kill -9 "$PID" 2>/dev/null || true
  info "stopped."
else
  info "process $PID not running (stale pid file)."
fi
rm -f "$PID_FILE"
