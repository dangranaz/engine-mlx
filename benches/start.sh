#!/usr/bin/env bash
#
# start.sh — bring up an OpenAI-compatible engine server for benchmarking.
#
# engine-bench is engine-agnostic: it only speaks HTTP. This script is a thin,
# overridable launcher so you can benchmark any engine with the same tool.
#
# Selection (in priority order):
#   1. BENCH_START_CMD  — an explicit command to run (backgrounded here).
#   2. engine-mlx       — default: delegate to ../engine-mlx/scripts/server.sh
#
# Config (env):
#   ENGINE            engine label + default resolver ("engine-mlx")
#   MODEL             model spec passed to the engine launcher
#   HOST              bind host  (default 127.0.0.1)
#   PORT              bind port  (default 11435 — nexum's standard local port;
#                     11434 is Ollama, 11435 keeps our engine distinct)
#   RELEASE           "1" to build/run optimized (engine-mlx)
#   BENCH_START_CMD   full custom launch command (overrides ENGINE)
#   MLX_C_PATH/MLX_PREFIX  passed through for engine-mlx builds
#
# Usage:
#   ./start.sh                       # engine-mlx, default model, :11435
#   ENGINE=engine-mlx MODEL=Qwen3-0.6B-MLX-4bit RELEASE=1 ./start.sh
#   BENCH_START_CMD="python -m vllm.entrypoints.openai.api_server --port 11435 ..." ./start.sh

set -euo pipefail

ENGINE="${ENGINE:-engine-mlx}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-11435}"
MODEL="${MODEL:-}"
RELEASE="${RELEASE:-0}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$HERE/.run"
mkdir -p "$RUN_DIR"
PID_FILE="$RUN_DIR/${ENGINE}.pid"
LOG_FILE="$RUN_DIR/${ENGINE}.log"

base_url() { echo "http://${HOST}:${PORT}"; }

wait_ready() {
  local url="$1/health"
  local alt="$1/v1/models"
  local timeout="${READY_TIMEOUT:-300}"
  echo "waiting for readiness at $url (or $alt), timeout ${timeout}s..."
  for ((i = 0; i < timeout; i++)); do
    if curl -fsS -o /dev/null "$url" 2>/dev/null || curl -fsS -o /dev/null "$alt" 2>/dev/null; then
      echo "server ready after ${i}s"
      return 0
    fi
    sleep 1
  done
  echo "ERROR: server not ready after ${timeout}s — see $LOG_FILE" >&2
  return 1
}

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "already running (pid $(cat "$PID_FILE")) — use ./stop.sh first"
  exit 0
fi

if [[ -n "${BENCH_START_CMD:-}" ]]; then
  echo "starting via BENCH_START_CMD: $BENCH_START_CMD"
  # shellcheck disable=SC2086
  ( NEXUM_HOST="$HOST" NEXUM_PORT="$PORT" exec bash -c "$BENCH_START_CMD" ) >"$LOG_FILE" 2>&1 &
  echo $! >"$PID_FILE"
elif [[ "$ENGINE" == "engine-mlx" ]]; then
  SERVER_SH="$HERE/../engine-mlx/scripts/server.sh"
  [[ -x "$SERVER_SH" ]] || { echo "ERROR: not found: $SERVER_SH" >&2; exit 1; }
  echo "starting engine-mlx via $SERVER_SH"
  rel_flag=()
  [[ "$RELEASE" == "1" ]] && rel_flag=(--release)
  # server.sh manages its own pidfile; we still track our launcher for stop.sh.
  ( NEXUM_HOST="$HOST" NEXUM_PORT="$PORT" \
      exec "$SERVER_SH" start ${MODEL:+"$MODEL"} "${rel_flag[@]}" ) >"$LOG_FILE" 2>&1 &
  echo $! >"$PID_FILE"
else
  echo "ERROR: unknown ENGINE='$ENGINE' and no BENCH_START_CMD set" >&2
  exit 1
fi

if wait_ready "$(base_url)"; then
  echo "engine '$ENGINE' up at $(base_url)"
else
  tail -n 20 "$LOG_FILE" >&2 || true
  exit 1
fi
