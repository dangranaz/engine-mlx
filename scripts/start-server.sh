#!/usr/bin/env bash
#
# start-server.sh — start the engine-mlx OpenAI-compatible server (Apple Silicon).
#
# Simple, user-facing launcher: point it at a model directory and it brings the
# server up on http://HOST:PORT with /v1/chat/completions, /v1/models, /health.
#
# Prerequisites:
#   - macOS on Apple Silicon
#   - Homebrew MLX-C:  brew install mlx-c
#   - the engine-mlx binary built with the `mlx` feature, OR run from the
#     engine-mlx repo with cargo (set ENGINE_MLX_DIR to that repo).
#
# Usage:
#   ./start-server.sh /path/to/Model-MLX-4bit          # explicit model dir
#   MODEL=/path/to/model ./start-server.sh             # or via env
#
# Config (env):
#   MODEL            model directory (contains config.json)   [required]
#   HOST             bind host                                [default 127.0.0.1]
#   PORT             bind port                                [default 11435]
#   ENGINE_MLX_BIN   path to a prebuilt engine-mlx-serve binary (optional)
#   ENGINE_MLX_DIR   path to the engine-mlx repo (to run via cargo) [default ~/engine-mlx]
#   READY_TIMEOUT    seconds to wait for /health              [default 300]
set -euo pipefail

MODEL="${MODEL:-${1:-}}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-11435}"
READY_TIMEOUT="${READY_TIMEOUT:-300}"

die()  { printf '\033[0;31merror:\033[0m %s\n' "$*" >&2; exit 1; }
info() { printf '\033[0;36m%s\033[0m\n' "$*"; }

[[ -n "$MODEL" ]] || die "no model given. Usage: ./start-server.sh /path/to/model  (or MODEL=...)"
[[ -f "$MODEL/config.json" ]] || die "not a model dir (no config.json): $MODEL"

# Auto-detect the Homebrew MLX prefixes so the user doesn't have to.
export MLX_C_PATH="${MLX_C_PATH:-$(brew --prefix mlx-c 2>/dev/null || echo /opt/homebrew/opt/mlx-c)}"
export MLX_PREFIX="${MLX_PREFIX:-$(brew --prefix mlx 2>/dev/null || echo /opt/homebrew/opt/mlx)}"

RUN_DIR="${TMPDIR:-/tmp}/engine-mlx"
mkdir -p "$RUN_DIR"
PID_FILE="$RUN_DIR/server.pid"
LOG_FILE="$RUN_DIR/server.log"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  die "server already running (pid $(cat "$PID_FILE")). Use ./stop-server.sh first."
fi

# Choose how to launch: prebuilt binary, or cargo from the engine-mlx repo.
if [[ -n "${ENGINE_MLX_BIN:-}" && -x "${ENGINE_MLX_BIN}" ]]; then
  LAUNCH=("$ENGINE_MLX_BIN" --model "$MODEL" --host "$HOST" --port "$PORT")
else
  ENGINE_MLX_DIR="${ENGINE_MLX_DIR:-$HOME/engine-mlx}"
  [[ -f "$ENGINE_MLX_DIR/Cargo.toml" ]] || die \
    "no prebuilt binary (ENGINE_MLX_BIN) and no engine-mlx repo at ENGINE_MLX_DIR=$ENGINE_MLX_DIR"
  LAUNCH=(cargo run --release --features mlx --manifest-path "$ENGINE_MLX_DIR/Cargo.toml" \
          -p engine-mlx-serve -- --model "$MODEL" --host "$HOST" --port "$PORT")
fi

info "starting engine-mlx on http://${HOST}:${PORT}"
info "  model = $MODEL"
info "  log   = $LOG_FILE"
( ENGINE_MLX_HOST="$HOST" ENGINE_MLX_PORT="$PORT" exec "${LAUNCH[@]}" ) >"$LOG_FILE" 2>&1 &
echo $! >"$PID_FILE"

info "waiting for readiness (/health), timeout ${READY_TIMEOUT}s..."
for ((i = 0; i < READY_TIMEOUT; i++)); do
  if curl -fsS -o /dev/null "http://${HOST}:${PORT}/health" 2>/dev/null; then
    info "server ready at http://${HOST}:${PORT} (after ${i}s)"
    exit 0
  fi
  sleep 1
done
die "server not ready after ${READY_TIMEOUT}s — see $LOG_FILE"
