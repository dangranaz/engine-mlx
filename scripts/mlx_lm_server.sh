#!/usr/bin/env bash
#
# mlx_lm_server.sh — start/stop/status the mlx_lm OpenAI-compatible server
# as a comparison baseline (market standard on Apple Silicon).
#
# Launches `python3 -m mlx_lm server --model <M> --host <H> --port <P>`.
# mlx_lm exposes /v1/chat/completions and /v1/models but has NO /health route,
# so readiness is probed on /v1/models instead.
#
# Usage:
#   scripts/mlx_lm_server.sh start
#   scripts/mlx_lm_server.sh stop
#   scripts/mlx_lm_server.sh status
#   scripts/mlx_lm_server.sh restart
#   scripts/mlx_lm_server.sh logs
#
# Config (env):
#   MLXLM_HOST   (default 127.0.0.1)
#   MLXLM_PORT   (default 8081 — distinct from engine-mlx 11435)
#   ENGINE_MLX_MODEL  (model dir or HF id passed to --model)
#   PYTHON       (python interpreter, default python3)
#   READY_TIMEOUT (seconds to wait, default 300)

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

NAME="mlx_lm"
MLXLM_HOST="${MLXLM_HOST:-127.0.0.1}"
MLXLM_PORT="${MLXLM_PORT:-8081}"
PYTHON="${PYTHON:-python3}"
READY_TIMEOUT="${READY_TIMEOUT:-300}"

usage() {
  cat <<EOF
${BOLD}mlx_lm server (baseline)${RESET}

Usage: $(basename "$0") <command>

Commands:
  start     Launch mlx_lm.server in the background, wait for /v1/models
  stop      Gracefully stop it
  restart   stop then start
  status    Show running state and /v1/models probe
  logs      Tail the log file

Env:
  MLXLM_HOST=$MLXLM_HOST  MLXLM_PORT=$MLXLM_PORT
  ENGINE_MLX_MODEL=$ENGINE_MLX_MODEL
  PYTHON=$PYTHON  READY_TIMEOUT=$READY_TIMEOUT
EOF
}

# wait_for_models BASE_URL TIMEOUT — mlx_lm has no /health; poll /v1/models.
wait_for_models() {
  local base="$1" timeout="${2:-300}" url waited=0
  url="$base/v1/models"
  info "waiting for readiness at $url (timeout ${timeout}s)"
  while (( waited < timeout )); do
    if [[ "$(http_get_code "$url")" == "200" ]]; then
      ok "mlx_lm ready after ${waited}s"
      return 0
    fi
    sleep 1; waited=$(( waited + 1 ))
  done
  err "mlx_lm not ready after ${timeout}s"
  return 1
}

cmd_start() {
  require_cmd "$PYTHON" curl

  if ! "$PYTHON" -c "import mlx_lm" >/dev/null 2>&1; then
    die "mlx_lm not importable by $PYTHON — install with: $PYTHON -m pip install mlx-lm"
  fi

  if is_running "$NAME"; then
    warn "$NAME already running (pid $(cat "$(pid_file "$NAME")"))"
    return 0
  fi

  # Model spec: positional arg wins over ENGINE_MLX_MODEL env.
  local model_spec="${1:-$ENGINE_MLX_MODEL}"
  # Try to resolve to a local dir; mlx_lm also accepts a bare HF id, so a
  # resolution miss is a soft fallback (pass the spec through unchanged).
  local model
  if model="$(resolve_model "$model_spec" 2>/dev/null)"; then
    ok "resolved model: $model"
  else
    model="$model_spec"
    warn "'$model_spec' not found locally — passing to --model as-is (may be an HF id)"
  fi

  local base; base="$(base_url "$MLXLM_HOST" "$MLXLM_PORT")"
  local lf; lf="$(log_file "$NAME")"
  local pf; pf="$(pid_file "$NAME")"

  info "starting $NAME on $base"
  log "model = $model"
  log "log   = $lf"

  (
    exec "$PYTHON" -m mlx_lm server \
      --model "$model" \
      --host "$MLXLM_HOST" \
      --port "$MLXLM_PORT"
  ) >"$lf" 2>&1 &
  local pid=$!
  echo "$pid" >"$pf"
  log "pid   = $pid"

  if wait_for_models "$base" "$READY_TIMEOUT"; then
    ok "$NAME up at $base"
  else
    err "readiness failed — check log: $lf"
    tail -n 20 "$lf" >&2 || true
    stop_by_pidfile "$NAME"
    return 1
  fi
}

cmd_status() {
  local base; base="$(base_url "$MLXLM_HOST" "$MLXLM_PORT")"
  if is_running "$NAME"; then
    local pid; pid="$(cat "$(pid_file "$NAME")")"
    local code; code="$(http_get_code "$base/v1/models")"
    ok "$NAME running (pid $pid) — /v1/models -> $code"
  else
    warn "$NAME not running"
  fi
}

cmd_logs() {
  local lf; lf="$(log_file "$NAME")"
  [[ -f "$lf" ]] || die "no log file at $lf"
  tail -f "$lf"
}

main() {
  local cmd="${1:-}"
  shift || true
  case "$cmd" in
    start)   cmd_start "$@" ;;
    stop)    stop_by_pidfile "$NAME" ;;
    restart) stop_by_pidfile "$NAME"; cmd_start "$@" ;;
    status)  cmd_status ;;
    logs)    cmd_logs ;;
    ""|-h|--help|help) usage ;;
    *) err "unknown command: $cmd"; usage; exit 1 ;;
  esac
}

main "$@"
