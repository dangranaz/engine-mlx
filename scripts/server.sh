#!/usr/bin/env bash
#
# server.sh — start/stop/status/restart the engine-mlx OpenAI-compatible server.
#
# The server binary is `engine-mlx-serve`, launched via cargo with the `mlx`
# feature. It reads ENGINE_MLX_HOST / ENGINE_MLX_PORT / ENGINE_MLX_MODEL from the environment
# and exposes /v1/chat/completions, /v1/models, /health.
#
# Usage:
#   scripts/server.sh start <model> [--release]
#   scripts/server.sh stop
#   scripts/server.sh status
#   scripts/server.sh restart <model> [--release]
#   scripts/server.sh logs             # tail the server log
#   scripts/server.sh models list      # list discovered models
#   scripts/server.sh models scan      # (re)generate the registry.json
#
# <model> may be:
#   - an absolute path to a valid model dir (contains config.json)
#   - a registry alias, or "default"
#   - "<org>/<name>" or a bare "<name>" resolved under ENGINE_MLX_MODELS_DIR
# If <model> is omitted, ENGINE_MLX_MODEL (env) is used. Resolution errors are fatal.
#
# Config (env):
#   ENGINE_MLX_HOST         (default 127.0.0.1)
#   ENGINE_MLX_PORT         (default 11435)
#   ENGINE_MLX_MODEL        (fallback model spec when start receives no argument)
#   ENGINE_MLX_MODELS_DIR   (default \$HOME/models — search root)
#   READY_TIMEOUT      (seconds to wait for /health, default 300 — load is slow)

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

NAME="engine-mlx"
READY_TIMEOUT="${READY_TIMEOUT:-300}"

usage() {
  cat <<EOF
${BOLD}engine-mlx server${RESET}

Usage: $(basename "$0") <command> [args]

Commands:
  start <model> [--release]   Resolve <model>, build+run in background, wait /health
  stop                        Gracefully stop the running server
  restart <model> [--release] stop then start
  status                      Show running state and /health probe
  logs                        Tail the server log file
  models list                 List models discovered under ENGINE_MLX_MODELS_DIR
  models scan                 (Re)generate \$ENGINE_MLX_MODELS_DIR/registry.json

<model>:
  absolute path | registry alias | "default" | "<org>/<name>" | bare "<name>"
  omit to fall back to ENGINE_MLX_MODEL.

Env:
  ENGINE_MLX_HOST=$ENGINE_MLX_HOST  ENGINE_MLX_PORT=$ENGINE_MLX_PORT
  ENGINE_MLX_MODEL=$ENGINE_MLX_MODEL
  ENGINE_MLX_MODELS_DIR=$ENGINE_MLX_MODELS_DIR
  READY_TIMEOUT=$READY_TIMEOUT
EOF
}

cmd_start() {
  # Parse args: an optional model spec (positional) and --release (any position).
  local profile="dev"
  local cargo_profile_flag=()
  local model_spec=""
  local a
  for a in "$@"; do
    case "$a" in
      --release) profile="release"; cargo_profile_flag=(--release) ;;
      -*) err "unknown start option: $a"; return 1 ;;
      *) if [[ -z "$model_spec" ]]; then model_spec="$a"; else err "unexpected extra argument: $a"; return 1; fi ;;
    esac
  done
  # Fall back to ENGINE_MLX_MODEL env when no positional model was given.
  [[ -z "$model_spec" ]] && model_spec="$ENGINE_MLX_MODEL"

  require_cmd cargo curl

  if is_running "$NAME"; then
    warn "$NAME already running (pid $(cat "$(pid_file "$NAME")"))"
    return 0
  fi

  # Resolve the model spec to an absolute dir. Not-found / ambiguous -> fatal.
  local model
  if ! model="$(resolve_model "$model_spec")"; then
    die "could not resolve model '$model_spec' — aborting start"
  fi
  ok "resolved model: $model"

  local base; base="$(base_url "$ENGINE_MLX_HOST" "$ENGINE_MLX_PORT")"
  local lf; lf="$(log_file "$NAME")"
  local pf; pf="$(pid_file "$NAME")"

  info "starting $NAME ($profile) on $base"
  log "model = $model"
  log "log   = $lf"

  # Launch detached; cargo run compiles first (may be slow on first build).
  # Env is exported so the child process picks up host/port/model.
  (
    cd "$REPO_ROOT"
    ENGINE_MLX_HOST="$ENGINE_MLX_HOST" ENGINE_MLX_PORT="$ENGINE_MLX_PORT" ENGINE_MLX_MODEL="$model" \
      exec cargo run "${cargo_profile_flag[@]}" -p engine-mlx-serve --features mlx
  ) >"$lf" 2>&1 &
  local pid=$!
  echo "$pid" >"$pf"
  log "pid   = $pid"

  if wait_for_health "$base" "$READY_TIMEOUT"; then
    ok "$NAME up at $base"
  else
    err "readiness failed — check log: $lf"
    err "last 20 log lines:"
    tail -n 20 "$lf" >&2 || true
    stop_by_pidfile "$NAME"
    return 1
  fi
}

# cmd_models list|scan
cmd_models() {
  local sub="${1:-list}"
  case "$sub" in
    list)
      info "models under $ENGINE_MLX_MODELS_DIR"
      local found=0 rel path
      while IFS=$'\t' read -r rel path; do
        [[ -z "$rel" ]] && continue
        printf '  %s%-45s%s %s%s%s\n' "$BOLD" "$rel" "$RESET" "$DIM" "$path" "$RESET"
        found=$(( found + 1 ))
      done < <(scan_models)
      [[ "$found" -eq 0 ]] && warn "no models found (need a config.json under <org>/<name>)"
      log "$found model(s)"
      ;;
    scan)
      require_cmd jq
      local reg="$ENGINE_MLX_MODELS_REGISTRY"
      info "scanning $ENGINE_MLX_MODELS_DIR -> $reg"
      # Build { "default": null, "models": { "<org/name>": "<abs>", "<name>": "<abs>" } }
      # Bare names are added only when unambiguous.
      local tmp; tmp="$(mktemp)"
      scan_models >"$tmp"
      local models_json
      models_json="$(awk -F'\t' '
        { full[$1]=$2; n=split($1,a,"/"); bare=a[n]; barecount[bare]++; barepath[bare]=$2 }
        END {
          printf "{"
          first=1
          for (k in full) {
            if (!first) printf ","
            first=0
            printf "\"%s\":\"%s\"", k, full[k]
          }
          for (b in barecount) {
            if (barecount[b]==1) {
              printf ",\"%s\":\"%s\"", b, barepath[b]
            }
          }
          printf "}"
        }' "$tmp")"
      rm -f "$tmp"
      # Validate & pretty-print via jq; preserve existing "default" if present.
      local prev_default="null"
      if [[ -f "$reg" ]]; then
        prev_default="$(jq -c '.default // null' "$reg" 2>/dev/null || echo null)"
      fi
      jq -n --argjson models "$models_json" --argjson def "$prev_default" \
        '{default: $def, models: $models}' >"$reg"
      ok "wrote registry with $(jq '.models | length' "$reg") entries to $reg"
      log "(edit .default in $reg to set a default model)"
      ;;
    *) err "unknown models subcommand: $sub (use list|scan)"; return 1 ;;
  esac
}

cmd_stop() {
  stop_by_pidfile "$NAME"
}

cmd_status() {
  status_of "$NAME" "$(base_url "$ENGINE_MLX_HOST" "$ENGINE_MLX_PORT")"
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
    stop)    cmd_stop ;;
    restart) cmd_stop; cmd_start "$@" ;;
    status)  cmd_status ;;
    logs)    cmd_logs ;;
    models)  cmd_models "$@" ;;
    ""|-h|--help|help) usage ;;
    *) err "unknown command: $cmd"; usage; exit 1 ;;
  esac
}

main "$@"
