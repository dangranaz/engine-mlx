#!/usr/bin/env bash
#
# common.sh — shared library for engine-mlx server/bench scripts.
#
# Sourced by server.sh, mlx_lm_server.sh, bench.sh, bench_compare.sh.
# Provides: colored logging, env config, dependency checks, HTTP helpers,
# and readiness waiting on the OpenAI-compatible /health endpoint.
#
# Model-agnostic: nothing here assumes Qwen3. Model is selected via ENGINE_MLX_MODEL.

# ── Strict mode (only when executed, not when sourced into set -e caller) ──
# Callers set their own `set -euo pipefail`; we avoid overriding it here.

# ── Paths ──────────────────────────────────────────────────────────────────
# SCRIPTS_DIR resolves to the scripts/ directory regardless of caller CWD.
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC2034  # REPO_ROOT is used by scripts that source this lib (e.g. server.sh)
REPO_ROOT="$(cd "$SCRIPTS_DIR/.." && pwd)"
RUN_DIR="${ENGINE_MLX_RUN_DIR:-$SCRIPTS_DIR/.run}"
mkdir -p "$RUN_DIR"

# ── Colors (TTY-aware, matches wizard style) ────────────────────────────────
if [[ -t 1 ]] && command -v tput >/dev/null 2>&1 && [[ "$(tput colors 2>/dev/null || echo 0)" -ge 8 ]]; then
  BOLD=$(tput bold); DIM=$(tput dim); RESET=$(tput sgr0)
  BLUE=$(tput setaf 4); GREEN=$(tput setaf 2); YELLOW=$(tput setaf 3); RED=$(tput setaf 1)
else
  BOLD=""; DIM=""; RESET=""; BLUE=""; GREEN=""; YELLOW=""; RED=""
fi

# ── Logging ─────────────────────────────────────────────────────────────────
_ts() { date +"%H:%M:%S"; }
log()  { printf '%s%s%s %s\n'      "$DIM" "$(_ts)" "$RESET" "$*"; }
info() { printf '%s%s▸%s %s\n'     "$BOLD" "$BLUE" "$RESET" "$*"; }
ok()   { printf '  %s✓%s %s\n'     "$GREEN" "$RESET" "$*"; }
warn() { printf '  %s⚠%s %s\n'     "$YELLOW" "$RESET" "$*" >&2; }
err()  { printf '  %s✗%s %s\n'     "$RED" "$RESET" "$*" >&2; }
die()  { err "$*"; exit 1; }

# ── Dependency checks ────────────────────────────────────────────────────────
require_cmd() {
  local missing=0
  for c in "$@"; do
    if ! command -v "$c" >/dev/null 2>&1; then
      err "required command not found: $c"
      missing=1
    fi
  done
  [[ "$missing" -eq 0 ]] || die "install the missing dependencies and retry"
}

# ── Config (env with sane defaults) ──────────────────────────────────────────
# Host/port are per-server; callers may override before sourcing or via env.
ENGINE_MLX_HOST="${ENGINE_MLX_HOST:-127.0.0.1}"
ENGINE_MLX_PORT="${ENGINE_MLX_PORT:-11435}"
# Model dir (absolute path to an MLX model). No Qwen3 assumption baked in;
# this default is only a convenience fallback for local dev.
ENGINE_MLX_MODEL="${ENGINE_MLX_MODEL:-$HOME/models/lmstudio-community/Qwen3-0.6B-MLX-4bit}"
# Root under which models are searched by name. Layout is HuggingFace-style:
#   $ENGINE_MLX_MODELS_DIR/<org>/<model-name>/config.json
ENGINE_MLX_MODELS_DIR="${ENGINE_MLX_MODELS_DIR:-$HOME/models}"
# Optional registry file (alias -> path, plus an optional "default"). If present
# it takes priority over the dynamic scan. Regenerate with `server.sh models scan`.
ENGINE_MLX_MODELS_REGISTRY="${ENGINE_MLX_MODELS_REGISTRY:-$ENGINE_MLX_MODELS_DIR/registry.json}"

# base_url HOST PORT -> http://HOST:PORT
base_url() {
  local host="${1:-$ENGINE_MLX_HOST}" port="${2:-$ENGINE_MLX_PORT}"
  printf 'http://%s:%s' "$host" "$port"
}

# ── Model resolution ─────────────────────────────────────────────────────────
# A model dir is "valid" if it contains a config.json (marker of a model dir).
is_model_dir() { [[ -f "$1/config.json" ]]; }

# scan_models — print "<org>/<name>\t<abs_path>" for every valid model dir found
# under ENGINE_MLX_MODELS_DIR at depth <org>/<name>. Sorted, deduped.
scan_models() {
  local root="$ENGINE_MLX_MODELS_DIR"
  [[ -d "$root" ]] || return 0
  local d
  # depth-2 dirs: <root>/<org>/<name>
  while IFS= read -r d; do
    if is_model_dir "$d"; then
      local rel="${d#"$root"/}"
      printf '%s\t%s\n' "$rel" "$d"
    fi
  done < <(find "$root" -mindepth 2 -maxdepth 2 -type d 2>/dev/null | sort)
}

# registry_lookup KEY — if a JSON registry exists, print the path mapped to KEY.
# Registry shape: { "default": "<alias-or-path>", "models": { "<alias>": "<path>" } }
# KEY may be an alias, or the literal "default". Prints nothing if not found.
registry_lookup() {
  local key="$1"
  [[ -f "$ENGINE_MLX_MODELS_REGISTRY" ]] || return 0
  command -v jq >/dev/null 2>&1 || return 0
  if [[ "$key" == "default" ]]; then
    local def
    def="$(jq -r '.default // empty' "$ENGINE_MLX_MODELS_REGISTRY" 2>/dev/null)"
    [[ -z "$def" ]] && return 0
    # default may itself be an alias -> resolve one more hop
    local p
    p="$(jq -r --arg k "$def" '.models[$k] // empty' "$ENGINE_MLX_MODELS_REGISTRY" 2>/dev/null)"
    [[ -n "$p" ]] && { printf '%s' "$p"; return 0; }
    printf '%s' "$def"   # default was itself a path/alias handled downstream
    return 0
  fi
  jq -r --arg k "$key" '.models[$k] // empty' "$ENGINE_MLX_MODELS_REGISTRY" 2>/dev/null
}

# resolve_model SPEC — resolve a user-provided model spec to an absolute dir.
# Resolution order (hybrid, option C):
#   1. Absolute path that is a valid model dir  -> use as-is
#   2. Registry alias (or "default")            -> mapped path
#   3. Dynamic scan under ENGINE_MLX_MODELS_DIR:
#        - exact "<org>/<name>" match
#        - unique "<name>" match (bare name)
# On success prints the absolute path and returns 0.
# On not-found or ambiguity prints an error (with a list) to stderr, returns 1.
resolve_model() {
  local spec="$1"
  [[ -n "$spec" ]] || { err "no model specified"; return 1; }

  # 1. Absolute path
  if [[ "$spec" == /* ]]; then
    if is_model_dir "$spec"; then printf '%s' "$spec"; return 0; fi
    err "path is not a valid model dir (no config.json): $spec"
    return 1
  fi

  # 2. Registry alias / default
  local reg; reg="$(registry_lookup "$spec")"
  if [[ -n "$reg" ]]; then
    # registry value may be absolute or relative to ENGINE_MLX_MODELS_DIR
    [[ "$reg" != /* ]] && reg="$ENGINE_MLX_MODELS_DIR/$reg"
    if is_model_dir "$reg"; then printf '%s' "$reg"; return 0; fi
    err "registry maps '$spec' to '$reg' but it is not a valid model dir"
    return 1
  fi

  # 3. Dynamic scan
  local matches=() rel path
  while IFS=$'\t' read -r rel path; do
    [[ -z "$rel" ]] && continue
    if [[ "$rel" == "$spec" || "${rel##*/}" == "$spec" ]]; then
      matches+=("$path")
    fi
  done < <(scan_models)

  case "${#matches[@]}" in
    0)
      err "model not found: '$spec' (searched $ENGINE_MLX_MODELS_DIR)"
      err "run 'server.sh models list' to see available models"
      return 1 ;;
    1)
      printf '%s' "${matches[0]}"; return 0 ;;
    *)
      err "ambiguous model '$spec' — ${#matches[@]} matches, qualify with <org>/<name>:"
      local m
      for m in "${matches[@]}"; do err "  - ${m#"$ENGINE_MLX_MODELS_DIR"/}"; done
      return 1 ;;
  esac
}

# ── HTTP helpers ─────────────────────────────────────────────────────────────
# http_get_code URL -> prints HTTP status code (000 on connection failure)
http_get_code() {
  local code
  code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$1" 2>/dev/null)"
  # curl prints "000" on connection failure and may also exit non-zero; guard
  # against an empty result so callers always get a 3-digit code.
  [[ -n "$code" ]] && printf '%s' "$code" || printf '000'
}

# wait_for_health BASE_URL [TIMEOUT_S] [INTERVAL_S]
# Polls BASE_URL/health until 200 or timeout. Returns 0 on ready, 1 on timeout.
wait_for_health() {
  local base="$1" timeout="${2:-120}" interval="${3:-1}"
  local url="$base/health"
  local waited=0
  info "waiting for readiness at $url (timeout ${timeout}s)"
  while (( waited < timeout )); do
    if [[ "$(http_get_code "$url")" == "200" ]]; then
      ok "server ready after ${waited}s"
      return 0
    fi
    sleep "$interval"
    waited=$(( waited + interval ))
  done
  err "server not ready after ${timeout}s"
  return 1
}

# ── PID file management ───────────────────────────────────────────────────────
# pid_file NAME -> path to the pid file for a named server
pid_file() { printf '%s/%s.pid' "$RUN_DIR" "$1"; }
log_file() { printf '%s/%s.log' "$RUN_DIR" "$1"; }

# is_running NAME -> 0 if the pid file exists and process is alive
is_running() {
  local pf; pf="$(pid_file "$1")"
  [[ -f "$pf" ]] || return 1
  local pid; pid="$(cat "$pf" 2>/dev/null || echo "")"
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

# stop_by_pidfile NAME [TIMEOUT_S] — graceful TERM then KILL, removes pid file
stop_by_pidfile() {
  local name="$1" timeout="${2:-15}"
  local pf; pf="$(pid_file "$name")"
  if ! is_running "$name"; then
    warn "$name not running (no live pid)"
    rm -f "$pf"
    return 0
  fi
  local pid; pid="$(cat "$pf")"
  info "stopping $name (pid $pid)"
  kill -TERM "$pid" 2>/dev/null || true
  local waited=0
  while (( waited < timeout )); do
    kill -0 "$pid" 2>/dev/null || { ok "$name stopped"; rm -f "$pf"; return 0; }
    sleep 1; waited=$(( waited + 1 ))
  done
  warn "$name did not stop gracefully, sending KILL"
  kill -KILL "$pid" 2>/dev/null || true
  rm -f "$pf"
  ok "$name killed"
}

# status_of NAME BASE_URL — prints running state + health probe
status_of() {
  local name="$1" base="$2"
  if is_running "$name"; then
    local pid; pid="$(cat "$(pid_file "$name")")"
    local code; code="$(http_get_code "$base/health")"
    ok "$name running (pid $pid) — /health -> $code"
  else
    warn "$name not running"
  fi
}

# ── Numeric helpers (percentiles) ─────────────────────────────────────────────
# percentile P  (reads whitespace/newline-separated numbers from stdin)
# P is 0..100. Uses nearest-rank method. Prints the value or empty if no input.
percentile() {
  local p="$1"
  awk -v p="$p" '
    { a[n++] = $1 }
    END {
      if (n == 0) { print ""; exit }
      # sort
      for (i = 0; i < n; i++)
        for (j = i+1; j < n; j++)
          if (a[j] < a[i]) { t = a[i]; a[i] = a[j]; a[j] = t }
      rank = int((p/100.0) * (n-1) + 0.5)
      if (rank < 0) rank = 0
      if (rank >= n) rank = n-1
      printf "%.3f", a[rank]
    }'
}

# mean (reads numbers from stdin)
mean() {
  awk '{ s += $1; n++ } END { if (n>0) printf "%.3f", s/n; else print "" }'
}
