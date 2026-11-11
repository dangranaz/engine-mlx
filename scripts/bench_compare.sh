#!/usr/bin/env bash
#
# bench_compare.sh — orchestrate a head-to-head benchmark of engine-mlx vs
# mlx_lm.server on the SAME model, prompt, and matrix, then print a comparison.
#
# It starts each server on its own port, waits for readiness, runs bench.sh
# against each, saves per-engine JSON, and prints a side-by-side table.
# Servers it started are stopped on exit (even on error/Ctrl-C).
#
# Model-agnostic: set ENGINE_MLX_MODEL to the model both engines should serve.
#
# Usage:
#   scripts/bench_compare.sh [options]
#
# Options:
#   --matrix "A B C"   max_tokens points (default "128")
#   --prompt TEXT      prompt (default fixed sentence)
#   --repeat N         reps per point (default 5)
#   --model SPEC       model both engines serve (path|alias|org/name|name);
#                      omit to fall back to ENGINE_MLX_MODEL
#   --release          build engine-mlx in release mode
#   --skip-engine      don't start engine-mlx (assume already running on 11435)
#   --skip-mlxlm       don't start mlx_lm (assume already running on 8081)
#   -h, --help
#
# Env:
#   ENGINE_MLX_MODEL   model dir served by BOTH engines (required to be meaningful)
#   ENGINE_MLX_PORT    engine-mlx port (default 11435)
#   MLXLM_PORT    mlx_lm port (default 8081)

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

MATRIX="128"
PROMPT="Explain the theory of relativity in simple terms."
REPEAT=5
RELEASE=""
SKIP_ENGINE=0
SKIP_MLXLM=0
MODEL_SPEC=""

ENGINE_PORT="${ENGINE_MLX_PORT:-11435}"
MLXLM_PORT="${MLXLM_PORT:-8081}"

STARTED_ENGINE=0
STARTED_MLXLM=0

usage() { sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --matrix)      MATRIX="$2"; shift 2 ;;
    --prompt)      PROMPT="$2"; shift 2 ;;
    --repeat)      REPEAT="$2"; shift 2 ;;
    --model)       MODEL_SPEC="$2"; shift 2 ;;
    --release)     RELEASE="--release"; shift ;;
    --skip-engine) SKIP_ENGINE=1; shift ;;
    --skip-mlxlm)  SKIP_MLXLM=1; shift ;;
    -h|--help)     usage; exit 0 ;;
    *) err "unknown option: $1"; usage; exit 1 ;;
  esac
done

require_cmd curl jq

SERVER_SH="$SCRIPTS_DIR/server.sh"
MLXLM_SH="$SCRIPTS_DIR/mlx_lm_server.sh"
BENCH_SH="$SCRIPTS_DIR/bench.sh"

ENGINE_JSON="$RUN_DIR/compare_engine-mlx.json"
MLXLM_JSON="$RUN_DIR/compare_mlx_lm.json"

cleanup() {
  [[ "$STARTED_ENGINE" -eq 1 ]] && { info "stopping engine-mlx (started by compare)"; ENGINE_MLX_PORT="$ENGINE_PORT" "$SERVER_SH" stop || true; }
  [[ "$STARTED_MLXLM" -eq 1 ]] && { info "stopping mlx_lm (started by compare)"; MLXLM_PORT="$MLXLM_PORT" "$MLXLM_SH" stop || true; }
}
trap cleanup EXIT INT TERM

# ── Start servers ─────────────────────────────────────────────────────────────
if [[ "$SKIP_ENGINE" -eq 0 ]]; then
  info "═══ starting engine-mlx on :$ENGINE_PORT ═══"
  # shellcheck disable=SC2086  # $RELEASE is an intentional word-split flag
  ENGINE_MLX_PORT="$ENGINE_PORT" "$SERVER_SH" start $MODEL_SPEC $RELEASE
  STARTED_ENGINE=1
else
  warn "skipping engine-mlx start (assuming :$ENGINE_PORT is up)"
fi

if [[ "$SKIP_MLXLM" -eq 0 ]]; then
  info "═══ starting mlx_lm on :$MLXLM_PORT ═══"
  # shellcheck disable=SC2086  # $MODEL_SPEC may be empty (falls back to ENGINE_MLX_MODEL)
  MLXLM_PORT="$MLXLM_PORT" "$MLXLM_SH" start $MODEL_SPEC
  STARTED_MLXLM=1
else
  warn "skipping mlx_lm start (assuming :$MLXLM_PORT is up)"
fi

# ── Run benchmarks ────────────────────────────────────────────────────────────
echo
info "═══ benchmarking engine-mlx ═══"
"$BENCH_SH" \
  --url "http://127.0.0.1:$ENGINE_PORT" \
  --label "engine-mlx" \
  --model "engine-mlx" \
  --prompt "$PROMPT" \
  --matrix "$MATRIX" \
  --repeat "$REPEAT" \
  --json "$ENGINE_JSON"

echo
info "═══ benchmarking mlx_lm ═══"
"$BENCH_SH" \
  --url "http://127.0.0.1:$MLXLM_PORT" \
  --label "mlx_lm" \
  --model "mlx_lm" \
  --prompt "$PROMPT" \
  --matrix "$MATRIX" \
  --repeat "$REPEAT" \
  --json "$MLXLM_JSON"

# ── Comparison table ──────────────────────────────────────────────────────────
echo
info "═══ Comparison (engine-mlx vs mlx_lm) ═══"
log "model = ${ENGINE_MLX_MODEL}"
echo

if [[ ! -f "$ENGINE_JSON" || ! -f "$MLXLM_JSON" ]]; then
  err "missing result JSON; cannot build comparison table"
  exit 1
fi

# Join by max_tokens and print tps + latency deltas.
printf '%s%-10s %14s %14s %12s %14s%s\n' \
  "$BOLD" "max_tokens" "engine t/s" "mlx_lm t/s" "delta %" "eng p95 ms" "$RESET"

jq -rn \
  --slurpfile e "$ENGINE_JSON" \
  --slurpfile m "$MLXLM_JSON" '
  ($e[0].points) as $ep |
  ($m[0].points) as $mp |
  $ep[] as $p |
  ($mp[] | select(.max_tokens == $p.max_tokens)) as $q |
  [ $p.max_tokens,
    $p.tps,
    $q.tps,
    (if $q.tps > 0 then (($p.tps - $q.tps) / $q.tps * 100) else 0 end),
    $p.latency_p95_ms
  ] | @tsv
' | while IFS=$'\t' read -r maxt etps mtps delta ep95; do
  printf '%-10s %14.2f %14.2f %+11.1f %14.1f\n' "$maxt" "$etps" "$mtps" "$delta" "$ep95"
done

echo
ok "engine-mlx JSON: $ENGINE_JSON"
ok "mlx_lm JSON:    $MLXLM_JSON"
log "(positive delta % = engine-mlx faster than mlx_lm)"
