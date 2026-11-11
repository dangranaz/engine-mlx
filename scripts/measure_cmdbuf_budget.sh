#!/usr/bin/env bash
#
# measure_cmdbuf_budget.sh — EXPERIMENT (not a product feature).
#
# Measures the effect of MLX's per-command-buffer budget on engine-mlx, by
# starting the server under different MLX_MAX_OPS_PER_BUFFER / MLX_MAX_MB_PER_BUFFER
# values and recording, for each:
#   - decode throughput (t/s) via scripts/bench.sh
#   - peak server RSS (MB) sampled while a long prompt is prefilled
#
# Rationale (lever 1 of the throughput study): MLX latches the command-buffer
# budget ONCE at Device construction from the env vars, so a per-phase RAII
# switch is not possible without patching MLX. The only MLX-C-reachable form is
# to set the env vars before the process touches the GPU. A high budget can
# speed decode but raises the prefill memory peak — this script quantifies BOTH
# on THIS machine (M1 16 GB) so we decide whether lever 1 is worth landing.
#
# This script changes no product code. It only sets env vars for the server it
# starts and reads back public metrics over HTTP + from `ps`.
#
# Usage:
#   scripts/measure_cmdbuf_budget.sh [--model NAME] [--release] \
#       [--mb "0 100 500 1000"] [--ops 1000] [--decode-tokens 256] \
#       [--prefill-prompt-tokens 2048] [--repeat 5]
#
#   --model   model spec for server.sh (default Qwen3-1.7B-MLX-4bit)
#   --release build engine-mlx in release mode (recommended for real numbers)
#   --mb      space-separated MLX_MAX_MB_PER_BUFFER values to test.
#             "0" means BASELINE: no env var set (MLX device default).
#   --ops     MLX_MAX_OPS_PER_BUFFER to pair with every non-zero mb value.
#             The input budget only binds once the op cap is raised, so a bare
#             mb with the default op cap often does nothing. Default 1000.
#   --decode-tokens         max_tokens for the decode throughput run (default 256)
#   --prefill-prompt-tokens approx prompt length (in words) for the peak-RSS
#                           probe (default 2048)
#   --repeat  bench reps per point (default 5)
#
# Output: a results table on stdout and a JSON summary under scripts/.run/.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/lib/common.sh"

MODEL="Qwen3-1.7B-MLX-4bit"
RELEASE=""
MB_VALUES="0 100 500 1000"
OPS="1000"
DECODE_TOKENS="256"
PREFILL_PROMPT_TOKENS="2048"
REPEAT="5"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --release) RELEASE="--release"; shift ;;
    --mb) MB_VALUES="$2"; shift 2 ;;
    --ops) OPS="$2"; shift 2 ;;
    --decode-tokens) DECODE_TOKENS="$2"; shift 2 ;;
    --prefill-prompt-tokens) PREFILL_PROMPT_TOKENS="$2"; shift 2 ;;
    --repeat) REPEAT="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

HOST="${ENGINE_MLX_HOST:-127.0.0.1}"
PORT="${ENGINE_MLX_PORT:-11435}"
BASE="http://${HOST}:${PORT}"
RUN_DIR="${HERE}/.run"
mkdir -p "$RUN_DIR"
OUT_JSON="${RUN_DIR}/cmdbuf_budget_results.json"

# Build a long prompt (~N words) for the prefill peak-RSS probe.
make_long_prompt() {
  local n="$1"
  local word="token "
  local out=""
  local i=0
  while [[ $i -lt $n ]]; do out+="$word"; i=$((i+1)); done
  printf 'Summarize the following text in one word: %s' "$out"
}

# Sample the RSS (KB) of the running server process every 50ms into a file,
# until told to stop. Returns peak MB on stdout.
PID_OF_SERVER=""
find_server_pid() {
  # server.sh records its pid under RUN_DIR via pid_file "engine-mlx".
  local pf; pf="$(pid_file "engine-mlx")"
  if [[ -f "$pf" ]]; then cat "$pf"; return; fi
  pgrep -f "engine-mlx-serve" | head -1 || true
}

peak_rss_mb_during() {
  # Runs the command passed as args while sampling RSS of the server pid.
  local pid="$1"; shift
  local peak_kb=0
  ( "$@" >/dev/null 2>&1 ) &
  local work_pid=$!
  while kill -0 "$work_pid" 2>/dev/null; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      local rss
      rss="$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ')"
      [[ -n "$rss" && "$rss" -gt "$peak_kb" ]] && peak_kb="$rss"
    fi
    sleep 0.05
  done
  wait "$work_pid" 2>/dev/null || true
  echo $(( peak_kb / 1024 ))
}

prefill_probe() {
  # One request with a long prompt and max_tokens=1 — dominated by prefill.
  local prompt; prompt="$(make_long_prompt "$PREFILL_PROMPT_TOKENS")"
  curl -s "${BASE}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(jq -n --arg m "$MODEL" --arg p "$prompt" \
          '{model:$m, messages:[{role:"user",content:$p}], max_tokens:1, temperature:0}')" \
    >/dev/null 2>&1 || true
}

results=()
echo "# command-buffer budget experiment — model=${MODEL} ops=${OPS} host=${BASE}"
printf '%-18s %-14s %-16s\n' "condition" "decode t/s" "prefill peak MB"
printf '%-18s %-14s %-16s\n' "---------" "----------" "---------------"

for mb in $MB_VALUES; do
  # Stop any running server first.
  "${HERE}/server.sh" stop >/dev/null 2>&1 || true
  sleep 1

  if [[ "$mb" == "0" ]]; then
    label="baseline"
    unset MLX_MAX_OPS_PER_BUFFER MLX_MAX_MB_PER_BUFFER || true
    MLX_MAX_OPS_PER_BUFFER="" MLX_MAX_MB_PER_BUFFER="" \
      "${HERE}/server.sh" start "$MODEL" ${RELEASE} >/dev/null 2>&1
  else
    label="ops${OPS}_mb${mb}"
    MLX_MAX_OPS_PER_BUFFER="$OPS" MLX_MAX_MB_PER_BUFFER="$mb" \
      "${HERE}/server.sh" start "$MODEL" ${RELEASE} >/dev/null 2>&1
  fi

  # Wait for readiness.
  if ! wait_for_health "$BASE" "${READY_TIMEOUT:-300}"; then
    warn "server not ready for condition ${label}; skipping"
    continue
  fi
  PID_OF_SERVER="$(find_server_pid)"

  # Warmup: the first request after load pays graph-compile/caching cost and
  # would skew the throughput of a cold model. Discard one request before
  # measuring.
  curl -s "${BASE}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(jq -n --arg m "$MODEL" '{model:$m, messages:[{role:"user",content:"warmup"}], max_tokens:32, temperature:0}')" \
    >/dev/null 2>&1 || true

  # Decode throughput. bench.sh writes {points:[{max_tokens, tps, ...}]}.
  tps_json="${RUN_DIR}/cmdbuf_${label}.json"
  "${HERE}/bench.sh" --url "$BASE" --model "$MODEL" \
    --max-tokens "$DECODE_TOKENS" --repeat "$REPEAT" \
    --json "$tps_json" --label "$label" >/dev/null 2>&1 || true
  tps="$(jq -r '(.points[0].tps // .tps // empty)' "$tps_json" 2>/dev/null || echo "")"
  [[ -z "$tps" ]] && tps="n/a"

  # Prefill peak RSS.
  peak="$(peak_rss_mb_during "$PID_OF_SERVER" prefill_probe)"

  printf '%-18s %-14s %-16s\n' "$label" "$tps" "$peak"
  results+=("$(jq -n --arg c "$label" --arg mb "$mb" --arg ops "$OPS" \
      --arg tps "$tps" --arg peak "$peak" \
      '{condition:$c, mb_per_buffer:$mb, ops_per_buffer:$ops, decode_tps:$tps, prefill_peak_mb:($peak|tonumber)}')")
done

"${HERE}/server.sh" stop >/dev/null 2>&1 || true

printf '%s\n' "${results[@]}" | jq -s '{model:"'"$MODEL"'", ops_per_buffer:"'"$OPS"'", points:.}' > "$OUT_JSON"
echo
echo "JSON summary: $OUT_JSON"
echo "Baseline is 'baseline'; a useful lever shows higher decode t/s WITHOUT a"
echo "large jump in prefill peak MB. On 16 GB, watch the peak closely."
