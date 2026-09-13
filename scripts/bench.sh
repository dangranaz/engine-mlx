#!/usr/bin/env bash
#
# bench.sh — HTTP benchmark client for any OpenAI-compatible /v1/chat/completions
# endpoint (engine-mlx, mlx_lm.server, or anything else that speaks the protocol).
#
# Model-agnostic: the target URL and model name are parameters. Nothing here
# assumes Qwen3.
#
# Metrics:
#   - throughput (t/s)     : completion_tokens / total_latency (from `usage`)
#   - TTFT (ms)            : time to first SSE chunk (streaming mode, best-effort)
#   - latency p50/p95 (ms) : per-request end-to-end wall time
#   - RPS                  : completed requests / wall time (under concurrency)
#
# ── TTFT caveat ──────────────────────────────────────────────────────────────
# engine-mlx currently generates the FULL completion, then fake-streams it word
# by word over SSE. So its "TTFT" reflects full generation time, not a true
# first-token latency. mlx_lm streams for real. TTFT is comparable ONLY between
# engines that both stream token-by-token. This is surfaced in the output.
#
# Usage:
#   scripts/bench.sh [options]
#
# Options:
#   -u, --url URL          Base URL (default http://127.0.0.1:11435)
#   -m, --model NAME       Model name sent in the request body (default "default")
#   -p, --prompt TEXT      Prompt text (default a fixed English sentence)
#   -n, --max-tokens N     max_tokens per request (default 128)
#   -r, --repeat N         Repetitions per matrix point (default 5)
#   -c, --concurrency N    Concurrent requests for RPS test (default 1)
#   -t, --temperature F    Sampling temperature (default 0)
#   --matrix "A B C"       Space-separated max_tokens points (overrides -n; runs a matrix)
#   --stream               Use SSE streaming to measure real TTFT
#   --json FILE            Write machine-readable results to FILE (JSON)
#   --label NAME           Label for the run (e.g. "engine-mlx"); default derived from URL
#   -h, --help             Show this help

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

# ── Defaults ──────────────────────────────────────────────────────────────────
URL="http://127.0.0.1:11435"
MODEL="default"
PROMPT="Explain the theory of relativity in simple terms."
MAX_TOKENS=128
REPEAT=5
CONCURRENCY=1
TEMPERATURE=0
MATRIX=""
STREAM=0
JSON_OUT=""
LABEL=""

usage() { sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; }

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    -u|--url)          URL="$2"; shift 2 ;;
    -m|--model)        MODEL="$2"; shift 2 ;;
    -p|--prompt)       PROMPT="$2"; shift 2 ;;
    -n|--max-tokens)   MAX_TOKENS="$2"; shift 2 ;;
    -r|--repeat)       REPEAT="$2"; shift 2 ;;
    -c|--concurrency)  CONCURRENCY="$2"; shift 2 ;;
    -t|--temperature)  TEMPERATURE="$2"; shift 2 ;;
    --matrix)          MATRIX="$2"; shift 2 ;;
    --stream)          STREAM=1; shift ;;
    --json)            JSON_OUT="$2"; shift 2 ;;
    --label)           LABEL="$2"; shift 2 ;;
    -h|--help)         usage; exit 0 ;;
    *) err "unknown option: $1"; usage; exit 1 ;;
  esac
done

require_cmd curl jq

[[ -z "$LABEL" ]] && LABEL="$(printf '%s' "$URL" | sed -E 's#https?://##; s#[/:]#_#g')"
CHAT_URL="$URL/v1/chat/completions"

# ── Request body builder ──────────────────────────────────────────────────────
# build_body MAX_TOKENS STREAM_BOOL -> JSON string
build_body() {
  local maxt="$1" stream="$2"
  jq -n \
    --arg model "$MODEL" \
    --arg prompt "$PROMPT" \
    --argjson maxt "$maxt" \
    --argjson temp "$TEMPERATURE" \
    --argjson stream "$stream" \
    '{
      model: $model,
      messages: [{role: "user", content: $prompt}],
      max_tokens: $maxt,
      temperature: $temp,
      stream: ($stream == 1)
    }'
}

# ── Single non-streaming request ──────────────────────────────────────────────
# Prints: "<latency_ms> <completion_tokens> <prompt_tokens>" or "ERR" on failure.
request_once() {
  local maxt="$1"
  local body; body="$(build_body "$maxt" 0)"
  local tmp; tmp="$(mktemp)"
  local t0 t1 latency_ms
  t0="$(date +%s.%N)"
  local http_code
  http_code="$(curl -s -o "$tmp" -w '%{http_code}' \
    --max-time 600 \
    -H 'Content-Type: application/json' \
    -X POST "$CHAT_URL" \
    -d "$body" 2>/dev/null || echo "000")"
  t1="$(date +%s.%N)"
  latency_ms="$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.1f", (b-a)*1000}')"

  if [[ "$http_code" != "200" ]]; then
    warn "request failed (HTTP $http_code): $(head -c 200 "$tmp")"
    rm -f "$tmp"
    echo "ERR"
    return 0
  fi

  # Parse OpenAI usage. Fall back to counting words if usage absent.
  local ctok ptok
  ctok="$(jq -r '.usage.completion_tokens // empty' "$tmp" 2>/dev/null || echo "")"
  ptok="$(jq -r '.usage.prompt_tokens // empty' "$tmp" 2>/dev/null || echo "")"
  if [[ -z "$ctok" ]]; then
    ctok="$(jq -r '.choices[0].message.content // ""' "$tmp" 2>/dev/null | wc -w | tr -d ' ')"
  fi
  [[ -z "$ptok" ]] && ptok=0
  rm -f "$tmp"
  echo "$latency_ms $ctok $ptok"
}

# ── Single streaming request (measures TTFT) ──────────────────────────────────
# Prints: "<latency_ms> <ttft_ms> <chunks>" or "ERR".
request_stream_once() {
  local maxt="$1"
  local body; body="$(build_body "$maxt" 1)"
  local t0 t1 tfirst latency_ms ttft_ms chunks=0 got_first=0
  t0="$(date +%s.%N)"
  tfirst=""
  # Read SSE line by line; the first `data:` line marks first token arrival.
  while IFS= read -r line; do
    if [[ "$line" == data:* ]]; then
      if [[ "$got_first" -eq 0 ]]; then
        tfirst="$(date +%s.%N)"
        got_first=1
      fi
      [[ "$line" == "data: [DONE]" ]] || chunks=$(( chunks + 1 ))
    fi
  done < <(curl -s -N \
      --max-time 600 \
      -H 'Content-Type: application/json' \
      -H 'Accept: text/event-stream' \
      -X POST "$CHAT_URL" \
      -d "$body" 2>/dev/null)
  t1="$(date +%s.%N)"

  if [[ "$got_first" -eq 0 ]]; then
    echo "ERR"
    return 0
  fi
  latency_ms="$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.1f", (b-a)*1000}')"
  ttft_ms="$(awk -v a="$t0" -v b="$tfirst" 'BEGIN{printf "%.1f", (b-a)*1000}')"
  echo "$latency_ms $ttft_ms $chunks"
}

# ── Run one matrix point (REPEAT sequential requests) ─────────────────────────
# Populates arrays and prints one table row. Also appends to JSON accumulator.
JSON_POINTS=()
run_point() {
  local maxt="$1"
  local latencies=() ctoks=() tps_list=() ttfts=()

  info "point max_tokens=$maxt  repeat=$REPEAT  stream=$STREAM"
  local i
  for (( i = 0; i < REPEAT; i++ )); do
    if [[ "$STREAM" -eq 1 ]]; then
      local out; out="$(request_stream_once "$maxt")"
      [[ "$out" == "ERR" ]] && { warn "rep $i failed"; continue; }
      local lat ttft chunks
      read -r lat ttft chunks <<<"$out"
      latencies+=("$lat")
      ttfts+=("$ttft")
      # tokens ~ chunks (fake-stream = words). t/s from chunks over latency.
      local tps; tps="$(awk -v c="$chunks" -v l="$lat" 'BEGIN{ if(l>0) printf "%.2f", c/(l/1000.0); else print 0 }')"
      tps_list+=("$tps")
      ctoks+=("$chunks")
      log "  rep $i: latency=${lat}ms ttft=${ttft}ms chunks=$chunks tps=${tps}"
    else
      local out; out="$(request_once "$maxt")"
      [[ "$out" == "ERR" ]] && { warn "rep $i failed"; continue; }
      local lat ctok ptok
      read -r lat ctok ptok <<<"$out"
      latencies+=("$lat")
      ctoks+=("$ctok")
      local tps; tps="$(awk -v c="$ctok" -v l="$lat" 'BEGIN{ if(l>0) printf "%.2f", c/(l/1000.0); else print 0 }')"
      tps_list+=("$tps")
      log "  rep $i: latency=${lat}ms completion_tokens=$ctok tps=${tps}"
    fi
  done

  if [[ "${#latencies[@]}" -eq 0 ]]; then
    err "all reps failed for max_tokens=$maxt"
    return 1
  fi

  local p50 p95 mean_tps mean_lat mean_ttft=""
  p50="$(printf '%s\n' "${latencies[@]}" | percentile 50)"
  p95="$(printf '%s\n' "${latencies[@]}" | percentile 95)"
  mean_lat="$(printf '%s\n' "${latencies[@]}" | mean)"
  mean_tps="$(printf '%s\n' "${tps_list[@]}" | mean)"
  if [[ "$STREAM" -eq 1 && "${#ttfts[@]}" -gt 0 ]]; then
    mean_ttft="$(printf '%s\n' "${ttfts[@]}" | mean)"
  fi
  local mean_ctok; mean_ctok="$(printf '%s\n' "${ctoks[@]}" | mean)"

  # Table row
  printf '%s%-8s%s | tps %7.2f | lat p50 %8.1f p95 %8.1f mean %8.1f | ttft %8s | ctok %6.0f\n' \
    "$BOLD" "$maxt" "$RESET" \
    "$mean_tps" "$p50" "$p95" "$mean_lat" "${mean_ttft:-—}" "$mean_ctok"

  # JSON point
  JSON_POINTS+=("$(jq -n \
    --argjson maxt "$maxt" \
    --argjson tps "$mean_tps" \
    --argjson p50 "$p50" \
    --argjson p95 "$p95" \
    --argjson mean_lat "$mean_lat" \
    --arg ttft "${mean_ttft:-null}" \
    --argjson ctok "$mean_ctok" \
    --argjson reps "${#latencies[@]}" \
    '{max_tokens: $maxt, tps: $tps, latency_p50_ms: $p50, latency_p95_ms: $p95,
      latency_mean_ms: $mean_lat, ttft_ms: (if $ttft=="null" then null else ($ttft|tonumber) end),
      completion_tokens_mean: $ctok, successful_reps: $reps}')")
}

# ── Concurrency / RPS test ────────────────────────────────────────────────────
run_concurrency() {
  local maxt="$1" conc="$2"
  info "concurrency test: $conc parallel requests, max_tokens=$maxt"
  local tmpdir; tmpdir="$(mktemp -d)"
  local t0 t1 wall
  t0="$(date +%s.%N)"
  local i
  for (( i = 0; i < conc; i++ )); do
    ( request_once "$maxt" >"$tmpdir/$i" ) &
  done
  wait
  t1="$(date +%s.%N)"
  wall="$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f", (b-a)}')"

  local ok_count=0 total_ctok=0
  local latencies=()
  for (( i = 0; i < conc; i++ )); do
    local out; out="$(cat "$tmpdir/$i" 2>/dev/null || echo ERR)"
    [[ "$out" == "ERR" || -z "$out" ]] && continue
    local lat ctok ptok
    read -r lat ctok ptok <<<"$out"
    latencies+=("$lat")
    ok_count=$(( ok_count + 1 ))
    total_ctok=$(( total_ctok + ctok ))
  done
  rm -rf "$tmpdir"

  if [[ "$ok_count" -eq 0 ]]; then
    err "all concurrent requests failed"
    return 1
  fi
  local rps agg_tps p95
  rps="$(awk -v c="$ok_count" -v w="$wall" 'BEGIN{ if(w>0) printf "%.2f", c/w; else print 0 }')"
  agg_tps="$(awk -v t="$total_ctok" -v w="$wall" 'BEGIN{ if(w>0) printf "%.2f", t/w; else print 0 }')"
  p95="$(printf '%s\n' "${latencies[@]}" | percentile 95)"

  printf '%sconcurrency=%d%s | completed %d/%d in %ss | RPS %6.2f | aggregate %7.2f t/s | lat p95 %8.1fms\n' \
    "$BOLD" "$conc" "$RESET" "$ok_count" "$conc" "$wall" "$rps" "$agg_tps" "$p95"

  CONC_JSON="$(jq -n \
    --argjson conc "$conc" --argjson maxt "$maxt" \
    --argjson ok "$ok_count" --argjson wall "$wall" \
    --argjson rps "$rps" --argjson agg "$agg_tps" --argjson p95 "$p95" \
    '{concurrency: $conc, max_tokens: $maxt, completed: $ok, wall_s: $wall,
      rps: $rps, aggregate_tps: $agg, latency_p95_ms: $p95}')"
}

# ── Main ──────────────────────────────────────────────────────────────────────
main() {
  info "benchmark target: $CHAT_URL"
  log "label=$LABEL model=$MODEL prompt_len=${#PROMPT} temp=$TEMPERATURE stream=$STREAM"

  # Reachability check
  local code; code="$(http_get_code "$URL/v1/models")"
  if [[ "$code" != "200" ]]; then
    warn "$URL/v1/models returned $code — server may not be ready"
  fi

  if [[ "$STREAM" -eq 1 ]]; then
    warn "TTFT caveat: comparable only across true token-streaming engines (see header)"
  fi

  echo
  info "── Throughput / latency ──"
  local points
  if [[ -n "$MATRIX" ]]; then
    points="$MATRIX"
  else
    points="$MAX_TOKENS"
  fi
  local pt
  for pt in $points; do
    run_point "$pt"
  done

  CONC_JSON="null"
  if [[ "$CONCURRENCY" -gt 1 ]]; then
    echo
    info "── Concurrency / RPS ──"
    # Use the first matrix point (or MAX_TOKENS) for the concurrency test.
    local cpt; cpt="$(echo "$points" | awk '{print $1}')"
    run_concurrency "$cpt" "$CONCURRENCY"
  fi

  # JSON output
  if [[ -n "$JSON_OUT" ]]; then
    local points_json
    points_json="$(printf '%s\n' "${JSON_POINTS[@]}" | jq -s '.')"
    jq -n \
      --arg label "$LABEL" \
      --arg url "$CHAT_URL" \
      --arg model "$MODEL" \
      --arg prompt "$PROMPT" \
      --argjson temp "$TEMPERATURE" \
      --argjson stream "$STREAM" \
      --argjson repeat "$REPEAT" \
      --argjson points "$points_json" \
      --argjson concurrency "$CONC_JSON" \
      '{label: $label, url: $url, model: $model, prompt: $prompt,
        temperature: $temp, stream: ($stream==1), repeat: $repeat,
        points: $points, concurrency: $concurrency,
        timestamp: (now | todate)}' >"$JSON_OUT"
    ok "wrote JSON results to $JSON_OUT"
  fi
}

main
