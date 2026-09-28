#!/usr/bin/env bash
#
# run.sh — full benchmark run: declarative Hurl gates + nxm-bench measurements.
#
# Assumes the server is already up (use ./start.sh or a per-model script in
# models/). Runs, in order:
#   1. Hurl declarative tests (api / coherence / perf) — fast pass/fail gates.
#      The coherence gate is BLOCKING per nexum RULES (corrupted output -> fail).
#   2. nxm-bench (Rust) — throughput, sustained-degradation, KV-ramp, with
#      timestamped JSON+MD reports.
#
# Exit code: 0 only if BOTH stages pass. Hurl missing -> skipped with a warning
# (nxm-bench still runs), so the run degrades gracefully.
#
# Config (env / flags):
#   URL     base URL of the server        (default http://127.0.0.1:11435)
#   ENGINE  engine label for reports      (default engine-mlx)
#   MODEL   model id in requests          (default: resolved from /v1/models)
#   ASSERT  "1" to pass --assert to nxm-bench (pass/fail + exit code)  [default 1]
#
# Usage:
#   ./run.sh
#   URL=http://127.0.0.1:11435 ENGINE=engine-mlx ./run.sh
#   ASSERT=0 ./run.sh          # measure-only nxm-bench (no pass/fail)

set -uo pipefail
shopt -s globstar nullglob

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
URL="${URL:-http://127.0.0.1:11435}"
ENGINE="${ENGINE:-engine-mlx}"
ASSERT="${ASSERT:-1}"

RESULTS_DIR="$HERE/results/$ENGINE"
STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
HURL_REPORT_DIR="$RESULTS_DIR/hurl_$STAMP"

bold() { printf '\033[1m%s\033[0m\n' "$1"; }
warn() { printf '\033[33m⚠ %s\033[0m\n' "$1" >&2; }
ok()   { printf '\033[32m✓ %s\033[0m\n' "$1"; }
err()  { printf '\033[31m✗ %s\033[0m\n' "$1" >&2; }

# ── Resolve model id from /v1/models if not provided ──
resolve_model() {
  [[ -n "${MODEL:-}" ]] && { echo "$MODEL"; return; }
  local id
  id="$(curl -fsS "$URL/v1/models" 2>/dev/null \
        | grep -o '"id"[[:space:]]*:[[:space:]]*"[^"]*"' \
        | head -1 | sed -E 's/.*"id"[[:space:]]*:[[:space:]]*"([^"]*)".*/\1/')"
  echo "${id:-default}"
}
MODEL="$(resolve_model)"

bold "engine-bench run"
echo "  url    = $URL"
echo "  engine = $ENGINE"
echo "  model  = $MODEL"
echo "  assert = $ASSERT"
echo ""

hurl_rc=0
bench_rc=0

# ── Stage 1: Hurl declarative gates ──
bold "── Stage 1: Hurl declarative tests ──"
if command -v hurl >/dev/null 2>&1; then
  mkdir -p "$HURL_REPORT_DIR"
  # Run the whole hurl/ tree; --test gives pass/fail + non-zero exit on failure.
  # JSON + HTML reports are written for historicity alongside nxm-bench results.
  if hurl --test \
        --variable "host=$URL" \
        --variable "model=$MODEL" \
        --report-html "$HURL_REPORT_DIR/html" \
        --report-json "$HURL_REPORT_DIR/json" \
        "$HERE"/hurl/**/*.hurl; then
    ok "Hurl tests passed"
  else
    hurl_rc=$?
    err "Hurl tests FAILED (rc=$hurl_rc) — see $HURL_REPORT_DIR"
  fi
  echo "  hurl report: $HURL_REPORT_DIR"
else
  warn "hurl not installed (brew install hurl) — skipping declarative gates"
fi
echo ""

# ── Stage 2: nxm-bench measurements ──
bold "── Stage 2: nxm-bench (throughput / degradation / ramp) ──"
BIN="$HERE/target/release/nxm-bench"
if [[ ! -x "$BIN" ]]; then
  warn "nxm-bench not built — building release"
  ( cd "$HERE" && cargo build --release ) || { err "build failed"; exit 2; }
fi

bench_args=(run --url "$URL" --engine "$ENGINE" --model "$MODEL")
[[ "$ASSERT" == "1" ]] && bench_args+=(--assert --engine-repo "$HERE/../engine-mlx")

if "$BIN" "${bench_args[@]}"; then
  ok "nxm-bench passed"
else
  bench_rc=$?
  err "nxm-bench FAILED (rc=$bench_rc)"
fi
echo ""

# ── Aggregate ──
bold "── Summary ──"
echo "  Hurl:      $([[ $hurl_rc -eq 0 ]] && echo PASS || echo FAIL) (rc=$hurl_rc)"
echo "  nxm-bench: $([[ $bench_rc -eq 0 ]] && echo PASS || echo FAIL) (rc=$bench_rc)"

if [[ $hurl_rc -eq 0 && $bench_rc -eq 0 ]]; then
  ok "ALL PASSED"
  exit 0
else
  err "FAILURES DETECTED"
  exit 1
fi
