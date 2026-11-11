#!/usr/bin/env bash
#
# check_expected.sh — automated PASS/FAIL verdict against an expected profile.
#
# Compares an nxm-bench result JSON to an expected/<profile>.json and decides:
#   - all `must_pass` tests present in the result actually passed
#   - sustained min t/s >= profile.min_tps  (engine-health throughput floor)
# Advisory tests (model capability) are reported but never fail the verdict.
#
# Hurl tests (api_*, perf_*, coherence_hurl) are NOT in the nxm-bench JSON; if a
# must_pass entry refers to them, this script reports them as "verify via run.sh
# exit code" (run.sh already gates on Hurl). This script judges the nxm-bench
# layer deterministically.
#
# Usage:
#   ./check_expected.sh <result.json> <profile>
#   ./check_expected.sh results/engine-mlx/2026-09-09_18-05-59.json tiny
#   ./check_expected.sh                 # latest result, profile "tiny"
#
# Exit: 0 = healthy, 1 = unhealthy, 2 = usage/tooling error.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

command -v jq >/dev/null 2>&1 || { echo "error: jq required" >&2; exit 2; }

RESULT="${1:-}"
PROFILE="${2:-tiny}"

if [[ -z "$RESULT" ]]; then
  RESULT="$(ls -t "$HERE"/results/*/*.json 2>/dev/null | head -1)"
  [[ -z "$RESULT" ]] && { echo "error: no result JSON found; pass one explicitly" >&2; exit 2; }
fi
PROFILE_FILE="$HERE/expected/${PROFILE}.json"
[[ -f "$RESULT" ]]       || { echo "error: result not found: $RESULT" >&2; exit 2; }
[[ -f "$PROFILE_FILE" ]] || { echo "error: profile not found: $PROFILE_FILE (choose tiny|small|large)" >&2; exit 2; }

bold() { printf '\033[1m%s\033[0m\n' "$1"; }
ok()   { printf '\033[32m✓ %s\033[0m\n' "$1"; }
bad()  { printf '\033[31m✗ %s\033[0m\n' "$1"; }
note() { printf '  %s\n' "$1"; }

MODEL="$(jq -r '.environment.model // "?"' "$RESULT")"
MIN_TPS_REQ="$(jq -r '.min_tps' "$PROFILE_FILE")"

bold "check_expected — profile: $PROFILE"
note "result : $RESULT"
note "model  : $MODEL"
note "min_tps floor: $MIN_TPS_REQ"
echo ""

RESULT_IDS=()
while IFS= read -r _rid; do RESULT_IDS+=("$_rid"); done < <(jq -r '.tests[].id' "$RESULT")
is_present() { local id="$1"; local x; for x in ${RESULT_IDS[@]+"${RESULT_IDS[@]}"}; do [[ "$x" == "$id" ]] && return 0; done; return 1; }
test_passed() { jq -e --arg id "$1" '.tests[] | select(.id==$id) | .passed==true' "$RESULT" >/dev/null 2>&1; }

verdict=0

bold "must_pass:"
while IFS= read -r id; do
  case "$id" in
    api_*|perf_*|coherence_hurl)
      note "• $id — Hurl layer: verify via run.sh exit code (not in nxm-bench JSON)"
      ;;
    *)
      if ! is_present "$id"; then
        bad "$id — NOT FOUND in result (was this test run?)"; verdict=1
      elif test_passed "$id"; then
        ok "$id — passed"
      else
        bad "$id — FAILED (required by profile '$PROFILE')"; verdict=1
      fi
      ;;
  esac
done < <(jq -r '.must_pass[]' "$PROFILE_FILE")
echo ""

SUS_MIN="$(jq -r '.tests[] | select(.id=="sustained") | (.metrics[] | select(.[0]=="min_tps") | .[1])' "$RESULT" 2>/dev/null)"
bold "throughput floor:"
if [[ -z "$SUS_MIN" || "$SUS_MIN" == "null" ]]; then
  note "sustained.min_tps not available — skipped"
else
  if awk "BEGIN{exit !($SUS_MIN >= $MIN_TPS_REQ)}"; then
    ok "sustained min_tps ${SUS_MIN} >= ${MIN_TPS_REQ}"
  else
    bad "sustained min_tps ${SUS_MIN} < ${MIN_TPS_REQ} (below floor)"; verdict=1
  fi
fi
echo ""

bold "advisory (model capability — informational only):"
while IFS= read -r id; do
  case "$id" in
    api_*|perf_*|coherence_hurl) continue ;;
  esac
  if is_present "$id"; then
    if test_passed "$id"; then note "• $id: pass"; else note "• $id: fail (allowed for '$PROFILE')"; fi
  fi
done < <(jq -r '.advisory[]' "$PROFILE_FILE")
echo ""

bold "── Summary table (test → question → answer → verdict) ──"
# One row per nxm-bench test: id, category, the question (prompt) asked, a
# short slice of the model's answer, and the pass/fail. Reads the enriched
# report fields (.prompt, .outputs). Falls back to "—" when absent (old JSON).
# Advisory tests (per the profile) render as "FAIL (advisory)" so a red row is
# not mistaken for an engine-health failure — the profile verdict is authority.
ADVISORY_CSV="$(jq -r '.advisory | join(",")' "$PROFILE_FILE")"
jq -r --arg advisory "$ADVISORY_CSV" '
  ($advisory | split(",")) as $adv
  | def trunc(n): if (.|length) > n then (.[0:n] + "…") else . end;
  .tests[]
  | [ .id,
      (.category // "-"),
      ((.prompt // "") | gsub("[\n\r]+"; " ") | trunc(60)),
      (((.outputs // []) | join(" ")) | gsub("[\n\r]+"; " ") | trunc(60)),
      (if .passed==true then "PASS"
       elif .passed==false then (if (.id as $i | $adv | index($i)) then "FAIL (advisory)" else "FAIL" end)
       else "—" end)
    ]
  | @tsv
' "$RESULT" | while IFS=$'\t' read -r id cat q a v; do
  case "$v" in
    PASS)             color='\033[32m' ;;
    "FAIL (advisory)") color='\033[33m' ;;
    FAIL)             color='\033[31m' ;;
    *)                color='\033[0m'  ;;
  esac
  printf '  \033[1m%-12s\033[0m [%-11s] '"$color"'%s\033[0m\n' "$id" "$cat" "$v"
  printf '      Q: %s\n' "$q"
  printf '      A: %s\n' "$a"
done
echo ""

bold "── Verdict ──"
if [[ $verdict -eq 0 ]]; then
  ok "HEALTHY for profile '$PROFILE' (engine-health nxm-bench tests passed)"
  echo "  Reminder: also confirm the Hurl layer via run.sh exit code."
  exit 0
else
  bad "UNHEALTHY for profile '$PROFILE' (a required test failed or throughput below floor)"
  exit 1
fi
