#!/usr/bin/env bash
# Start engine-mlx serving K2-Horizon-3.7B-4bit on the standard port 11435.
#
# K2-Horizon is a dense transformer with MIXED per-tensor quantization (qkv
# 8-bit/group-64, gate/up 4-bit/group-32, etc.), grouped RMSNorm (2 groups),
# and a nested rope_theta (1e7, inside rope_parameters). engine-mlx serves it
# via the dense Qwen3 forward: the per-weight quant params and the fusion
# gating (tasks B/C) make it token-exact vs the reference k2_horizon.py, and
# chat requests are rendered through the model's own chat_template.jinja (the
# ChatTemplate module), not a hardcoded template. See the k2_horizon expected
# profile.
#
# Layout note: these benches live INSIDE the engine-mlx repo (benches/), so the
# engine's server.sh is normally two levels up under scripts/. This launcher
# resolves it at (in priority order):
#   1. $ENGINE_MLX_REPO/scripts/server.sh    (explicit override)
#   2. $HERE/../../scripts/server.sh          (benches inside engine-mlx repo)
#
# Weights live under ~/nexum/models by default; override with ENGINE_MLX_MODELS_DIR.
#
# Usage: models/start-k2-horizon-3.7b.sh   (RELEASE=1 is the default for a fair bench)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

find_server_sh() {
  local candidates=(
    "${ENGINE_MLX_REPO:-}/scripts/server.sh"
    "$HERE/../../scripts/server.sh"
  )
  local c
  for c in "${candidates[@]}"; do
    [[ -n "$c" && -x "$c" ]] && { echo "$c"; return 0; }
  done
  return 1
}

MODEL="mlx-community/K2-Horizon-3.7B-4bit"
PORT="${PORT:-11435}"
RELEASE="${RELEASE:-1}"
export ENGINE_MLX_MODELS_DIR="${ENGINE_MLX_MODELS_DIR:-$HOME/nexum/models}"

if SERVER_SH="$(find_server_sh)"; then
  rel_flag=()
  [[ "$RELEASE" == "1" ]] && rel_flag=(--release)
  echo "starting engine-mlx (K2-Horizon-3.7B) via $SERVER_SH"
  ENGINE=engine-mlx MODEL="$MODEL" PORT="$PORT" RELEASE="$RELEASE" \
  BENCH_START_CMD="ENGINE_MLX_MODELS_DIR='$ENGINE_MLX_MODELS_DIR' exec '$SERVER_SH' start '$MODEL' ${rel_flag[*]}" \
    exec "$HERE/../start.sh"
else
  echo "ERROR: could not locate engine-mlx server.sh." >&2
  echo "Set ENGINE_MLX_REPO=/path/to/engine-mlx and retry." >&2
  exit 1
fi
