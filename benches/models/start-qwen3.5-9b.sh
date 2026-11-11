#!/usr/bin/env bash
# Start engine-mlx serving Qwen3.5-9B-MLX-4bit (hybrid GDN + gated full-attn)
# on the standard port 11435.
#
# Qwen3.5 is a HYBRID model: 3 of every 4 layers are Gated DeltaNet (linear
# attention), the rest are gated full attention with partial RoPE. engine-mlx
# runs it via a dedicated eager hybrid forward — token-exact vs mlx_lm, but not
# yet compiled, so throughput is lower than the dense Qwen3 path (see the
# qwen3_5 expected profile).
#
# Layout note: these benches live INSIDE the engine-mlx repo (benches/), so the
# engine's server.sh is normally two levels up under scripts/. This launcher
# resolves it at (in priority order):
#   1. $ENGINE_MLX_REPO/scripts/server.sh    (explicit override)
#   2. $HERE/../../scripts/server.sh          (benches inside engine-mlx repo)
#
# Weights live under ~/models by default; override with ENGINE_MLX_MODELS_DIR.
#
# Usage: models/start-qwen3.5-9b.sh   (RELEASE=1 is the default for a fair bench)
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

MODEL="mlx-community/Qwen3.5-9B-MLX-4bit"
PORT="${PORT:-11435}"
RELEASE="${RELEASE:-1}"
export ENGINE_MLX_MODELS_DIR="${ENGINE_MLX_MODELS_DIR:-$HOME/models}"

if SERVER_SH="$(find_server_sh)"; then
  rel_flag=()
  [[ "$RELEASE" == "1" ]] && rel_flag=(--release)
  echo "starting engine-mlx (Qwen3.5-9B) via $SERVER_SH"
  ENGINE=engine-mlx MODEL="$MODEL" PORT="$PORT" RELEASE="$RELEASE" \
  BENCH_START_CMD="ENGINE_MLX_MODELS_DIR='$ENGINE_MLX_MODELS_DIR' exec '$SERVER_SH' start '$MODEL' ${rel_flag[*]}" \
    exec "$HERE/../start.sh"
else
  echo "ERROR: could not locate engine-mlx server.sh." >&2
  echo "Set ENGINE_MLX_REPO=/path/to/engine-mlx and retry." >&2
  exit 1
fi
