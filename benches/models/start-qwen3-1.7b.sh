#!/usr/bin/env bash
# Start engine-mlx serving Qwen3-1.7B-MLX-8bit on nexum's standard port 11435.
# Wraps the generic ../start.sh with the model name baked in.
#
# Usage: models/start-qwen3-1.7b.sh   (add RELEASE=1 for an optimized build)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE=engine-mlx \
MODEL="Qwen3-1.7B-MLX-8bit" \
PORT="${PORT:-11435}" \
RELEASE="${RELEASE:-1}" \
  exec "$HERE/../start.sh"
