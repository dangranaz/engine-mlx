#!/usr/bin/env bash
# Start engine-mlx serving Qwen3-0.6B-MLX-4bit on nexum's standard port 11435.
# This wraps the generic ../start.sh with the model name baked in, so each model
# has its own one-command launcher.
#
# Usage: models/start-qwen3-0.6b.sh   (add RELEASE=1 for an optimized build)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE=engine-mlx \
MODEL="Qwen3-0.6B-MLX-4bit" \
PORT="${PORT:-11435}" \
RELEASE="${RELEASE:-1}" \
  exec "$HERE/../start.sh"
