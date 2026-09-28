#!/usr/bin/env bash
# Stop the engine-mlx server started for Qwen3-1.7B-MLX-8bit.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE=engine-mlx exec "$HERE/../stop.sh"
