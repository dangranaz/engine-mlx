#!/usr/bin/env bash
# Stop the engine-mlx server started for K2-Horizon-3.7B-4bit.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE=engine-mlx exec "$HERE/../stop.sh"
