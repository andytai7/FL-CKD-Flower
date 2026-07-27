#!/usr/bin/env bash
# Centralized (pooled-data) ceiling baselines.
# No args  -> runs all three models (logreg + mlp + gbt).
# Override -> ./scripts/baseline.sh --model gbt
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "$#" -eq 0 ]; then set -- --model all; fi
exec uv run ckd-baseline "$@"
