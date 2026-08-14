#!/usr/bin/env bash
# Centralized (pooled-data) ceiling baselines.
# No args  -> runs all three models (logreg + mlp + xgboost) on the flat synthetic CSV.
# Override -> ./scripts/baseline.sh --model xgboost
#             ./scripts/baseline.sh --clinics   # pool data/clinics/ — the ceiling that is
#                                               # directly comparable to `ckd-simulate --clinics`
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "$#" -eq 0 ]; then set -- --model all; fi
exec uv run ckd-baseline "$@"
