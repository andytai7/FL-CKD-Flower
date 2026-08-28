#!/usr/bin/env bash
# Centralized (pooled-data) ceiling baseline: logistic regression on the flat synthetic CSV.
# Override -> ./scripts/baseline.sh --clinics   # pool data/clinics/ — the ceiling that is
#                                               # directly comparable to `ckd-simulate --clinics`
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run ckd-baseline "$@"
