#!/usr/bin/env bash
# Generate synthetic per-clinic CKD datasets (one CSV per clinic = a simulated extract_features.sql
# output) into data/clinics/. Default: 10 clinics. Override e.g. ./scripts/clinics.sh --clinics 25
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run ckd-clinics "$@"
