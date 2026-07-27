#!/usr/bin/env bash
# Re-run the exploration notebook headless (verifies it still works end-to-end).
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=300 notebooks/01_explore_and_baselines.ipynb
