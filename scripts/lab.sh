#!/usr/bin/env bash
# Open the exploration notebook interactively in JupyterLab.
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run jupyter lab notebooks/01_explore_and_baselines.ipynb
