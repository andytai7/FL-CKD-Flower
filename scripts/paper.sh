#!/usr/bin/env bash
# Build the technical expert report's figures, tables and macros from one pinned run.
#   ./scripts/paper.sh              render from the cache (fast)
#   ./scripts/paper.sh --recompute  re-run every experiment (~10 min), then render
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python paper/make_paper.py "$@"
