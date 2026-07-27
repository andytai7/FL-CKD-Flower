#!/usr/bin/env bash
# Remove generated caches and macOS .DS_Store junk.
set -euo pipefail
cd "$(dirname "$0")/.."
find . -type d -name __pycache__ -prune -exec rm -rf {} +
find . -name '.DS_Store' -delete
rm -rf .ruff_cache .pytest_cache
echo "Cleaned caches, .DS_Store, .ruff_cache, .pytest_cache."
