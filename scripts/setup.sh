#!/usr/bin/env bash
# Create/repair the venv and install all dependencies (dev + notebook extras).
# Run this once after cloning or after moving the project folder.
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv sync --extra dev --extra notebook
