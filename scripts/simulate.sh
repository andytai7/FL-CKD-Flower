#!/usr/bin/env bash
# Federated (Ray-free) simulation. Defaults: logreg, 12 practices, 20 rounds, non-IID.
# Override -> ./scripts/simulate.sh --rounds 10
#             ./scripts/simulate.sh --iid
#             ./scripts/simulate.sh --practices 25 --alpha 0.1
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run ckd-simulate "$@"
