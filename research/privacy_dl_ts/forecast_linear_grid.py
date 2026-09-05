"""Linear-forecaster columns for the reference cross-check grid (results/dl_ts_forecast_grid.json).

Runs run_forecast_smoke with model_kind="linear" over the GRID_HORIZONS suite map (same
rounds/seeds/epochs as the GRU/LSTM columns) and appends rows to the shared grid file.
Idempotent: (suite, model, horizon) triples already present are skipped.
"""

from __future__ import annotations

import json
from pathlib import Path

from .forecast import GRID_HORIZONS, SUITES, run_forecast_smoke

RESULTS = Path(__file__).resolve().parents[2] / "results" / "dl_ts_forecast_grid.json"


def main() -> None:
    rows: list[dict] = json.loads(RESULTS.read_text())["rows"] if RESULTS.exists() else []
    done = {(r["suite"], r["model"], r["horizon"]) for r in rows}
    for suite in SUITES:
        for horizon in GRID_HORIZONS[suite]:
            if (suite, "linear", horizon) in done:
                continue
            col = run_forecast_smoke(suite=suite, horizon=horizon, rounds=5, seeds=(42,),
                                     epochs=1, model_kind="linear")
            rows += col
            envelope = json.loads(RESULTS.read_text())
            envelope["rows"] = rows
            RESULTS.write_text(json.dumps(envelope, indent=2))
            print(f"[grid-linear] {suite}/h{horizon} -> MSE {col[-1]['mse']:.4f}")
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
