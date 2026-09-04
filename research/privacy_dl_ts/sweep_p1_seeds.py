"""Seed-dispersion sweep for the P1 spine (seeds 43-46 x headline eps {off,1,4}).

Seed 42's full 6-epsilon grid lives in results/dl_ts_p1.json (single-seed shape data).
The matrix needs dispersion where decisions get made: off / strict (1) / loose (4).
Rows accumulate into results/dl_ts_p1_seeds.json; merge at matrix-emission time.
Each (cell, seed) draw is independent here (the seed-42 rows shared subsample RNGs across
eps rows by design, and redecorrelation at sweep time was the documented follow-up).
"""

from __future__ import annotations

import json
from pathlib import Path

from .p1_spine import run_grid

RESULTS = Path(__file__).resolve().parents[2] / "results" / "dl_ts_p1_seeds.json"
RESULTS.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    for eps in (None, 1.0, 4.0):
        col = run_grid(epsilons=(eps,), rounds=10, seeds=(43, 44, 45, 46), quiet=True)
        rows = json.loads(RESULTS.read_text())["rows"] if RESULTS.exists() else []
        rows += col
        RESULTS.write_text(json.dumps({"arm": "P1 DP-SGD seed sweep (LSTM, CinC-2017 ECG)",
                                       "scope": "eps {off,1,4} x seeds 43-46; seed-42 full grid in dl_ts_p1.json",
                                       "rows": rows}, indent=2))
        for r in col:
            print(f"  sweep eps={'off' if eps is None else eps} seed={r['seed']} -> "
                  f"AUROC {r['final_auc']:.3f} (worst {r['final_auc_worst']:.3f})")
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
