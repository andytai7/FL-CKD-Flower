"""Seed-dispersion sweep for the image P1 arm (seeds 43-46 x headline eps {off,1,4}).

Mirrors research/privacy_dl_ts/sweep_p1_seeds.py; seed 42's full 7-epsilon grid lives in
results/dl_image_p1.json. The seed-42 rows share DP subsample draws across eps (flagged in
that commit); sweep cells use fresh per-seed RNGs, so dispersion estimates are clean.
"""

from __future__ import annotations

import json
from pathlib import Path

from .p1_image import run_grid

RESULTS = Path(__file__).resolve().parents[2] / "results" / "dl_image_p1_seeds.json"
RESULTS.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    for eps in (None, 1.0, 4.0):
        col = run_grid(epsilons=(eps,), rounds=10, seeds=(43, 44, 45, 46), quiet=True)
        rows = json.loads(RESULTS.read_text())["rows"] if RESULTS.exists() else []
        rows += col
        RESULTS.write_text(json.dumps({"arm": "P1 DP-SGD seed sweep (CNN-Small, DermaMNIST)",
                                       "scope": "eps {off,1,4} x seeds 43-46; seed-42 full grid in dl_image_p1.json",
                                       "rows": rows}, indent=2))
        for r in col:
            print(f"  sweep eps={'off' if eps is None else eps} seed={r['seed']} -> "
                  f"AUROC {r['final_auc']:.3f} (worst {r['final_auc_worst']:.3f})")
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
