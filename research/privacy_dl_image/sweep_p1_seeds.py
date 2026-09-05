"""Seed-dispersion sweep for the image P1 arm (seeds 43-46 x headline eps {off,1,4}).

Mirrors research/privacy_dl_ts/sweep_p1_seeds.py; seed 42's full 7-epsilon grid lives in
results/dl_image_p1.json. The seed-42 rows share DP subsample draws across eps (flagged in
that commit); sweep cells use fresh per-seed RNGs, so dispersion estimates are clean.

WRITER DISCIPLINE (2026-09-05 fix): this script calls run_cell directly; it MUST NOT call
run_grid — run_grid's per-cell checkpoint targets dl_image_p1.json and clobbered the seed-42
full grid on 2026-09-04 (recovered in the rerun that followed).
"""

from __future__ import annotations

import json
from pathlib import Path

from .federated import _split, load_image_clinics
from .p1_image import run_cell

RESULTS = Path(__file__).resolve().parents[2] / "results" / "dl_image_p1_seeds.json"
RESULTS.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    clinics = load_image_clinics()
    rows = json.loads(RESULTS.read_text())["rows"] if RESULTS.exists() else []
    done = {(r["seed"], r["target_epsilon"], r.get("clipping", "global")) for r in rows}
    for eps in (None, 1.0, 4.0):
        for seed in (43, 44, 45, 46):
            if (seed, eps, "global") in done:
                continue
            splits = [_split(X, y, seed, k) for k, (X, y) in enumerate(clinics)]
            ns = [len(s["ytr"]) for s in splits]
            r = run_cell(epsilon=eps, rounds=10, splits=splits, clinic_ns=ns, seed=seed,
                         quiet=True)
            rows.append(r)
            RESULTS.write_text(json.dumps({"arm": "P1 DP-SGD seed sweep (CNN-Small, DermaMNIST)",
                                           "scope": "eps {off,1,4} x seeds 43-46; seed-42 full grid in dl_image_p1.json",
                                           "rows": rows}, indent=2))
            print(f"  sweep eps={'off' if eps is None else eps} seed={r['seed']} -> "
                  f"AUROC {r['final_auc']:.3f} (worst {r['final_auc_worst']:.3f})")
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
