"""Per-layer clipping arm for the image P1 benchmark (dp_sgd_local clipping="perlayer").

Same ε-grid and conventions as p1_image.run_grid (rounds, steps_epochs, FedProx μ=0.1,
per-clinic standardised plans) with the layer mechanism: 5 module groups on CNN-S
(conv.0/conv.3/conv.6/head.1/head.3), per-record per-layer clip C, noise σ√L·C per layer so
the composed ε equals the global-clip cell — same accountant input, different clipping
geometry. Each row carries the filter-health trace (per-round per-layer clip fraction +
pre-clip norm median/p95, clinic-mean) plus the same per-cell MIA audit as the global arm.

Output: results/dl_image_p1_layerwise.json (rows have "clipping": "perlayer").
"""

from __future__ import annotations

import json
from pathlib import Path

from .federated import _split, load_image_clinics
from .p1_image import run_cell

RESULTS = Path(__file__).resolve().parents[2] / "results" / "dl_image_p1_layerwise.json"
RESULTS.parent.mkdir(parents=True, exist_ok=True)

ENVELOPE = {"arm": "P1 record-level DP-SGD, PER-LAYER clip arm (CNN-Small, DermaMNIST melanoma)",
            "mechanism": ("module-group per-record clip C=1 each of L=5 layers; per-layer noise "
                          "sigma*sqrt(L)*C -> composed epsilon matches the global-clip cells; "
                          "accounting unchanged (dp.py RDP route)"),
            "mia": "loss-threshold attack, pooled members/non-members cap 2000"}


def run_grid(*, epsilons=(None, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0), rounds: int = 10,
             seeds=(42,), quiet: bool = False) -> list[dict]:
    clinics = load_image_clinics()
    rows = []
    if RESULTS.exists():  # idempotent resume: skip completed (seed, epsilon) cells
        rows = json.loads(RESULTS.read_text())["rows"]
    done = {(r["seed"], r["target_epsilon"]) for r in rows}
    for seed in seeds:
        splits = [_split(X, y, seed, k) for k, (X, y) in enumerate(clinics)]
        ns = [len(s["ytr"]) for s in splits]
        for eps in epsilons:
            if (seed, eps) in done:
                continue
            rows.append(run_cell(epsilon=eps, rounds=rounds, splits=splits, clinic_ns=ns,
                                 seed=seed, clipping="perlayer", quiet=quiet))
            RESULTS.write_text(json.dumps({**ENVELOPE, "rows": rows}, indent=2))
    return rows


if __name__ == "__main__":
    run_grid()
    print(f"wrote {RESULTS}")
