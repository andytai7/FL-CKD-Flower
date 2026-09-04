"""P1 record-level rerun under DP-Adam semantics (post 2026-09-04 local-optimizer audit).

The first TS P1 pass ran plain per-record SGD at lr 0.5; an lr probe + r20 probe at seed 43
showed that local rule does not converge at all (curve flat at ~0.5 AUROC). `dp_sgd_local`
now applies DP-Adam (moments = post-processing of the noised clipped aggregate; per-record
clip+noise path unchanged), p1_spine.run_cell defaults lr=0.001. This runner regenerates the
whole P1 evidence surface under the corrected local rule and writes it to FRESH files,
leaving the superseded results/dp_ts_p1{,_seeds}.json pass-1 artifacts in place for diffs:
  - results/dl_ts_p1_adam.json            full eps grid, seed 42 (FedAvgM no prox row)
  - results/dl_ts_p1_adam_transport.json  eps=1 transport arms {fedavg, fedprox-0.1, fedavgm}
  - results/dl_ts_p1_adam_seeds.json      eps {off,1,4} x seeds 43-46
"""

from __future__ import annotations

import json
from pathlib import Path

from .federated import _split, load_seq_clinics
from .p1_spine import run_cell, run_grid

RBASE = Path(__file__).resolve().parents[2] / "results"


def _dump(name: str, meta: dict, rows: list) -> None:
    (RBASE / name).write_text(json.dumps({**meta, "rows": rows}, indent=2))
    print(f"wrote {RBASE / name} ({len(rows)} rows)")


def main() -> None:
    clinics = load_seq_clinics(downsample=4)
    # full seed-42 grid, default transport (FedAvgM)
    rows42 = run_grid(rounds=10, seeds=(42,), quiet=True)
    _dump("dl_ts_p1_adam.json",
          {"arm": "P1 record-level DP-Adam (LSTM, CinC seq clinics)",
           "grid": "eps x seed-42; FedAvgM transport; DP-Adam local lr=0.001"}, rows42)
    for r in rows42:
        print(f"  eps={'off' if r['target_epsilon'] is None else r['target_epsilon']:>4} "
              f"-> {r['final_auc']:.3f} (worst {r['final_auc_worst']:.3f})")

    # ε=1 transport arms (matrix Q3): same local rule, strategy swap only
    splits = [_split(X, y, 42, k) for k, (X, y) in enumerate(clinics)]
    ns = [len(s["ytr"]) for s in splits]
    arms = []
    for tag, kw in (("fedavg", dict(transport="fedavg")),
                    ("fedprox-0.1", dict(transport="fedavg", proximal_mu=0.1)),
                    ("fedavgm", dict(transport="fedavgm"))):
        r = run_cell(epsilon=1.0, rounds=10, splits=splits, clinic_ns=ns, seed=42,
                     quiet=True, **kw)
        r["transport_arm"] = tag
        arms.append(r)
    _dump("dl_ts_p1_adam_transport.json",
          {"arm": "P1 transport comparator at eps=1 (DP-Adam local)",
           "note": "fedavg / fedprox(mu=0.1) / fedavgm(beta=0.6, built-in)"}, arms)
    for r in arms:
        print(f"  transport {r['transport_arm']:<11} -> {r['final_auc']:.3f} "
              f"(worst {r['final_auc_worst']:.3f})")

    # seed dispersion on headline eps
    sweep = []
    for eps in (None, 1.0, 4.0):
        col = run_grid(epsilons=(eps,), rounds=10, seeds=(43, 44, 45, 46), quiet=True)
        sweep += col
    _dump("dl_ts_p1_adam_seeds.json",
          {"arm": "P1 DP-Adam seed sweep", "scope": "eps {off,1,4} x seeds 43-46"}, sweep)
    for r in sweep:
        print(f"  sweep eps={'off' if r['target_epsilon'] is None else r['target_epsilon']:>4} "
              f"seed={r['seed']} -> {r['final_auc']:.3f} (worst {r['final_auc_worst']:.3f})")


if __name__ == "__main__":
    main()
