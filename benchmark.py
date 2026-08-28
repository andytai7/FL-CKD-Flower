"""Run the full protocol benchmark and write machine-readable results.

Protocols under test — **FedProx**, **FedMosaic** — plus the two reference
baselines (`local`, `fedavg`) needed to interpret them, on both datasets (all logistic regression):

- `clinics` — the signal-bearing non-IID practices from `ckd-clinics`. This is the real comparison.
- `flat`    — `synthetic_ckd_data.csv`, which has no learnable signal. This is the **negative
  control**: any protocol scoring meaningfully above AUROC 0.5 here is reporting a bug, not a
  result. It is what makes the `clinics` numbers credible.

Everything is held fixed except the protocol (immutable rule 6): same seeds, same local 80/20
splits, same local scaler, same local optimiser.

    uv run ckd-benchmark                     # both datasets, 20 rounds, -> results/benchmark.json
    uv run ckd-benchmark --rounds 30
    uv run ckd-benchmark --dataset clinics
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import time
from pathlib import Path

import numpy as np
from flwr.common.logger import log as _flwr_log  # noqa: F401  (imported to force logger setup)

from centralized import run_centralized
from data import load_clinic_frames, load_partition, to_xy
from data.loader import FEATURE_COLS
from models.protocols import BASELINES, PROTOCOLS
from simulate import run_simulation

# Flower's strategies log an INFO line per aggregate call; at 4 comparators x 2 datasets x 20
# rounds that is pure noise around the results table.
logging.getLogger("flwr").setLevel(logging.ERROR)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
# Reported alongside the curves: how many rounds each protocol needs to first reach this AUROC.
TARGET_AUROC = 0.80


def _history(dataset: str, protocol: str, rounds: int, practices: int, seed: int) -> list[dict]:
    """One protocol run; per-round aggregated metrics."""
    with contextlib.redirect_stdout(io.StringIO()):
        return run_simulation(
            num_practices=practices,
            num_rounds=rounds,
            quiet=True,
            clinics_dir=True if dataset == "clinics" else None,
            protocol=protocol,
            seed=seed,
        )


def _rounds_to_target(history: list[dict], target: float) -> int | None:
    for i, row in enumerate(history, start=1):
        if float(row.get("auc", float("nan"))) >= target:
            return i
    return None


def _summarize(history: list[dict]) -> dict:
    final = history[-1]
    aucs = [float(r.get("auc", np.nan)) for r in history]
    return {
        "final_auc": float(final.get("auc", np.nan)),
        "final_auc_worst": float(final.get("auc_worst", np.nan)),
        "final_sensitivity": float(final.get("sensitivity", np.nan)),
        "best_auc": float(np.nanmax(aucs)),
        "rounds_to_target": _rounds_to_target(history, TARGET_AUROC),
        "uplink_bits_per_client_per_round": float(final.get("uplink-bits-per-client", np.nan)),
        "seconds_per_round": float(np.mean([
            float(r.get("round-seconds", np.nan)) for r in history
        ])),
        "alpha_mean_final": float(final.get("alpha-mean", np.nan)),
        "consensus_agreement_final": float(final.get("consensus-agreement", np.nan)),
        "auc_curve": [None if np.isnan(a) else round(a, 4) for a in aucs],
        "auc_worst_curve": [
            round(float(r.get("auc_worst", np.nan)), 4)
            if not np.isnan(float(r.get("auc_worst", np.nan))) else None
            for r in history
        ],
    }


def _ceilings(dataset: str, seed: int) -> dict:
    """Pooled-data ceiling on the SAME dataset — the only valid 'price of privacy' reference."""
    with contextlib.redirect_stdout(io.StringIO()):
        return {"logreg": run_centralized(seed=seed, clinics=(dataset == "clinics"))}


def _cohort_shape(dataset: str, practices: int, seed: int) -> dict:
    frames = (
        load_clinic_frames() if dataset == "clinics"
        else [load_partition(i, practices, seed=seed) for i in range(practices)]
    )
    sizes, rates = [], []
    for df in frames:
        _, y = to_xy(df)
        sizes.append(int(len(y)))
        rates.append(float(np.mean(y)))
    return {
        "practices": len(frames),
        "patients_total": int(sum(sizes)),
        "practice_sizes": sizes,
        "ckd_rate_per_practice": [round(r, 3) for r in rates],
        "features": FEATURE_COLS,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Protocol benchmark for the FLIP-IT CKD baseline")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--practices", type=int, default=12,
                        help="only used for the flat-CSV partitioned dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default="both", choices=["both", "clinics", "flat"])
    parser.add_argument("--out", default=str(RESULTS_DIR / "benchmark.json"))
    args = parser.parse_args()

    datasets = ["clinics", "flat"] if args.dataset == "both" else [args.dataset]
    comparators = [*PROTOCOLS, *BASELINES]

    results: dict = {
        "config": {
            "rounds": args.rounds,
            "seed": args.seed,
            "practices": args.practices,
            "target_auroc": TARGET_AUROC,
            "protocols": list(PROTOCOLS),
            "baselines": list(BASELINES),
        },
        "datasets": {},
    }

    for dataset in datasets:
        print(f"\n=== dataset: {dataset} ===")
        entry: dict = {
            "cohort": _cohort_shape(dataset, args.practices, args.seed),
            "centralized_ceiling": _ceilings(dataset, args.seed),
            "runs": {},
        }
        for name in comparators:
            started = time.perf_counter()
            history = _history(dataset, name, args.rounds, args.practices, args.seed)
            summary = _summarize(history)
            summary["wall_seconds"] = round(time.perf_counter() - started, 2)
            summary["kind"] = "protocol" if name in PROTOCOLS else "baseline"
            entry["runs"][name] = summary
            print(
                f"  {name:>10} [{summary['kind']:>8}]  AUROC={summary['final_auc']:.3f}  "
                f"worst={summary['final_auc_worst']:.3f}  "
                f"sens={summary['final_sensitivity']:.3f}  "
                f"rounds->{TARGET_AUROC}={summary['rounds_to_target']}  "
                f"uplink={summary['uplink_bits_per_client_per_round']:.0f} bits"
            )
        results["datasets"][dataset] = entry

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
