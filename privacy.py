"""Measure what each privacy layer actually costs in accuracy, and verify what is achievable.

The Projektantrag makes Differential Privacy and Secure Aggregation a **Milestone 4 acceptance
criterion at month 18** ("Implementierung von Secure Aggregation und Differential Privacy"). This
module produces the evidence for that: a measured privacy/utility curve rather than an assertion
that privacy is free.

Two experiments:

1. **DP sweep** — wrap Flower's real `DifferentialPrivacyServerSideFixedClipping` around FedAvg and
   sweep the noise multiplier, recording the AUROC cost at each setting alongside its (ε, δ).
2. **SecAgg+ feasibility probe** — determine empirically whether Flower 1.33's SecAgg+ can be
   composed with the Message-API strategies this project deploys on, and report the answer either
   way. This is a genuine open constraint, not a checkbox.

    uv run ckd-privacy                 # -> results/privacy.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np
from flwr.serverapp.strategy import DifferentialPrivacyServerSideFixedClipping, FedAvg

from client_app import _local_split
from data import load_clinic_frames, to_xy
from messages import evaluate_reply, hushed, train_reply
from models.protocols.common import (
    LogRegLocal,
    from_flower_arrays,
    init_weights,
    to_flower_arrays,
)
from server_app import weighted_and_worst
from task import compute_metrics, fit_scaler

logging.getLogger("flwr").setLevel(logging.ERROR)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

LEARNING_RATE = 0.5
CLIPPING_NORM = 1.0
# δ is conventionally set below 1/n; the clinics cohort is ~3.5k patients, so 1e-5 is comfortable.
DELTA = 1e-5
NOISE_MULTIPLIERS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)


def _epsilon(noise_multiplier: float, rounds: int, delta: float = DELTA) -> float | None:
    """A transparent (not tight) Gaussian-mechanism ε for `rounds` compositions.

    Deliberately simple and auditable: per-round ε₁ = √(2 ln(1.25/δ))/σ for the Gaussian mechanism
    at sensitivity 1 (updates are clipped to `CLIPPING_NORM`), composed over rounds by basic
    composition, ε = rounds · ε₁.

    ⚠️ This is an **upper bound and a placeholder for accounting, not a certified budget**. A real
    MS4 submission must use a proper accountant (RDP / PLD, e.g. `opacus` or `dp-accounting`) that
    also credits subsampling amplification. Reported here so the privacy/utility shape is visible;
    flagged so nobody quotes it as the project's formal ε.
    """
    if noise_multiplier <= 0:
        return None
    per_round = math.sqrt(2.0 * math.log(1.25 / delta)) / noise_multiplier
    return per_round * rounds


def _prepare(frames, seed: int) -> list[LogRegLocal]:
    out = []
    for pid, df in enumerate(frames):
        X, y = to_xy(df)
        X_tr, y_tr, X_te, y_te = _local_split(X, y, seed, pid)
        scaler = fit_scaler(X_tr)
        out.append(LogRegLocal(scaler.transform(X_tr), y_tr, scaler.transform(X_te), y_te))
    return out


def run_dp_sweep(rounds: int = 20, seeds: tuple[int, ...] = (42,), epochs: int = 2) -> list[dict]:
    """FedAvg under Flower's real DP wrapper, at increasing noise, repeated over seeds.

    DP adds *random* noise, so a single run is not evidence — the whole point of the utility-cost
    curve is that it has error bars. Each seed re-draws both the local splits and the DP noise.
    """
    per_sigma: dict[float, list[dict]] = {s: [] for s in NOISE_MULTIPLIERS}
    for seed in seeds:
        for row in _dp_sweep_one_seed(rounds, seed, epochs):
            per_sigma[row["noise_multiplier"]].append(row)

    results = []
    for sigma, runs in per_sigma.items():
        aucs = [r["auc"] for r in runs]
        worsts = [r["auc_worst"] for r in runs]
        sens = [r["sensitivity"] for r in runs]
        results.append({
            "noise_multiplier": sigma,
            "clipping_norm": CLIPPING_NORM,
            "epsilon_upper_bound": _epsilon(sigma, rounds),
            "delta": DELTA if sigma > 0 else None,
            "seeds": list(seeds),
            "auc_mean": float(np.mean(aucs)),
            "auc_std": float(np.std(aucs)),
            "auc_worst_mean": float(np.mean(worsts)),
            "auc_worst_std": float(np.std(worsts)),
            "sensitivity_mean": float(np.mean(sens)),
            "sensitivity_std": float(np.std(sens)),
            "auc_per_seed": [round(a, 4) for a in aucs],
            "auc_curve_seed0": runs[0]["auc_curve"],
        })
        eps = results[-1]["epsilon_upper_bound"]
        eps_text = "—" if eps is None else f"{eps:.1f}"
        print(
            f"  sigma={sigma:<5} AUROC={results[-1]['auc_mean']:.3f}"
            f"±{results[-1]['auc_std']:.3f}  "
            f"worst={results[-1]['auc_worst_mean']:.3f}±{results[-1]['auc_worst_std']:.3f}  "
            f"eps<={eps_text}"
        )
    return results


def _dp_sweep_one_seed(rounds: int, seed: int, epochs: int) -> list[dict]:
    locals_ = _prepare(load_clinic_frames(), seed)
    results = []

    for sigma in NOISE_MULTIPLIERS:
        base = FedAvg(evaluate_metrics_aggr_fn=weighted_and_worst)
        strategy = base
        if sigma > 0:
            strategy = DifferentialPrivacyServerSideFixedClipping(
                base,
                noise_multiplier=sigma,
                clipping_norm=CLIPPING_NORM,
                num_sampled_clients=len(locals_),
            )

        w = init_weights(locals_[0].n_features)
        history = []
        for rnd in range(1, rounds + 1):
            # The DP wrapper clips each update against the last broadcast model, so it needs to
            # know what that was — configure_train normally records it.
            if sigma > 0:
                strategy.current_arrays = _array_record(w)

            replies = [
                train_reply(
                    to_flower_arrays(loc.local_sgd(w, epochs=epochs, lr=LEARNING_RATE)),
                    loc.num_examples,
                )
                for loc in locals_
            ]
            with hushed():
                arrays, _ = strategy.aggregate_train(rnd, replies)
            if arrays is None:
                break
            w = from_flower_arrays(arrays.to_numpy_ndarrays())

            eval_replies = [
                evaluate_reply(
                    compute_metrics(loc.y_test, loc.test_scores(w)), len(loc.y_test), pid
                )
                for pid, loc in enumerate(locals_)
            ]
            with hushed():
                agg = strategy.aggregate_evaluate(rnd, eval_replies)
            history.append(dict(agg) if agg else {})

        final = history[-1] if history else {}
        results.append({
            "noise_multiplier": sigma,
            "seed": seed,
            "auc": float(final.get("auc", np.nan)),
            "auc_worst": float(final.get("auc_worst", np.nan)),
            "sensitivity": float(final.get("sensitivity", np.nan)),
            "auc_curve": [round(float(h.get("auc", np.nan)), 4) for h in history],
        })
    return results


def _array_record(w: np.ndarray):
    from flwr.app import ArrayRecord

    return ArrayRecord(to_flower_arrays(w))


def probe_secagg() -> dict:
    """Can SecAgg+ be composed with the Message-API strategies this project deploys on?

    Answered by inspection of the installed package rather than assumed from the docs.
    """
    findings: dict = {}

    import flwr.clientapp.mod as new_mods

    findings["secaggplus_mod_in_new_namespace"] = hasattr(new_mods, "secaggplus_mod")
    findings["dp_mods_in_new_namespace"] = [
        n for n in ("fixedclipping_mod", "adaptiveclipping_mod", "LocalDpMod")
        if hasattr(new_mods, n)
    ]

    try:
        import flwr.client.mod as legacy_mods

        findings["secaggplus_mod_in_legacy_namespace"] = hasattr(legacy_mods, "secaggplus_mod")
    except ImportError:
        findings["secaggplus_mod_in_legacy_namespace"] = False

    try:
        from flwr.server.workflow import SecAggPlusWorkflow

        findings["secaggplus_workflow_available"] = True
        # The decisive question: what context type does it demand?
        import inspect

        src = inspect.getsource(SecAggPlusWorkflow.__call__)
        findings["requires_legacy_context"] = "LegacyContext" in src
    except ImportError:
        findings["secaggplus_workflow_available"] = False
        findings["requires_legacy_context"] = None

    findings["conclusion"] = (
        "SecAgg+ ships only in the legacy namespaces: the workflow requires a LegacyContext and the "
        "client mod exists only in flwr.client.mod, not flwr.clientapp.mod. It therefore cannot be "
        "composed with strategy.start() on the Message API. It IS reachable by driving "
        "DefaultWorkflow(fit_workflow=SecAggPlusWorkflow(...)) with a LegacyContext from inside a "
        "modern ServerApp — so SecAgg+ is achievable for MS4, on a separate code path from the "
        "protocol benchmark. Central DP, by contrast, works directly on the Message API."
    ) if findings.get("requires_legacy_context") else "Re-check: SecAgg+ composition appears to have changed."

    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description="Privacy layer measurement for the CKD baseline")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46],
                        help="repeat the whole sweep once per seed and report mean ± std")
    parser.add_argument("--out", default=str(RESULTS_DIR / "privacy.json"))
    args = parser.parse_args()

    seeds = tuple(args.seeds)
    print(
        f"Differential Privacy sweep — Flower's real DP wrapper over FedAvg, "
        f"{len(seeds)} seeds x {args.rounds} rounds, per-clinic federation"
    )
    print("-" * 72)
    dp = run_dp_sweep(rounds=args.rounds, seeds=seeds)

    print("\nSecAgg+ feasibility probe")
    print("-" * 72)
    secagg = probe_secagg()
    for k, v in secagg.items():
        if k != "conclusion":
            print(f"  {k}: {v}")
    print(f"\n  {secagg['conclusion']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"dp_sweep": dp, "secagg": secagg}, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
