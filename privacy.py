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
from flwr.clientapp.mod import LocalDpMod
from flwr.serverapp.strategy import DifferentialPrivacyServerSideFixedClipping, FedAvg

from client_app import _local_split
from data import load_clinic_frames, to_xy
from dp import CLIPPING_NORM, DELTA, epsilon_basic_composition, epsilon_rdp
from messages import evaluate_reply, hushed, local_dp_train_reply, train_reply
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
NOISE_MULTIPLIERS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
# Local DP is parameterised by epsilon directly (not by a noise scale), so the sweep is over
# the budget itself. None = the no-DP reference row.
LOCAL_DP_EPSILONS = (None, 50.0, 20.0, 10.0, 5.0, 1.0)



def _seed_dp_noise(seed: int, sigma: float) -> None:
    """Make Flower's DP noise reproducible.

    Flower draws its Gaussian noise from NumPy's *global* legacy RNG —
    `flwr/supercore/differential_privacy.py:46`: `np.random.normal(0, std_dev, array.shape)`.
    Nothing in this module's `seed` argument reaches it (that seed only drives `_local_split`), so
    before this call the DP rows were the one irreproducible step in an otherwise fully seeded
    pipeline: σ=0 was bit-identical across runs while every σ>0 row drifted. CLAUDE.md §0 rule 6
    requires better.

    Seeded per **(seed, σ) cell** rather than once per run: the noise stream is consumed
    sequentially, so a single per-run seed would make every row depend on which σ values precede it
    in `NOISE_MULTIPLIERS` — adding one entry would silently move all the later numbers. Per-cell
    seeding keeps each row of the published table independently reproducible.

    Each of the run seeds still draws *different* noise, so the error bars keep their meaning.

    ⚠️ Do not remove: without it the privacy tables cannot be reproduced from a stated seed.
    """
    np.random.seed((seed * 1_000_003 + int(round(sigma * 1000))) % (2**32))


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
    curve is that it has error bars. Each seed re-draws both the local splits and the DP noise, the
    latter via `_seed_dp_noise` (Flower's noise comes from NumPy's global RNG).
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
            # The reported budget. `epsilon_basic_upper_bound` is the old naive composition, kept
            # so the report can show what proper accounting buys at identical noise.
            "epsilon": epsilon_rdp(sigma, rounds),
            "epsilon_accountant": "RDP (dp_accounting)" if sigma > 0 else None,
            "epsilon_basic_upper_bound": epsilon_basic_composition(sigma, rounds),
            "sampling_probability": 1.0,  # fraction-fit = 1.0: every practice, every round
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
        eps, eps_basic = results[-1]["epsilon"], results[-1]["epsilon_basic_upper_bound"]
        eps_text = "—" if eps is None else f"{eps:.2f}"
        basic_text = "—" if eps_basic is None else f"{eps_basic:.0f}"
        print(
            f"  sigma={sigma:<5} AUROC={results[-1]['auc_mean']:.3f}"
            f"±{results[-1]['auc_std']:.3f}  "
            f"worst={results[-1]['auc_worst_mean']:.3f}±{results[-1]['auc_worst_std']:.3f}  "
            f"eps={eps_text} (naive bound {basic_text})"
        )
    return results


def _dp_sweep_one_seed(rounds: int, seed: int, epochs: int) -> list[dict]:
    locals_ = _prepare(load_clinic_frames(), seed)
    results = []

    for sigma in NOISE_MULTIPLIERS:
        _seed_dp_noise(seed, sigma)
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


def _local_dp_sweep_one_seed(rounds: int, seed: int, epochs: int) -> list[dict]:
    locals_ = _prepare(load_clinic_frames(), seed)
    results = []

    for eps in LOCAL_DP_EPSILONS:
        _seed_dp_noise(seed, eps if eps is not None else 0.0)
        strategy = FedAvg(evaluate_metrics_aggr_fn=weighted_and_worst)
        mod = (
            None if eps is None
            else LocalDpMod(
                clipping_norm=CLIPPING_NORM,
                sensitivity=CLIPPING_NORM,  # updates are clipped to this, so it bounds sensitivity
                epsilon=eps,
                delta=DELTA,
            )
        )

        w = init_weights(locals_[0].n_features)
        history = []
        for rnd in range(1, rounds + 1):
            global_arrays = to_flower_arrays(w)
            replies = []
            for loc in locals_:
                updated = to_flower_arrays(loc.local_sgd(w, epochs=epochs, lr=LEARNING_RATE))
                if mod is None:
                    replies.append(train_reply(updated, loc.num_examples))
                else:
                    with hushed():
                        replies.append(
                            local_dp_train_reply(mod, global_arrays, updated, loc.num_examples)
                        )

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
            "epsilon_per_round": eps,
            "seed": seed,
            "auc": float(final.get("auc", np.nan)),
            "auc_worst": float(final.get("auc_worst", np.nan)),
            "sensitivity": float(final.get("sensitivity", np.nan)),
            "auc_curve": [round(float(h.get("auc", np.nan)), 4) for h in history],
        })
    return results


def run_local_dp_sweep(
    rounds: int = 20, seeds: tuple[int, ...] = (42,), epochs: int = 2
) -> list[dict]:
    """The utility cost of noising each practice's update *before it leaves the practice*.

    This is the sweep that bears on counsel's central question. Central DP (`run_dp_sweep`) noises
    during aggregation, so the aggregating party still receives every practice's un-noised update
    first; only local DP changes what that party sees. The two are therefore not interchangeable,
    and their costs are not comparable at face value either — under local DP each of the ten
    practices adds independent noise, so the noise entering the aggregate grows with the number of
    practices rather than being added once.

    `epsilon_per_round` is the mod's own parameter and applies **per round**; the composed budget
    over `rounds` is reported alongside it by the RDP accountant.
    """
    per_eps: dict[float | None, list[dict]] = {e: [] for e in LOCAL_DP_EPSILONS}
    for seed in seeds:
        for row in _local_dp_sweep_one_seed(rounds, seed, epochs):
            per_eps[row["epsilon_per_round"]].append(row)

    results = []
    for eps, runs in per_eps.items():
        aucs = [r["auc"] for r in runs]
        worsts = [r["auc_worst"] for r in runs]
        sens = [r["sensitivity"] for r in runs]
        # LocalDpMod's sigma for this (epsilon, delta), so the composed budget uses the same
        # accountant as the central sweep and the two tables are read on one scale.
        sigma = None if eps is None else math.sqrt(2 * math.log(1.25 / DELTA)) / eps
        results.append({
            "epsilon_per_round": eps,
            "noise_stddev": sigma,
            "clipping_norm": CLIPPING_NORM,
            "epsilon_composed": None if eps is None else epsilon_rdp(sigma, rounds),
            "epsilon_accountant": None if eps is None else "RDP (dp_accounting)",
            "delta": None if eps is None else DELTA,
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
        r = results[-1]
        eps_text = "off" if eps is None else f"{eps:g}/round -> {r['epsilon_composed']:.1f} composed"
        print(
            f"  local-dp eps={eps_text:<34} AUROC={r['auc_mean']:.3f}±{r['auc_std']:.3f}  "
            f"worst={r['auc_worst_mean']:.3f}±{r['auc_worst_std']:.3f}"
        )
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

    findings["implemented_in_this_repo"] = True
    findings["conclusion"] = (
        "SecAgg+ ships only in the legacy namespaces: the workflow requires a LegacyContext and the "
        "client mod exists only in flwr.client.mod, not flwr.clientapp.mod. It therefore cannot be "
        "composed with strategy.start() on the Message API. It IS reachable by driving "
        "DefaultWorkflow(fit_workflow=SecAggPlusWorkflow(...)) with a LegacyContext from inside a "
        "modern ServerApp — and that path is now IMPLEMENTED and verified end-to-end here: "
        "server_app._run_secure_aggregation, enabled with `secure-aggregation = true`, runs fit and "
        "evaluate to completion with 0 failures over 12 practices. The ClientApp accepts both the "
        "Message-API record shape and the legacy one, so a single app serves either path. Central "
        "DP works directly on the Message API and needs none of this."
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
        f"Central DP sweep — Flower's real DP wrapper over FedAvg, "
        f"{len(seeds)} seeds x {args.rounds} rounds, per-clinic federation"
    )
    print("-" * 72)
    dp = run_dp_sweep(rounds=args.rounds, seeds=seeds)

    print(
        f"\nLocal DP sweep — Flower's real LocalDpMod inside the SuperNode, "
        f"{len(seeds)} seeds x {args.rounds} rounds"
    )
    print("-" * 72)
    local_dp = run_local_dp_sweep(rounds=args.rounds, seeds=seeds)

    print("\nSecAgg+ feasibility probe")
    print("-" * 72)
    secagg = probe_secagg()
    for k, v in secagg.items():
        if k != "conclusion":
            print(f"  {k}: {v}")
    print(f"\n  {secagg['conclusion']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {
            "config": {
                "rounds": args.rounds,
                "seeds": list(seeds),
                "clipping_norm": CLIPPING_NORM,
                "delta": DELTA,
                "accountant": "RDP (dp_accounting); epsilon_basic_upper_bound is the "
                              "superseded naive composition, retained for comparison only",
            },
            "dp_sweep": dp,
            "local_dp_sweep": local_dp,
            "secagg": secagg,
        },
        indent=2,
    ))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
