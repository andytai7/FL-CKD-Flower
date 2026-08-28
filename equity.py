"""Era 14 — Equity-Priced Privacy Orchestration in cross-silo federated learning.

Registered design: `docs/ERAS.md` §3 (frozen 2026-08-27 before any run; thresholds never
move after results exist). One sentence: at a FROZEN clinic-mean privacy budget, shape each
clinic's epsilon by census (ε_k ∝ N_k^(−γ)) and measure whether the pre-specified vulnerable
cohort — the three smallest practices — gains AUROC against the deployed uniform-ε* standard
without paying it back on global AUROC.

Everything except γ is inherited from the standard (`notebooks/03_dpsgd_secagg_standard.ipynb`
and `docs/PRIVACY.md` §3.5): same five seeds, R = 10 rounds, E = 2 local epochs, lr = 0.5,
δ = 1e-5, the same `prepare_practices` splits, the same `run_standard` FedAvg +
SecAgg-semantics runner, the same `weighted_and_worst` dual-level logging every round.
γ = 0 reproduces the deployment standard exactly — the module certifies that identity
(plans bit-identical to `orchestrator.plan`) at every invocation, so a drifted allocator can
never publish. γ < 0 rows are the anti-equity twin: the control that must not buy
vulnerable-cohort utility (kill letter K2).

The orchestration side is entirely rule-table + accountant arithmetic (`orchestrator.plan_equity`
wraps the same certified `plan()` machinery per clinic); no learned component anywhere.

Kill letters (frozen in `docs/ERAS.md` §3, evaluated verbatim below): K0 replication guard,
K1 primary (vulnerable-trio Δ at γ=+0.5 vs 0, ε*=2), K2 anti-equity twin, K3 global price
letter, K4 budget integrity (structural raise in the planner).

Run: `uv run ckd-equity` (full registered grid) or `uv run ckd-equity --quick` (smoke:
one seed, ε* = 2, γ ∈ {−0.5, 0, +0.5}). Output: `results/equity.json`.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import stats

import dp
import dpsgd
import orchestrator
from data import load_clinic_frames
from privacy import LEARNING_RATE
from task import compute_metrics

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# ── frozen constants (docs/ERAS.md §3; identical to the deployment standard's sweep) ──────────
SEEDS = (42, 43, 44, 45, 46)  # the same five seeds as results/privacy.json and notebook 03
ROUNDS = 10  # repo default since PRIVACY.md §3.4 (rounds are the cheapest ε lever)
EPOCHS = 2  # pyproject local-epochs
LR = LEARNING_RATE  # 0.5 — privacy.py's DP-sweep rate; never a second convention
BUDGETS = (0.5, 2.0, 8.0)  # the standard's composed targets (PRIVACY.md §3.5)
GAMMAS = (-1.0, -0.5, 0.0, 0.5, 1.0)

VULNERABLE_K = 3  # pre-specified: the three smallest train censuses (never the min estimator)
PRIMARY_BUDGET = 2.0  # letters are evaluated at this budget only
PRIMARY_GAMMA = 0.5  # K1: utility-equity direction
TWIN_GAMMA = -0.5  # K2: anti-equity twin (must NOT beat the standard on the trio)
GLOBAL_PRICE = 0.01  # K3: equity purchase may not cost global AUROC beyond this
# K0 replication guard: the γ=0 row must recomputethe published standard row (PRIVACY.md §3.5).
K0_REFERENCE = {"auc_mean": 0.797, "auc_std": 0.011, "worst_mean": 0.649, "worst_std": 0.060}


def _certify_standard_identity(sizes: list[int]) -> None:
    """plan_equity(γ=0) MUST reproduce the deployed standard's plans bit-for-bit.

    This is the structural half of the replication guard: it fails loudly (raise, never a
    bare assert — `python -O` strips those) before any compute is spent, so an allocator
    defect cannot masquerade as a γ=0 baseline.
    """
    census = {str(k): n for k, n in enumerate(sizes)}
    for budget in BUDGETS:
        standard = orchestrator.plan(
            census, target_epsilon=budget, fed_rounds=ROUNDS, epochs=EPOCHS
        )
        shaped = orchestrator.plan_equity(
            census, budget_epsilon=budget, gamma=0.0, fed_rounds=ROUNDS, epochs=EPOCHS
        )
        for clinic, (ref, got) in enumerate(zip(standard, shaped, strict=True)):
            if (ref.batch_size, ref.sigma, ref.achieved_epsilon) != (
                got.batch_size,
                got.sigma,
                got.achieved_epsilon,
            ):
                raise RuntimeError(
                    f"γ=0 allocator drifted from the standard at clinic {clinic}, "
                    f"ε*={budget}: standard (b={ref.batch_size}, σ={ref.sigma}) vs "
                    f"shaped (b={got.batch_size}, σ={got.sigma})"
                )


def _plans_for_cell(
    sizes: list[int], *, gamma: float | None, budget: float | None
) -> tuple[list, dict]:
    """The executed plan for one (γ, ε*) cell, with its audit trail.

    `budget=None` is the σ=0 clipped reference (gamma irrelevant), mirroring the leading
    None row of `dpsgd.run_epsilon_sweep`: full-batch clipped GD, no noise, rng untouched.
    """
    if budget is None:
        plans = [
            SimpleNamespace(batch_size=n, sigma=0.0, clip=1.0, local_epochs=EPOCHS) for n in sizes
        ]
        return plans, {"gamma": None, "target_epsilon": None}
    census = {str(k): n for k, n in enumerate(sizes)}
    plans = orchestrator.plan_equity(
        census, budget_epsilon=budget, gamma=gamma, fed_rounds=ROUNDS, epochs=EPOCHS
    )
    targets = orchestrator.equity_targets(census, budget_epsilon=budget, gamma=gamma)
    audit = {
        "gamma": gamma,
        "target_epsilon": budget,
        "epsilon_min": min(targets.values()),
        "epsilon_max": max(targets.values()),
        "plan_seed0": [
            {
                "clinic": p.clinic,
                "n": p.n_patients,
                "batch_size": p.batch_size,
                "sigma": p.sigma,
                "target_epsilon": targets[p.clinic],
                "achieved_epsilon": p.achieved_epsilon,
            }
            for p in plans
        ],
    }
    # K4 is structural and loud: every row certified below target by plan(); re-verified here
    # so the result JSON's audit trail is self-contained even against a future planner edit.
    for p in plans:
        if p.achieved_epsilon > targets[p.clinic]:
            raise RuntimeError(
                f"clinic {p.clinic} achieved ε {p.achieved_epsilon} exceeds its "
                f"target {targets[p.clinic]} — budget integrity broken"
            )
    return plans, audit


def _run_cell(
    locals_by_seed: dict, sizes: list[int], trio: list[int], *, gamma: float | None, budget
) -> dict:
    """One (γ, ε*) cell across every registered seed; per-clinic finals kept for the trio."""
    plans, audit = _plans_for_cell(sizes, gamma=gamma, budget=budget)
    runs = []
    for seed in locals_by_seed:
        # Per-cell seeding identical to dpsgd._cell_seed: rows at the same (seed, budget)
        # share generator seeds across γ arms — the pairing the registered contrasts rely on.
        history, w = dpsgd.run_standard(
            locals_by_seed[seed],
            plans,
            rounds=ROUNDS,
            lr=LR,
            seed=dpsgd._cell_seed(seed, budget),
            keep_final_model=True,
        )
        final = history[-1] if history else {}
        per_clinic = [
            compute_metrics(loc.y_test, loc.test_scores(w))["auc"] for loc in locals_by_seed[seed]
        ]
        runs.append(
            {
                "seed": seed,
                "auc": float(final.get("auc", np.nan)),
                "auc_worst": float(final.get("auc_worst", np.nan)),
                "sensitivity": float(final.get("sensitivity", np.nan)),
                "trio_auc": float(np.nanmean([per_clinic[i] for i in trio])),
                "per_clinic_auc": [round(float(a), 4) for a in per_clinic],
            }
        )
    return {**audit, "runs": runs}


def _paired_ci(arm: list[float], base: list[float]) -> dict:
    """Paired per-seed contrast, 95% t-CI on the difference (the registered estimator)."""
    d = np.asarray(arm, dtype=float) - np.asarray(base, dtype=float)
    n = len(d)
    half = float(stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / math.sqrt(n)) if n > 1 else float(
        "nan"
    )
    return {
        "delta_mean": round(float(d.mean()), 4),
        "ci_half_width": round(half, 4),
        "ci": [round(float(d.mean()) - half, 4), round(float(d.mean()) + half, 4)],
        "per_seed_deltas": [round(float(x), 4) for x in d],
    }


def _evaluate_letters(cells: dict) -> dict:
    """The frozen kill letters of docs/ERAS.md §3, evaluated verbatim against the cells."""

    def cell(gamma, budget=PRIMARY_BUDGET):
        return cells[(gamma, budget)]

    def series(c, key):
        return [r[key] for r in c["runs"]]

    std = cell(0.0)
    k0_auc_ok = abs(np.mean(series(std, "auc")) - K0_REFERENCE["auc_mean"]) <= 2 * K0_REFERENCE[
        "auc_std"
    ]
    k0_worst_ok = abs(
        np.mean(series(std, "auc_worst")) - K0_REFERENCE["worst_mean"]
    ) <= 2 * K0_REFERENCE["worst_std"]
    k0 = {
        "letter": "replication guard — γ=0 recomputes PRIVACY.md §3.5",
        "reference": K0_REFERENCE,
        "observed": {
            "auc_mean": round(float(np.mean(series(std, "auc"))), 4),
            "worst_mean": round(float(np.mean(series(std, "auc_worst"))), 4),
        },
        "verdict": "PASS" if (k0_auc_ok and k0_worst_ok) else "VOID (instrument drift)",
    }
    k1_c = _paired_ci(series(cell(PRIMARY_GAMMA), "trio_auc"), series(std, "trio_auc"))
    k1 = {
        "letter": f"primary — trio Δ(γ=+{PRIMARY_GAMMA} − 0) at ε*={PRIMARY_BUDGET}",
        **k1_c,
        "verdict": (
            "NULL (claim dead, CI contains 0)"
            if k1_c["ci"][0] <= 0 <= k1_c["ci"][1]
            else ("POSITIVE" if k1_c["ci"][0] > 0 else "NEGATIVE (allocation harms the trio)")
        ),
    }
    k2_c = _paired_ci(series(cell(TWIN_GAMMA), "trio_auc"), series(std, "trio_auc"))
    k2 = {
        "letter": f"anti-equity twin — trio Δ(γ={TWIN_GAMMA} − 0) at ε*={PRIMARY_BUDGET}",
        **k2_c,
        "verdict": (
            "CLAIM DEAD (twin beats standard: mechanism reads backwards)"
            if k2_c["ci"][0] > 0
            else "quiet (twin does not beat the standard)"
        ),
    }
    k3_c = _paired_ci(series(cell(PRIMARY_GAMMA), "auc"), series(std, "auc"))
    k3 = {
        "letter": f"price letter — global Δ(γ=+{PRIMARY_GAMMA} − 0) may not fall below −{GLOBAL_PRICE}",
        **k3_c,
        "verdict": (
            f"FIRED (global cost beyond {GLOBAL_PRICE}: equity-at-a-price, win narrowed)"
            if k3_c["ci"][1] < -GLOBAL_PRICE
            else "quiet"
        ),
    }
    k4 = {
        "letter": "budget integrity — achieved ≤ target per row; clinic-mean == budget",
        "verdict": "PASS by construction (raises in equity_targets / _plans_for_cell)",
    }
    return {"K0": k0, "K1": k1, "K2": k2, "K3": k3, "K4": k4}


def main() -> None:
    parser = argparse.ArgumentParser(description="Era 14 — equity-priced privacy orchestration")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "equity.json")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="smoke: seed 42 only, ε*=2, γ ∈ {-0.5, 0, +0.5} plus the σ=0 reference",
    )
    args = parser.parse_args()

    seeds = (42,) if args.quick else SEEDS
    budgets = (PRIMARY_BUDGET,) if args.quick else BUDGETS
    gammas = (TWIN_GAMMA, 0.0, PRIMARY_GAMMA) if args.quick else GAMMAS

    frames = load_clinic_frames()
    locals_by_seed = {seed: dpsgd.prepare_practices(frames, seed) for seed in seeds}
    sizes = [len(loc.y) for loc in locals_by_seed[seeds[0]]]
    trio = sorted(range(len(sizes)), key=lambda i: sizes[i])[:VULNERABLE_K]
    print(
        f"Era 14 census (train N): {sizes}\n"
        f"pre-specified vulnerable trio: clinics {trio} (N = {[sizes[i] for i in trio]})"
    )
    _certify_standard_identity(sizes)
    print("γ=0 allocator identity vs orchestrator.plan: certified (bit-identical plans)")

    cells: dict = {}
    ref = _run_cell(locals_by_seed, sizes, trio, gamma=None, budget=None)
    cells[("ref", None)] = ref
    print(f"  σ=0 reference: trio={np.mean([r['trio_auc'] for r in ref['runs']]):.3f}")
    for budget in budgets:
        for gamma in gammas:
            cell = _run_cell(locals_by_seed, sizes, trio, gamma=gamma, budget=budget)
            cells[(gamma, budget)] = cell
            m = lambda k: np.mean([r[k] for r in cell["runs"]])  # noqa: E731
            print(
                f"  γ={gamma:+.1f} ε*={budget:g}: trio={m('trio_auc'):.3f}  "
                f"worst={m('auc_worst'):.3f}  global={m('auc'):.3f}  "
                f"sens={m('sensitivity'):.3f}  (ε_k ∈ [{cell['epsilon_min']:.2f}, "
                f"{cell['epsilon_max']:.2f}])"
            )

    letters = _evaluate_letters(cells) if not args.quick else {"skipped": "smoke run"}
    payload = {
        "experiment": "Era 14 — Equity-Priced Privacy Orchestration (docs/ERAS.md §3)",
        "constants": {
            "seeds": list(seeds),
            "rounds": ROUNDS,
            "epochs": EPOCHS,
            "lr": LR,
            "delta": dp.DELTA,
            "budgets": list(budgets),
            "gammas": list(gammas),
            "vulnerable_trio": {"clinics": trio, "train_n": [sizes[i] for i in trio]},
        },
        "cells": [
            {"gamma": g, "budget_epsilon": b, **c}
            for (g, b), c in sorted(cells.items(), key=lambda kv: (str(kv[0][1]), str(kv[0][0])))
        ],
        "letters": letters,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    if not args.quick:
        print("\n── registered letters (docs/ERAS.md §3) ──")
        for key, letter in letters.items():
            print(f"  {key}: {letter['verdict']}")
            if "delta_mean" in letter:
                print(f"      Δ={letter['delta_mean']} 95% CI {letter['ci']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
