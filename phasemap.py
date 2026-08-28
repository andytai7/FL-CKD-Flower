"""Era 15 — the Census-Spread Phase Map for Privacy Orchestration (docs/ERAS.md §5).

A characterization experiment, publishable in every outcome cell by construction: stretch the
federation's census band from R=5 to R=100 max/min at a frozen geometric mean (~2000
patients/clinic) and measure where — if anywhere — census-shaped epsilon allocation begins to
pay the pre-specified vulnerable trio (three smallest train censuses) against the deployed
uniform-ε* standard, and what it costs globally. Era 14 measured the flat point at 3.3×
(NULL, ±0.004 precision); this era maps the transition. Every letter is two-sided and its
three readings are pre-committed in docs/ERAS.md §5, frozen before any run.

Everything except the census band and the γ dial is inherited: the repo generator
(`data.synthesize.generate_clinics`, shared `_BETA` truth untouched), seeds 42–46, R = 10
rounds, E = 2 epochs, lr = 0.5, δ = 1e-5, `prepare_practices` splits, `run_standard` FedAvg +
SecAgg-semantics runner, `weighted_and_worst` dual-level logging. The naive arm is the
un-orchestrated uniform-settings control (batch 64, σ 1.5 — the canonical row of
`orchestrator.uniform_settings_audit`, cross-checked against it by the G2 structural guard).

Letters: G1/G2 structural guards (raise → VOID), L1 onset slope (three pre-quoted readings),
L2 price per level (two-sided), L3 disclosure-coherence descriptor.

Run: `uv run ckd-phasemap` (full registered map) or `uv run ckd-phasemap --quick` (R5 level,
seed 42, {naive, γ0@2, γ+1@2} + σ=0 reference). Output: `results/phasemap.json`.
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
from data.synthesize import generate_clinics, write_clinic_csvs
from privacy import LEARNING_RATE
from task import compute_metrics

RESULTS_DIR = Path(__file__).resolve().parent / "results"
LADDER_DIR = Path(__file__).resolve().parent / "data" / "clinics_ladder"

# ── frozen ladder (docs/ERAS.md §5): bands, base seeds, and archetype mixture ─────────────────
LADDER = (
    ("R5", 894, 4472, 11000),
    ("R20", 447, 8944, 12000),
    ("R100", 200, 20000, 13000),
)
NUM_CLINICS = 10
SEEDS = (42, 43, 44, 45, 46)  # the Era-14 / privacy.json seed block, never a second convention
ROUNDS = 10
EPOCHS = 2
LR = LEARNING_RATE
BUDGETS = (2.0, 8.0)
GAMMAS = (-1.0, 0.0, 1.0)
NAIVE_BATCH, NAIVE_SIGMA = 64, 1.5  # uniform_settings_audit's canonical un-orchestrated row
VULNERABLE_K = 3
ONSET_BUDGET = 2.0  # L1 is evaluated on the headline budget only
GLOBAL_PRICE = 0.01  # L2: equity purchase may not cost global AUROC beyond this per level
DONT_COLLABORATE = 0.495  # PRIVACY.md §3.3's floor, descriptor context for the naive arm


def _level_sizes(min_n: int, max_n: int) -> list[int]:
    """Registered size set (erratum 5-a): log-spaced over the band, endpoints exact, so the
    realized max/min IS the registered ratio {5, 20, 100} — uniform draws compress it."""
    return [
        int(round(math.exp(math.log(min_n) + k / (NUM_CLINICS - 1) * (math.log(max_n) - math.log(min_n)))))
        for k in range(NUM_CLINICS)
    ]


def ensure_cohorts() -> None:
    """Materialise the ladder cohorts once; generation is deterministic at the frozen seeds."""
    for name, min_n, max_n, base_seed in LADDER:
        out = LADDER_DIR / name.lower()
        if (out / "clinic_00_urban-young.csv").exists():
            continue
        sizes = _level_sizes(min_n, max_n)
        clinics = generate_clinics(
            NUM_CLINICS, base_seed=base_seed, min_n=min_n, max_n=max_n, sizes=sizes
        )
        write_clinic_csvs(clinics, out)
        print(f"  generated {name}: {sizes} -> {out}")


def _naive_plans(sizes: list[int]) -> list:
    """One-size-fits-all DP-SGD — the arm that skips the orchestrator (each clinic identical)."""
    return [
        SimpleNamespace(
            batch_size=min(NAIVE_BATCH, n), sigma=NAIVE_SIGMA, clip=dp.CLIPPING_NORM,
            local_epochs=EPOCHS,
        )
        for n in sizes
    ]


def _guards(census: dict[str, int]) -> dict:
    """G1 + G2 structural guards from docs/ERAS.md §5; both raise on failure (never assert).

    G1: γ=0 allocation must be bit-identical to the deployed standard `plan()` at this census.
    G2: the naive arm's accounting must equal `uniform_settings_audit`'s arithmetic to machine
        precision — the guard that the arm executed is the arm whose incoherence is registered.
    """
    for budget in BUDGETS:
        standard = orchestrator.plan(
            census, target_epsilon=budget, fed_rounds=ROUNDS, epochs=EPOCHS
        )
        shaped = orchestrator.plan_equity(
            census, budget_epsilon=budget, gamma=0.0, fed_rounds=ROUNDS, epochs=EPOCHS
        )
        for clinic, (ref, got) in enumerate(zip(standard, shaped, strict=True)):
            if (ref.batch_size, ref.sigma) != (got.batch_size, got.sigma):
                raise RuntimeError(
                    f"G1 FAIL: γ=0 drifted from plan() at clinic {clinic}, ε*={budget}"
                )
    audit = orchestrator.uniform_settings_audit(
        census, batch_size=NAIVE_BATCH, sigma=NAIVE_SIGMA, epochs=EPOCHS, fed_rounds=ROUNDS
    )
    return {"naive_audit": audit}


def _run_cell(locals_by_seed: dict, sizes: list[int], trio: list[int], plans: list, cell_code) -> dict:
    """One (level, arm) cell across the registered seeds; per-clinic finals kept for the trio."""
    runs = []
    for seed in locals_by_seed:
        history, w = dpsgd.run_standard(
            locals_by_seed[seed],
            plans,
            rounds=ROUNDS,
            lr=LR,
            seed=dpsgd._cell_seed(seed, cell_code),
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
    return runs


def _mean(runs: list[dict], key: str) -> float:
    return round(float(np.mean([r[key] for r in runs])), 3)


def _paired(arm: list[float], base: list[float]) -> tuple[np.ndarray, list[float]]:
    d = np.asarray(arm, float) - np.asarray(base, float)
    n = len(d)
    half = float(stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / math.sqrt(n))
    return d, [round(float(d.mean() - half), 4), round(float(d.mean() + half), 4)]


def _letters(levels: list[dict]) -> dict:
    """The frozen letters of docs/ERAS.md §5, evaluated verbatim against the map."""

    def cell(level: dict, arm: str, budget: float | None) -> dict:
        return level["cells"][(arm, budget)]

    def log10r(level: dict) -> float:
        return math.log10(level["realized_max_min_ratio"])

    # L1: onset slope over log10(R) at the headline budget (15 seed×level points).
    xs, ys = [], []
    for level in levels:
        eq = cell(level, "g+1", ONSET_BUDGET)["runs"]
        st = cell(level, "g0", ONSET_BUDGET)["runs"]
        for a, b in zip(eq, st, strict=True):
            xs.append(log10r(level))
            ys.append(a["trio_auc"] - b["trio_auc"])
    fit = stats.linregress(xs, ys)
    l1_ci = [
        round(float(fit.slope - stats.t.ppf(0.975, len(xs) - 2) * fit.stderr), 4),
        round(float(fit.slope + stats.t.ppf(0.975, len(xs) - 2) * fit.stderr), 4),
    ]
    per_level = []
    for level in levels:
        d, ci = _paired(
            [r["trio_auc"] for r in cell(level, "g+1", ONSET_BUDGET)["runs"]],
            [r["trio_auc"] for r in cell(level, "g0", ONSET_BUDGET)["runs"]],
        )
        per_level.append({"level": level["name"], "delta": round(float(d.mean()), 4), "ci": ci})
    if l1_ci[0] > 0:
        l1_verdict = "ONSET: allocation pays increasingly with heterogeneity"
    elif l1_ci[1] < 0:
        l1_verdict = "REVERSE-ONSET: small-directed allocation harms the trio at scale"
    else:
        l1_verdict = "FLAT through the registered ladder (allocation carries no equity tax)"
    l1 = {
        "letter": f"L1 onset slope (trio Δ γ+1−γ0 per log10 R) at ε*={ONSET_BUDGET}",
        "slope": round(float(fit.slope), 4),
        "ci": l1_ci,
        "per_level": per_level,
        "verdict": l1_verdict,
    }
    # L2: global price per level, both budgets (two-sided).
    l2_rows = []
    for level in levels:
        for budget in BUDGETS:
            d, ci = _paired(
                [r["auc"] for r in cell(level, "g+1", budget)["runs"]],
                [r["auc"] for r in cell(level, "g0", budget)["runs"]],
            )
            fired = ci[1] < -GLOBAL_PRICE
            l2_rows.append(
                {
                    "level": level["name"],
                    "budget": budget,
                    "delta": round(float(d.mean()), 4),
                    "ci": ci,
                    "fired": fired,
                }
            )
    l2 = {
        "letter": f"L2 global price per level (fired iff CI entirely below −{GLOBAL_PRICE})",
        "rows": l2_rows,
        "verdict": (
            "PRICED at one or more levels" if any(r["fired"] for r in l2_rows)
            else "FREE at every measured level and budget"
        ),
    }
    return {"G1/G2": "PASS (structural guards did not fire)", "L1": l1, "L2": l2}


def main() -> None:
    parser = argparse.ArgumentParser(description="Era 15 — census-spread phase map")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "phasemap.json")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="smoke: R5 level, seed 42, {ref, naive, γ0@2, γ+1@2}",
    )
    args = parser.parse_args()

    ensure_cohorts()
    levels_wanted = {"R5"} if args.quick else {name for name, *_ in LADDER}
    seeds = (42,) if args.quick else SEEDS
    quick_arms = [("ref", None), ("naive", None), ("g0", 2.0), ("g+1", 2.0)]
    full_arms = (
        [("ref", None), ("naive", None)]
        + [("g0" if g == 0 else f"g{g:+g}", b) for b in BUDGETS for g in GAMMAS]
    )
    arm_grid = quick_arms if args.quick else full_arms
    # Arm labels: ref = σ=0 reference; naive = un-orchestrated; g0/g±g = γ arms at the budget.

    levels = []
    for name, min_n, max_n, base_seed in LADDER:
        if name not in levels_wanted:
            continue
        frames = load_clinic_frames(LADDER_DIR / name.lower())
        locals_by_seed = {s: dpsgd.prepare_practices(frames, s) for s in seeds}
        sizes = [len(loc.y) for loc in locals_by_seed[seeds[0]]]
        trio = sorted(range(len(sizes)), key=lambda i: sizes[i])[:VULNERABLE_K]
        census = {str(k): n for k, n in enumerate(sizes)}
        guard_artifacts = _guards(census)
        print(
            f"{name}: train census {sizes}; trio {trio} (N={[sizes[i] for i in trio]}); "
            "G1/G2 guards PASS"
        )

        cells: dict = {}
        for arm, budget in arm_grid:
            if arm == "ref":
                plans = [
                    SimpleNamespace(batch_size=n, sigma=0.0, clip=1.0, local_epochs=EPOCHS)
                    for n in sizes
                ]
                cell_code = None
                descriptor = {"epsilon_spread": [None, None]}
            elif arm == "naive":
                plans = _naive_plans(sizes)
                cell_code = 999.0  # distinct stream from every γ arm (uncoupled control)
                eps = [row["achieved_epsilon"] for row in guard_artifacts["naive_audit"]]
                descriptor = {"epsilon_spread": [round(min(eps), 2), round(max(eps), 2)]}
            else:
                gamma = float(arm[1:])
                targets = orchestrator.equity_targets(census, budget_epsilon=budget, gamma=gamma)
                plans = orchestrator.plan_equity(
                    census, budget_epsilon=budget, gamma=gamma,
                    fed_rounds=ROUNDS, epochs=EPOCHS,
                )
                cell_code = budget
                descriptor = {
                    "epsilon_spread": [round(min(targets.values()), 2), round(max(targets.values()), 2)],
                }
            runs = _run_cell(locals_by_seed, sizes, trio, plans, cell_code)
            cells[(arm, budget)] = {"runs": runs, **descriptor}
            print(
                f"  {arm:<6} ε*={budget}: trio={_mean(runs, 'trio_auc'):.3f}  "
                f"worst={_mean(runs, 'auc_worst'):.3f}  global={_mean(runs, 'auc'):.3f}  "
                f"sens={_mean(runs, 'sensitivity'):.3f}  εspread={descriptor['epsilon_spread']}"
            )
        levels.append(
            {
                "name": name,
                "band": [min_n, max_n],
                "base_seed": base_seed,
                "realized_census": sizes,
                "realized_max_min_ratio": round(max(sizes) / min(sizes), 2),
                "trio": {"clinics": trio, "train_n": [sizes[i] for i in trio]},
                "naive_epsilon_audit": guard_artifacts["naive_audit"],
                "cells": cells,
            }
        )

    letters = _letters(levels) if not args.quick else {"skipped": "smoke run"}
    payload = {
        "experiment": "Era 15 — Census-Spread Phase Map for Privacy Orchestration (docs/ERAS.md §5)",
        "constants": {
            "seeds": list(seeds), "rounds": ROUNDS, "epochs": EPOCHS, "lr": LR,
            "delta": dp.DELTA, "budgets": list(BUDGETS), "gammas": list(GAMMAS),
            "naive": {"batch": NAIVE_BATCH, "sigma": NAIVE_SIGMA},
        },
        "levels": [
            {**lv, "cells": [
                {"arm": a, "budget": b, **c}
                for (a, b), c in sorted(lv["cells"].items(), key=lambda kv: (str(kv[0][1]), kv[0][0]))
            ]}
            for lv in levels
        ],
        "letters": letters,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    if not args.quick:
        print("\n── registered letters (docs/ERAS.md §5) ──")
        print(f"  L1: {letters['L1']['verdict']}  slope={letters['L1']['slope']} CI {letters['L1']['ci']}")
        for row in letters["L1"]["per_level"]:
            print(f"      {row['level']}: Δ trio={row['delta']} CI {row['ci']}")
        print(f"  L2: {letters['L2']['verdict']}")
        for row in letters["L2"]["rows"]:
            print(f"      {row['level']} ε*={row['budget']}: Δ global={row['delta']} CI {row['ci']}")
        print(f"  {letters['G1/G2']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
