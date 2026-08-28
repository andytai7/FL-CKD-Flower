"""Patient-level DP-SGD inside each clinic, composed with Secure Aggregation — the privacy standard.

The deployment this module implements moves the privacy boundary **inside the clinic**:

1. **Record-level DP-SGD** (`dp_sgd_local`): each clinic trains its logistic model with
   Poisson-sampled, per-sample-clipped, Gaussian-noised gradient steps. The protected unit is the
   *patient record*, not the clinic update: one record changes one per-sample gradient, which is
   clipped to norm C before it can influence anything else. Per-sample gradients are UNWEIGHTED
   (see `per_sample_grads`): the balanced class weights the non-DP training paths use are
   dataset-dependent, and folding them in would make one record change every record's clipped
   term — silently voiding the sensitivity-C account the accountant composes. This is the Opacus
   mechanism — the Abadi et al. 2016 sampled Gaussian with RDP composition — made *exact* here
   because logistic gradients are closed form, so `per_sample_grads` vectorises every record's
   gradient with no per-example autograd and no gradient-estimation gap.
2. **Secure Aggregation on the channel** (`run_standard`): the server receives only the masked
   *sum* of clinic updates, never an individual one. The in-process runner cannot do real
   cryptography, so it enforces the same information boundary instead: updates go straight into
   Flower's real `FedAvg`, are never retained or inspected past `aggregate_train`, and the average
   of exactly-K-summed-and-noised updates is what the masked sum would reveal anyway. The actual
   masking on the live path is the `secure-aggregation=true` deployment verified by
   `privacy.probe_secagg`.

Why record-level and not update-level local DP: under update-level LDP (`privacy.run_local_dp_sweep`,
Flower's `LocalDpMod`) every clinic noises its whole update independently, so the noise entering the
aggregate grows with the number of clinics. Here the noise is added once per step against the sum of
*patient-example* contributions — the multi-client noise penalty disappears, and the budget is spent
per patient, which is the guarantee a data-protection reviewer actually asks for.

Accounting contract (shared with the orchestrator, never hand-rolled here): one local step is a
Poisson-sampled Gaussian with rate `q = batch_size / N` and noise multiplier `sigma`;
`T = fed_rounds * ceil(local_epochs * N / batch_size)` steps compose via `dp.epsilon_rdp`, and
`dp.sigma_for_epsilon` inverts that to the smallest σ meeting a target ε. The orchestrator picks
(batch_size, sigma, clip, local_epochs) per clinic so every clinic lands on the SAME target ε
regardless of its N; this module just executes the plan.

Determinism: every draw comes from seeded `numpy.random.default_rng` Generators — never the global
legacy RNG that `privacy._seed_dp_noise` has to tame. Seeds are hash-combined per (seed, clinic,
round) and per (seed, ε) cell, following the per-cell rationale of `_seed_dp_noise`: each table row
is independently reproducible, and adding an ε value to a sweep does not shift neighbouring rows.
"""

from __future__ import annotations

import logging
import math
from types import SimpleNamespace

import numpy as np
from flwr.serverapp.strategy import FedAvg

from client_app import _local_split
from data import load_clinic_frames, to_xy
from messages import evaluate_reply, hushed, train_reply
from models.protocols.common import (
    L2,
    LogRegLocal,
    from_flower_arrays,
    init_weights,
    predict_proba,
    to_flower_arrays,
)
from server_app import weighted_and_worst
from task import compute_metrics, fit_scaler

logging.getLogger("flwr").setLevel(logging.ERROR)


def per_sample_grads(loc: LogRegLocal, w: np.ndarray) -> np.ndarray:
    """Every record's UNWEIGHTED logistic gradient `(p_i - y_i) * [x_i, 1]`, shape (N, d+1).

    Deliberately unweighted inside the privacy-charged mechanism: `LogRegLocal.sw` is
    dataset-dependent (`n / (2 * n_class)`) — adding or removing one record changes every record's
    `sw`, so an add/remove-one-record neighbour would move ALL N clipped terms, not one, and the
    accountant's 'one record alters one clipped gradient (norm ≤ C)' premise quietly becomes
    false. DP-SGD therefore charges the plain per-record logistic gradient; clip at C then bounds
    a record's total influence on the noised sum *exactly*. If class reweighting is ever wanted
    back under DP it must use dataset-independent constants (e.g. a fixed prior), never the local
    label counts — and it must be re-accounted.
    """
    Xb = np.column_stack([loc.X, np.ones(len(loc.X))])
    residual = predict_proba(loc.X, w) - loc.y
    return residual[:, None] * Xb


def dp_sgd_local(
    loc: LogRegLocal,
    w: np.ndarray,
    *,
    batch_size: int,
    sigma: float,
    clip: float,
    epochs: int,
    lr: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """One clinic's local DP-SGD: Poisson batches, per-sample clipping, Gaussian noise.

    Per step (Opacus convention):

    - Poisson inclusion: each record joins the batch independently with rate `q = batch_size / N`.
      Empty draws are SKIPPED — no step, and crucially no noise draw, so the seeded RNG stream stays
      aligned with executed steps and runs stay bit-reproducible.
    - Each sampled per-sample gradient is clipped to L2 norm `clip` (C) BEFORE summing; clipping is
      what bounds the weighted per-record contribution the accountant charges for.
    - Noisy gradient: `(sum_i clip(g_i, C) + Normal(0, (sigma*C)² I)) / batch_size` — the
      denominator is the EXPECTED batch size, not the realised one (dividing by the realised size
      would leak the batch cardinality).
    - The data-independent ridge term `L2 * w[:-1]` rides on the coef block unnoised; it touches no
      record and is not a privacy cost.

    `sigma=0` is the clipped-but-noiseless ablation: full-batch clipped GD, and NO rng consumption
    at all (a σ=0 reference row must not drift when neighbouring cells change).

    Number of steps is `ceil(epochs * N / batch_size)` — the same T the orchestrator accounts with
    (`dp.sigma_for_epsilon(target, T, delta, q)`). The accountant therefore OVER-COVERS the
    executed spend: empty Poisson lots are skipped with no noise draw, so the executed budget is
    at or below the planned one — the safe direction, and the guarantee sentence stays clean.
    """
    w = np.asarray(w, dtype=np.float64).copy()
    N = loc.num_examples
    steps = math.ceil(epochs * N / batch_size)
    q = min(batch_size / N, 1.0)
    for _ in range(steps):
        g = per_sample_grads(loc, w)
        norms = np.linalg.norm(g, axis=1)
        scale = np.minimum(1.0, clip / np.maximum(norms, 1e-12))
        if sigma > 0:
            mask = rng.random(N) < q
            if not mask.any():
                continue
            summed = (g[mask] * scale[mask, None]).sum(axis=0)
            noise = rng.normal(0.0, sigma * clip, size=w.shape)
            step = (summed + noise) / batch_size
        else:
            step = (g * scale[:, None]).sum(axis=0) / N
        step[:-1] += L2 * w[:-1]
        w -= lr * step
    return w


def _round_seed(seed: int, clinic: int, rnd: int) -> int:
    """Hash-combine (seed, clinic, round) into one deterministic Generator seed.

    Per-clinic-and-round rather than one stream per run: a fresh Generator per (clinic, round)
    means a clinic's noise depends only on its own cell coordinates, never on how many draws a
    sibling clinic consumed before it. Same rationale as `privacy._seed_dp_noise`'s per-cell seeding,
    expressed with Generators instead of the global legacy RNG.
    """
    return (seed * 1_000_003 + clinic * 9_973 + rnd * 91_193) % (2**32)


def run_standard(locals_: list, plans: list, *, rounds: int, lr: float, seed: int,
    keep_final_model: bool = False,
):
    """One federated run under the orchestrator's per-clinic DP plans; returns the metric history.

    `plans[k]` is duck-typed (attributes `batch_size`, `sigma`, `clip`, `local_epochs`) and is
    executed verbatim — this module never re-derives hyperparameters, so the accounted T and the
    executed T cannot diverge.

    Every round: each clinic runs `dp_sgd_local` from the current global w under its own seeded
    Generator, replies with `messages.train_reply`, and Flower's real `FedAvg`
    (`evaluate_metrics_aggr_fn=weighted_and_worst`) aggregates. **SecAgg boundary**: the runner
    emulates masked-sum semantics — individual updates are never read, copied, or retained; they go
    into `aggregate_train` and the reply list is deleted immediately after. (The cryptographic
    masking on the live path is the `secure-aggregation=true` run path verified by
    `privacy.probe_secagg`; what this process observes is exactly what a SecAgg server would learn —
    the weighted sum — and nothing more.) The new global model is then evaluated per clinic and
    passed through `aggregate_evaluate`, yielding the same per-round dict shape `privacy.py`'s
    sweeps produce.
    `keep_final_model=True` additionally returns the final global weights, letting a caller
    score per-clinic metrics that the aggregate history collapses (Era 14's pre-specified
    vulnerable-cohort estimator in equity.py needs per-clinic finals; the default call shape
    is unchanged).
    """
    strategy = FedAvg(evaluate_metrics_aggr_fn=weighted_and_worst)
    w = init_weights(locals_[0].n_features)
    history = []
    for rnd in range(1, rounds + 1):
        replies = [
            train_reply(
                to_flower_arrays(
                    dp_sgd_local(
                        loc,
                        w,
                        batch_size=plan.batch_size,
                        sigma=plan.sigma,
                        clip=plan.clip,
                        epochs=plan.local_epochs,
                        lr=lr,
                        rng=np.random.default_rng(_round_seed(seed, pid, rnd)),
                    )
                ),
                loc.num_examples,
            )
            for pid, (loc, plan) in enumerate(zip(locals_, plans, strict=True))
        ]
        with hushed():
            arrays, _ = strategy.aggregate_train(rnd, replies)
        del replies  # the individual update exists nowhere past the aggregate
        if arrays is None:
            break
        w = from_flower_arrays(arrays.to_numpy_ndarrays())

        eval_replies = [
            evaluate_reply(compute_metrics(loc.y_test, loc.test_scores(w)), len(loc.y_test), pid)
            for pid, loc in enumerate(locals_)
        ]
        with hushed():
            agg = strategy.aggregate_evaluate(rnd, eval_replies)
        history.append(dict(agg) if agg else {})
    if keep_final_model:
        return history, w
    return history

def prepare_practices(frames, seed: int) -> list[LogRegLocal]:
    """The public twin of `privacy._prepare`: split, scale on train only, wrap in LogRegLocal."""
    out = []
    for pid, df in enumerate(frames):
        X, y = to_xy(df)
        X_tr, y_tr, X_te, y_te = _local_split(X, y, seed, pid)
        scaler = fit_scaler(X_tr)
        out.append(LogRegLocal(scaler.transform(X_tr), y_tr, scaler.transform(X_te), y_te))
    return out


def _cell_seed(seed: int, target_epsilon: float | None) -> int:
    """Per-(seed, ε) sweep seed, so each result row reproduces independently of its neighbours."""
    cell = 0 if target_epsilon is None else int(round(target_epsilon * 1000))
    return (seed * 1_000_003 + cell) % (2**32)


def _plan_row(plans: list, sizes: list[int], rounds: int) -> list[dict]:
    """One executed plan as plain dicts, for the result row's audit trail."""
    return [
        {
            "clinic": k,
            "n": n,
            "batch_size": int(p.batch_size),
            "sigma": float(p.sigma),
            "steps": rounds * math.ceil(p.local_epochs * n / p.batch_size),
        }
        for k, (p, n) in enumerate(zip(plans, sizes, strict=True))
    ]


def run_epsilon_sweep(
    target_epsilons: tuple,
    *,
    rounds: int,
    epochs: int,
    lr: float,
    seeds: tuple,
    planner,
) -> list[dict]:
    """Utility at each target ε under the orchestrator's standardised plans, repeated over seeds.

    `planner(sizes, target_epsilon, rounds, epochs) -> list[plan]` is INJECTED — this module never
    imports the orchestrator (the notebook wires the two), so the sweep can run against any planner
    satisfying the duck-typed plan contract. `sizes` are TRAIN sizes (`len(loc.y)`): DP-SGD runs on
    the train splits, so train N is what sets both q and T — not `num_examples` vs test confusion
    would silently mis-set the accounted budget.

    A leading `None` row is prepended as the σ=0 clipped reference (like `LOCAL_DP_EPSILONS`' None
    row in privacy.py). Locals are prepared once per seed and shared across ε cells of that seed, so
    cells differ only in noise, not in data splits. Result rows mirror privacy.py's sweep rows,
    plus `plan_seed0` — the executed plan for the first seed, kept in the row so every reported
    number is auditable against concrete (batch_size, σ, steps).
    """
    frames = load_clinic_frames()
    epsilons = (None, *target_epsilons)
    locals_by_seed = {seed: prepare_practices(frames, seed) for seed in seeds}

    results = []
    for eps in epsilons:
        # The planner dominates runtime (dp_accounting bisection), so it is called ONCE per ε
        # cell and shared across seeds. Sound because sizes are seed-invariant by construction:
        # _local_split always 80/20-splits the same frames, so len(loc.y) never depends on seed.
        sizes = [len(loc.y) for loc in locals_by_seed[seeds[0]]]
        if eps is None:
            # σ=0 reference: full-batch clipped GD; clip/epochs still exercised, no noise.
            plans = [
                SimpleNamespace(batch_size=n, sigma=0.0, clip=1.0, local_epochs=epochs)
                for n in sizes
            ]
        else:
            plans = planner(sizes, float(eps), rounds, epochs)

        runs = []
        for seed in seeds:
            history = run_standard(
                locals_by_seed[seed], plans, rounds=rounds, lr=lr, seed=_cell_seed(seed, eps)
            )
            final = history[-1] if history else {}
            runs.append(
                {
                    "auc": float(final.get("auc", np.nan)),
                    "auc_worst": float(final.get("auc_worst", np.nan)),
                    "sensitivity": float(final.get("sensitivity", np.nan)),
                    "auc_curve": [round(float(h.get("auc", np.nan)), 4) for h in history],
                }
            )
        aucs = [r["auc"] for r in runs]
        worsts = [r["auc_worst"] for r in runs]
        sens = [r["sensitivity"] for r in runs]
        results.append(
            {
                "target_epsilon": eps,
                "seeds": list(seeds),
                "auc_mean": float(np.mean(aucs)),
                "auc_std": float(np.std(aucs)),
                "auc_worst_mean": float(np.mean(worsts)),
                "auc_worst_std": float(np.std(worsts)),
                "sensitivity_mean": float(np.mean(sens)),
                "sensitivity_std": float(np.std(sens)),
                "auc_per_seed": [round(a, 4) for a in aucs],
                "auc_curve_seed0": runs[0]["auc_curve"],
                "plan_seed0": _plan_row(plans, sizes, rounds),
            }
        )
        r = results[-1]
        eps_text = "off" if eps is None else f"{eps:g}"
        print(
            f"  eps={eps_text:<6} AUROC={r['auc_mean']:.3f}±{r['auc_std']:.3f}  "
            f"worst={r['auc_worst_mean']:.3f}±{r['auc_worst_std']:.3f}"
        )
    return results
