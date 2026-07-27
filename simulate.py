"""In-process federated simulation driven by Flower.ai's **real** strategies.

`flwr run` uses Flower's Ray simulation engine, which cannot handle a project path containing a
space (this folder, `.../FL CKD Flower`, has one) — Ray's worker launcher splits the path and
crashes. This runner avoids Ray, but **everything that matters is still genuine `flwr` code**:

- `logreg` / `mlp` are the real `flwr.client.NumPyClient` aggregated by the real
  `flwr.server.strategy.FedAvg` (we call its `aggregate_fit` / `aggregate_evaluate` each round);
- `xgboost` is federated by the real `flwr.server.strategy.FedXgbBagging` — each practice grows a
  few local trees and the strategy bags them into one global ensemble (see `models/fedxgb.py`);
- metric aggregation is the real `server_app.weighted_and_worst`, wired as the strategy's
  `evaluate_metrics_aggregation_fn`, exactly as the ServerApp does it.

So this is not a re-implementation of federation — it is Flower's own strategies, executed without
the Ray transport layer. For the full Ray engine, run `flwr run .` from a space-free path (see
CLAUDE.md §6: set `UV_PROJECT_ENVIRONMENT` to a path without spaces).

    uv run ckd-simulate                       # 12 practices, 20 rounds, logreg (FedAvg), non-IID
    uv run ckd-simulate --model mlp --rounds 10
    uv run ckd-simulate --model xgboost       # federated trees via FedXgbBagging
    uv run ckd-simulate --iid                 # IID comparison
    uv run ckd-simulate --clinics             # one practice per on-disk data/clinics/ CSV
    uv run ckd-simulate --clinics --model xgboost
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math

import numpy as np
from flwr.common import (
    Code,
    EvaluateRes,
    FitRes,
    Parameters,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.strategy import FedAvg, FedXgbBagging

from client_app import _local_split, build_client_from_frame
from data import NUM_FEATURES, load_clinic_frames, load_partition, to_xy
from models import make_model
from models.fedxgb import XgbPractice
from server_app import weighted_and_worst
from task import compute_metrics

# Mirrors [tool.flwr.app.config]; overridable via CLI below.
DEFAULT_CONFIG = {
    "model": "logreg",
    "local-epochs": 2,
    "alpha": 0.5,
    "iid": False,
    "class-weight-balanced": True,
    "seed": 42,
}

_OK = Status(code=Code.OK, message="")


# ── FedAvg (logreg / mlp) ───────────────────────────────────────────────────


def _make_fedavg() -> FedAvg:
    """The same FedAvg the ServerApp builds — used here for its aggregate_fit/aggregate_evaluate."""
    return FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        evaluate_metrics_aggregation_fn=weighted_and_worst,
        fit_metrics_aggregation_fn=lambda _metrics: {},  # silence the "no fn provided" warning
    )


def fedavg(updates: list[tuple[list[np.ndarray], int]]) -> list[np.ndarray]:
    """Sample-weighted average via Flower's real FedAvg strategy (`aggregate_fit`).

    Kept as a small helper (used by the notebooks) so callers get genuine `flwr` aggregation from a
    simple ``[(weights, n), ...]`` input rather than constructing Flower message types themselves.
    """
    results = [
        (
            None,
            FitRes(status=_OK, parameters=ndarrays_to_parameters(params), num_examples=n,
                   metrics={}),
        )
        for params, n in updates
    ]
    aggregated, _ = _make_fedavg().aggregate_fit(1, results, [])
    return parameters_to_ndarrays(aggregated)


def _run_fedavg(frames: list, num_rounds: int, quiet: bool, config: dict) -> list[dict]:
    global_model = make_model(
        str(config["model"]),
        class_weight_balanced=bool(config["class-weight-balanced"]),
        seed=int(config["seed"]),
    )
    global_model.initialize(NUM_FEATURES)
    ndarrays = global_model.get_parameters()

    clients = [build_client_from_frame(df, pid, config) for pid, df in enumerate(frames)]
    strategy = _make_fedavg()

    history = []
    for rnd in range(1, num_rounds + 1):
        # fit: every practice trains locally, then Flower's FedAvg averages the updates.
        fit_results = [
            (
                None,
                FitRes(status=_OK, parameters=ndarrays_to_parameters(params),
                       num_examples=n, metrics=metrics),
            )
            for params, n, metrics in (client.fit(ndarrays, {}) for client in clients)
        ]
        aggregated, _ = strategy.aggregate_fit(rnd, fit_results, [])
        ndarrays = parameters_to_ndarrays(aggregated)

        # evaluate: per-client local validation, then Flower's dual-level aggregation.
        eval_results = [
            (None, EvaluateRes(status=_OK, loss=float(loss), num_examples=n, metrics=metrics))
            for loss, n, metrics in (client.evaluate(ndarrays, {}) for client in clients)
        ]
        history.append(_aggregate_and_print(strategy, rnd, eval_results, quiet))
    return history


# ── FedXgbBagging (xgboost) ─────────────────────────────────────────────────


def _run_fedxgb(frames: list, num_rounds: int, quiet: bool, config: dict) -> list[dict]:
    seed = int(config["seed"])
    practices = []
    for pid, df in enumerate(frames):
        X, y = to_xy(df)
        X_train, y_train, X_test, y_test = _local_split(X, y, seed, pid)
        practices.append(
            XgbPractice(
                X_train, y_train, X_test, y_test,
                num_local_round=int(config["local-epochs"]),
                class_weight_balanced=bool(config["class-weight-balanced"]),
                seed=seed,
            )
        )
    strategy = FedXgbBagging(evaluate_metrics_aggregation_fn=weighted_and_worst)
    global_model: bytes | None = None

    history = []
    for rnd in range(1, num_rounds + 1):
        # fit: each practice grows local trees; FedXgbBagging bags them into the global ensemble.
        fit_results = [
            (
                None,
                FitRes(
                    status=_OK,
                    parameters=Parameters(tensor_type="", tensors=[pr.local_trees(global_model)]),
                    num_examples=pr.num_examples, metrics={},
                ),
            )
            for pr in practices
        ]
        aggregated, _ = strategy.aggregate_fit(rnd, fit_results, [])
        global_model = aggregated.tensors[0]

        # evaluate: each practice scores the global ensemble on its local validation set.
        eval_results = []
        for pid, pr in enumerate(practices):
            metrics = compute_metrics(pr.y_val, pr.predict_proba(global_model))
            metrics["partition_id"] = float(pid)
            eval_results.append(
                (None, EvaluateRes(status=_OK, loss=1.0 - metrics["accuracy"],
                                   num_examples=len(pr.y_val), metrics=metrics))
            )
        history.append(_aggregate_and_print(strategy, rnd, eval_results, quiet))
    return history


# ── shared ──────────────────────────────────────────────────────────────────


def _aggregate_and_print(strategy, rnd: int, eval_results, quiet: bool) -> dict:
    """Run the strategy's real aggregate_evaluate (→ weighted_and_worst) and print the round line."""
    if quiet:  # aggregate_evaluate triggers weighted_and_worst's per-client prints; hide them
        with contextlib.redirect_stdout(io.StringIO()):
            _, agg = strategy.aggregate_evaluate(rnd, eval_results, [])
    else:
        print(f"[ROUND {rnd}]")
        _, agg = strategy.aggregate_evaluate(rnd, eval_results, [])
    auc, auc_w = agg.get("auc", math.nan), agg.get("auc_worst", math.nan)
    sens = agg.get("sensitivity", math.nan)
    print(f"  round {rnd:>2}: AUROC={auc:.3f} (worst practice {auc_w:.3f})  sensitivity={sens:.3f}")
    return agg


def _load_frames(num_practices: int, config: dict, clinics_dir) -> tuple[list, str]:
    """Per-practice DataFrames + a one-line source description.

    `clinics_dir` is None for the simulated Dirichlet/IID partition of the flat CSV; `True` for the
    default `data/clinics/`; or a path to a specific clinics directory (one practice per clinic CSV).
    """
    if clinics_dir is not None:
        frames = load_clinic_frames(None if clinics_dir is True else clinics_dir)
        return frames, f"on-disk clinics (data/clinics, {len(frames)} practices)"
    seed = int(config["seed"])
    frames = [
        load_partition(i, num_practices, alpha=float(config["alpha"]),
                       iid=bool(config["iid"]), seed=seed)
        for i in range(num_practices)
    ]
    return frames, f"partitioned CSV ({num_practices} practices, alpha={config['alpha']} iid={config['iid']})"


def run_simulation(
    num_practices: int, num_rounds: int, quiet: bool, *, clinics_dir=None, **overrides
) -> list[dict]:
    config = {**DEFAULT_CONFIG, **overrides}
    model = str(config["model"])
    frames, source = _load_frames(num_practices, config, clinics_dir)
    strategy_name = "FedXgbBagging" if model == "xgboost" else "FedAvg"
    print(
        f"Flower {strategy_name} (in-process) | model={model} source={source} "
        f"rounds={num_rounds}\n" + "-" * 64
    )
    runner = _run_fedxgb if model == "xgboost" else _run_fedavg
    # Full per-round history (one aggregated-metrics dict per round) so callers/notebooks can plot
    # learning curves; `main()` ignores the return value.
    return runner(frames, num_rounds, quiet, config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Flower federated CKD simulation (Ray-free)")
    parser.add_argument("--practices", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--model", default="logreg", choices=["logreg", "mlp", "xgboost"])
    parser.add_argument("--alpha", type=float, default=DEFAULT_CONFIG["alpha"])
    parser.add_argument("--local-epochs", type=int, default=DEFAULT_CONFIG["local-epochs"])
    parser.add_argument("--iid", action="store_true", help="IID even split instead of Dirichlet")
    parser.add_argument(
        "--clinics", action="store_true",
        help="train on the on-disk per-clinic CSVs in data/clinics/ (one practice per clinic, "
             "the natural boundary) instead of partitioning the flat CSV. Run ckd-clinics first.",
    )
    parser.add_argument(
        "--clinics-dir", default=None,
        help="directory of clinic CSVs to use (implies --clinics)",
    )
    parser.add_argument("--quiet", action="store_true", help="hide per-client metric lines")
    args = parser.parse_args()

    # None = partition the flat CSV; True = default data/clinics/; a path = that clinics dir.
    clinics_dir = args.clinics_dir if args.clinics_dir is not None else (True if args.clinics else None)

    run_simulation(
        num_practices=args.practices,
        num_rounds=args.rounds,
        quiet=args.quiet,
        clinics_dir=clinics_dir,
        model=args.model,
        alpha=args.alpha,
        iid=args.iid,
        **{"local-epochs": args.local_epochs},
    )


if __name__ == "__main__":
    main()
