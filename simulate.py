"""In-process federated simulation driven by Flower.ai's **real** strategies (flwr 1.33).

`flwr run .` runs the full Ray simulation engine with the real `ServerApp`/`ClientApp` and is the
right tool for an end-to-end check (see CLAUDE.md §6). This runner exists for a different job:
**fast, reproducible, per-round benchmarking** — it returns the full metrics history each round so
callers can plot learning curves and compare protocols, without Ray process startup between runs.

Everything that matters is still genuine `flwr` code — this is not a reimplementation of
federation:

- the logistic regression is aggregated by the real `flwr.serverapp.strategy.FedAvg`, by calling
  its `aggregate_train` / `aggregate_evaluate` on real `Message` objects each round;
- metric aggregation is the real `server_app.weighted_and_worst`, wired as the strategy's
  `evaluate_metrics_aggr_fn`, exactly as the ServerApp does it;
- the protocol benchmark uses Flower's built-in `FedAvg` and `FedProx` plus `FedMosaic`, a
  `Strategy` subclass in `models/protocols/` — Flower's own extension point (CLAUDE.md §0 rule 8).

This runner is logreg-only: logistic regression is the only model the project trains.

    uv run ckd-simulate                       # 12 practices, 20 rounds, logreg (FedAvg), non-IID
    uv run ckd-simulate --clinics             # one practice per on-disk data/clinics/ CSV
    uv run ckd-simulate --protocol fedprox --clinics     # protocol benchmark
    uv run ckd-simulate --protocol fedmosaic --clinics
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math

import numpy as np
from flwr.serverapp.strategy import FedAvg

from client_app import build_client_from_frame
from data import NUM_FEATURES, load_clinic_frames, load_partition
from messages import evaluate_reply, hushed, train_reply
from models import LogRegModel
from models.protocols import BASELINES, PROTOCOLS
from server_app import configure_metric_privacy, weighted_and_worst

# Mirrors [tool.flwr.app.config]; overridable via CLI below.
DEFAULT_CONFIG = {
    "local-epochs": 2,
    "alpha": 0.5,
    "iid": False,
    "class-weight-balanced": True,
    "seed": 42,
}

# The three protocols under test, plus the two reference baselines needed to read them.
PROTOCOL_CHOICES = (*PROTOCOLS, *BASELINES)


# ── Strategy helpers ────────────────────────────────────────────────────────


def _make_fedavg() -> FedAvg:
    """The same FedAvg the ServerApp builds — used here for its aggregate_* methods."""
    return FedAvg(
        fraction_train=1.0,
        fraction_evaluate=1.0,
        evaluate_metrics_aggr_fn=weighted_and_worst,
    )


def fedavg(updates: list[tuple[list[np.ndarray], int]]) -> list[np.ndarray]:
    """Sample-weighted average via Flower's real FedAvg strategy (`aggregate_train`).

    Kept as a small helper (used by the notebooks) so callers get genuine `flwr` aggregation from a
    simple ``[(weights, n), ...]`` input rather than constructing Flower message types themselves.
    """
    replies = [train_reply(params, n) for params, n in updates]
    with contextlib.redirect_stdout(io.StringIO()):
        arrays, _ = _make_fedavg().aggregate_train(1, replies)
    return arrays.to_numpy_ndarrays()


# ── FedAvg (logreg) ─────────────────────────────────────────────────────────


def _run_fedavg(frames: list, num_rounds: int, quiet: bool, config: dict) -> list[dict]:
    global_model = LogRegModel(
        class_weight_balanced=bool(config["class-weight-balanced"]),
        seed=int(config["seed"]),
    )
    global_model.initialize(NUM_FEATURES)
    ndarrays = global_model.get_parameters()

    clients = [build_client_from_frame(df, pid, config) for pid, df in enumerate(frames)]
    strategy = _make_fedavg()

    history = []
    for rnd in range(1, num_rounds + 1):
        # train: every practice trains locally, then Flower's FedAvg averages the updates.
        replies = []
        for client in clients:
            params, n = client.fit(ndarrays)
            replies.append(train_reply(params, n))
        with hushed():
            arrays, _ = strategy.aggregate_train(rnd, replies)
        ndarrays = arrays.to_numpy_ndarrays()

        # evaluate: per-client local validation, then Flower's dual-level aggregation.
        eval_replies = []
        for client in clients:
            metrics, n = client.evaluate(ndarrays)
            eval_replies.append(evaluate_reply(metrics, n, client.partition_id))
        history.append(_aggregate_and_print(strategy, rnd, eval_replies, quiet))
    return history


# ── shared ──────────────────────────────────────────────────────────────────


def _aggregate_and_print(strategy, rnd: int, eval_replies, quiet: bool) -> dict:
    """Run the strategy's real aggregate_evaluate (→ weighted_and_worst) and print the round line."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        agg = strategy.aggregate_evaluate(rnd, eval_replies)
    metrics = dict(agg) if agg else {}

    if not quiet:
        print(f"[ROUND {rnd}]")
        # Re-emit only weighted_and_worst's per-practice lines, not Flower's INFO chatter.
        for line in buf.getvalue().splitlines():
            if "practice" in line or "suppressed" in line:
                print(line)

    auc = metrics.get("auc", math.nan)
    auc_w = metrics.get("auc_worst", math.nan)
    sens = metrics.get("sensitivity", math.nan)
    print(f"  round {rnd:>2}: AUROC={auc:.3f} (worst practice {auc_w:.3f})  sensitivity={sens:.3f}")
    return metrics


def _load_frames(num_practices: int, config: dict, clinics_dir) -> tuple[list, str]:
    """Per-practice DataFrames + a one-line source description.

    `clinics_dir` is None for the simulated Dirichlet/IID partition of the flat CSV; `True` for the
    default `data/clinics/`; or a path to a specific clinics directory (one practice per clinic CSV).
    """
    if clinics_dir is not None:
        frames = load_clinic_frames(None if clinics_dir is True else clinics_dir)
        return frames, f"on-disk clinics ({len(frames)} practices)"
    seed = int(config["seed"])
    frames = [
        load_partition(i, num_practices, alpha=float(config["alpha"]),
                       iid=bool(config["iid"]), seed=seed)
        for i in range(num_practices)
    ]
    return frames, (
        f"partitioned CSV ({num_practices} practices, alpha={config['alpha']} iid={config['iid']})"
    )


def run_simulation(
    num_practices: int,
    num_rounds: int,
    quiet: bool,
    *,
    clinics_dir=None,
    protocol: str | None = None,
    **overrides,
) -> list[dict]:
    """Run one federation.

    `protocol=None` runs the production path — the sklearn logistic regression under `FedAvg`.
    Passing a `protocol` runs the logistic-regression protocol benchmark instead, where every
    comparator shares one local learner so only the protocol varies.
    """
    config = {**DEFAULT_CONFIG, **overrides}
    frames, source = _load_frames(num_practices, config, clinics_dir)

    if protocol is not None:
        from models.protocols import run_protocol

        print(
            f"Flower protocol={protocol} (in-process) | model=logreg source={source} "
            f"rounds={num_rounds}\n" + "-" * 64
        )
        return run_protocol(protocol, frames, num_rounds, quiet, config, _aggregate_and_print)

    print(
        f"Flower FedAvg (in-process) | model=logreg source={source} "
        f"rounds={num_rounds}\n" + "-" * 64
    )
    # Full per-round history (one aggregated-metrics dict per round) so callers/notebooks can plot
    # learning curves; `main()` ignores the return value.
    return _run_fedavg(frames, num_rounds, quiet, config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Flower federated CKD simulation (in-process)")
    parser.add_argument("--practices", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument(
        "--protocol", default=None, choices=list(PROTOCOL_CHOICES),
        help=f"run the protocol benchmark instead of the production model path. Protocols: "
             f"{', '.join(PROTOCOLS)}. Reference baselines: {', '.join(BASELINES)}. The "
             f"logistic-regression comparators share one local learner so only the protocol varies.",
    )
    parser.add_argument("--alpha", type=float, default=DEFAULT_CONFIG["alpha"])
    parser.add_argument("--local-epochs", type=int, default=DEFAULT_CONFIG["local-epochs"])
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
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
    parser.add_argument(
        "--metric-privacy", action="store_true",
        help="privacy layer L5: report per-practice metrics anonymously (see docs/PRIVACY.md §1)",
    )
    parser.add_argument(
        "--min-cohort-size", type=int, default=0,
        help="privacy layer L5: suppress per-practice lines for cohorts smaller than this",
    )
    args = parser.parse_args()

    configure_metric_privacy(
        enabled=args.metric_privacy, min_cohort_size=args.min_cohort_size
    )

    # None = partition the flat CSV; True = default data/clinics/; a path = that clinics dir.
    clinics_dir = args.clinics_dir if args.clinics_dir is not None else (args.clinics or None)

    run_simulation(
        num_practices=args.practices,
        num_rounds=args.rounds,
        quiet=args.quiet,
        clinics_dir=clinics_dir,
        protocol=args.protocol,
        alpha=args.alpha,
        iid=args.iid,
        seed=args.seed,
        **{"local-epochs": args.local_epochs},
    )


if __name__ == "__main__":
    main()
