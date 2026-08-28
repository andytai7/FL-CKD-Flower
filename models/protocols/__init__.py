"""The protocol benchmark (CLAUDE.md §4).

**The three protocols under test:**

| protocol    | strategy                                      | model    | what crosses the wire          |
|-------------|-----------------------------------------------|----------|--------------------------------|
| `fedprox`   | `flwr.serverapp.strategy.FedProx`  (built-in) | logreg   | model coefficients             |
| `fedxgb`    | `flwr.serverapp.strategy.FedXgbBagging` (built-in) | xgboost  | serialized decision trees |
| `fedmosaic` | `models.protocols.fedmosaic.FedMosaic`        | logreg   | predictions + expertise on `U` |

**Plus two reference baselines**, which are not protocols but are required to interpret the
protocols:

| baseline | why it must be there                                                              |
|----------|-----------------------------------------------------------------------------------|
| `local`  | no collaboration at all. The FedMosaic paper's central empirical finding is that local training is a strong baseline that most personalized-FL methods fail to beat — without it, "federation helped" is unfalsifiable. |
| `fedavg` | plain weight averaging. FedProx *is* FedAvg + a proximal term, so this isolates what the μ term actually bought. |

Every logistic-regression comparator shares one local learner, splits, scaler and seeds — only the
protocol varies (immutable rule 6). `fedxgb` is a different model class and is reported alongside,
not as a head-to-head row.
"""

from __future__ import annotations

import time

import numpy as np
from flwr.serverapp.strategy import FedAvg, FedProx

from client_app import _local_split
from data import to_xy
from data.synthesize import generate_public_cohort
from messages import evaluate_reply, hushed, train_reply
from server_app import weighted_and_worst
from task import compute_metrics, fit_scaler

from .common import LogRegLocal, from_flower_arrays, init_weights, to_flower_arrays
from .fedmosaic import FedMosaic, MosaicPractice, build_reply as mosaic_reply, uplink_bits

# The three protocols under test.
PROTOCOLS = ("fedprox", "fedxgb", "fedmosaic")
# Reference comparators — needed to interpret the protocols, not competing with them.
BASELINES = ("local", "fedavg")
# `fedxgb` is dispatched by simulate.py to the FedXgbBagging tree path, not to run_protocol.
LOGREG_RUNNABLE = ("local", "fedavg", "fedprox", "fedmosaic")

# Local optimiser settings shared by every protocol, so the comparison isn't confounded.
LEARNING_RATE = 0.5
PROXIMAL_MU = 0.1     # FedProx μ; 0 would make it identical to FedAvg
PUBLIC_COHORT_SIZE = 400


def _prepare(frames: list, config: dict) -> list[LogRegLocal]:
    """Local split + local scaler per practice — identical to `build_client_from_frame`."""
    seed = int(config["seed"])
    locals_ = []
    for pid, df in enumerate(frames):
        X, y = to_xy(df)
        X_train, y_train, X_test, y_test = _local_split(X, y, seed, pid)
        scaler = fit_scaler(X_train)
        locals_.append(
            LogRegLocal(
                scaler.transform(X_train), y_train,
                scaler.transform(X_test), y_test,
                class_weight_balanced=bool(config["class-weight-balanced"]),
            )
        )
    return locals_


def _public_features(config: dict) -> np.ndarray:
    """The shared public cohort, scaled with a scaler fit on the public data itself.

    Every practice holds the identical copy of `U`, so scaling it locally would make each
    practice's predictions incomparable — which is exactly what the consensus depends on.
    """
    cohort = generate_public_cohort(PUBLIC_COHORT_SIZE, base_seed=int(config["seed"]))
    X_pub, _ = to_xy(cohort)
    return fit_scaler(X_pub).transform(X_pub)


def run_protocol(
    name: str, frames: list, num_rounds: int, quiet: bool, config: dict, report
) -> list[dict]:
    """Run one protocol for `num_rounds` and return its per-round aggregated metrics."""
    locals_ = _prepare(frames, config)
    epochs = int(config["local-epochs"])

    if name == "local":
        return _run_local(locals_, num_rounds, quiet, epochs, report)
    if name in ("fedavg", "fedprox"):
        return _run_weight_sharing(name, locals_, num_rounds, quiet, epochs, report)
    if name == "fedmosaic":
        return _run_fedmosaic(locals_, frames, config, num_rounds, quiet, epochs, report)
    raise ValueError(
        f"Unknown logistic-regression protocol {name!r}; choose from {LOGREG_RUNNABLE}. "
        "('fedxgb' is the tree path — simulate.py routes it to FedXgbBagging.)"
    )


def _evaluate(scores_and_locals, report, strategy, rnd: int, quiet: bool, extra: dict) -> dict:
    replies = [
        evaluate_reply(compute_metrics(loc.y_test, scores), len(loc.y_test), pid)
        for pid, (scores, loc) in enumerate(scores_and_locals)
    ]
    metrics = report(strategy, rnd, replies, quiet)
    metrics.update(extra)
    return metrics


def _run_local(locals_, num_rounds, quiet, epochs, report) -> list[dict]:
    """No collaboration: each practice trains on its own data only."""
    strategy = FedAvg(evaluate_metrics_aggr_fn=weighted_and_worst)
    weights = [init_weights(loc.n_features) for loc in locals_]
    history = []
    for rnd in range(1, num_rounds + 1):
        started = time.perf_counter()
        weights = [
            loc.local_sgd(w, epochs=epochs, lr=LEARNING_RATE)
            for loc, w in zip(locals_, weights)
        ]
        pairs = [(loc.test_scores(w), loc) for loc, w in zip(locals_, weights)]
        history.append(_evaluate(pairs, report, strategy, rnd, quiet, {
            "uplink-bits-per-client": 0.0,
            "round-seconds": time.perf_counter() - started,
        }))
    return history


def _run_weight_sharing(name, locals_, num_rounds, quiet, epochs, report) -> list[dict]:
    """FedAvg / FedProx — Flower's built-in strategies, coefficients on the wire."""
    # Dual-level metrics are mandatory (immutable rule 5), so every strategy gets the project's
    # aggregator rather than Flower's default weighted mean.
    strategy = (
        FedProx(proximal_mu=PROXIMAL_MU, evaluate_metrics_aggr_fn=weighted_and_worst)
        if name == "fedprox"
        else FedAvg(evaluate_metrics_aggr_fn=weighted_and_worst)
    )
    w_global = init_weights(locals_[0].n_features)
    mu = PROXIMAL_MU if name == "fedprox" else 0.0
    # 32-bit floats for coef_ + intercept.
    uplink = 32 * (locals_[0].n_features + 1)

    history = []
    for rnd in range(1, num_rounds + 1):
        started = time.perf_counter()
        replies = []
        for loc in locals_:
            w_local = loc.local_sgd(
                w_global, epochs=epochs, lr=LEARNING_RATE, proximal_mu=mu, w_global=w_global
            )
            replies.append(train_reply(to_flower_arrays(w_local), loc.num_examples))
        with hushed():
            arrays, _ = strategy.aggregate_train(rnd, replies)
        w_global = from_flower_arrays(arrays.to_numpy_ndarrays())

        pairs = [(loc.test_scores(w_global), loc) for loc in locals_]
        history.append(_evaluate(pairs, report, strategy, rnd, quiet, {
            "uplink-bits-per-client": float(uplink),
            "round-seconds": time.perf_counter() - started,
        }))
    return history


def _run_fedmosaic(locals_, frames, config, num_rounds, quiet, epochs, report) -> list[dict]:
    """FedMosaic — predictions + expertise on a shared public cohort; no model ever shared."""
    X_public = _public_features(config)
    strategy = FedMosaic(evaluate_metrics_aggr_fn=weighted_and_worst)
    practices = [
        MosaicPractice(loc, X_public, lr=LEARNING_RATE, epochs=epochs) for loc in locals_
    ]
    uplink = float(uplink_bits(len(X_public)))

    history = []
    for rnd in range(1, num_rounds + 1):
        started = time.perf_counter()
        alphas = [pr.local_step() for pr in practices]           # lines 3-7

        replies = []
        for pr in practices:                                     # lines 9-11
            labels, expertise = pr.share()
            replies.append(mosaic_reply(labels, expertise, pr.num_examples))
        with hushed():
            arrays, train_metrics = strategy.aggregate_train(rnd, replies)  # lines 14-18
        consensus = arrays.to_numpy_ndarrays()[0]
        for pr in practices:                                     # line 12
            pr.adopt(consensus)

        pairs = [(pr.test_scores(), pr.local) for pr in practices]
        history.append(_evaluate(pairs, report, strategy, rnd, quiet, {
            "uplink-bits-per-client": uplink,
            "round-seconds": time.perf_counter() - started,
            "alpha-mean": float(np.mean(alphas)),
            "consensus-agreement": float(dict(train_metrics).get("consensus-agreement", np.nan)),
        }))
    return history
