"""Flower ServerApp — FedAvg with dual-level metric logging.

Per CLAUDE.md §5, a non-IID federation can look strong on the global aggregate while collapsing on
an outlier practice. So the evaluate-metrics aggregator reports BOTH the sample-weighted global
mean AND the worst (min) client, and logs every client's metrics each round.
"""

from __future__ import annotations

import math

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from data import NUM_FEATURES
from models import make_model

_AGG_KEYS = ("accuracy", "sensitivity", "auc")


def weighted_and_worst(metrics: list[tuple[int, Metrics]]) -> Metrics:
    """Aggregate per-client evaluate metrics into global weighted means + worst-client values."""
    out: Metrics = {}
    for key in _AGG_KEYS:
        pairs = [
            (n, float(m[key]))
            for n, m in metrics
            if key in m and not math.isnan(float(m[key]))
        ]
        if pairs:
            total = sum(n for n, _ in pairs)
            out[key] = sum(n * v for n, v in pairs) / total
            out[f"{key}_worst"] = min(v for _, v in pairs)

    # Per-client visibility (dual-level logging).
    for n, m in sorted(metrics, key=lambda t: t[1].get("partition_id", -1)):
        pid = int(m.get("partition_id", -1))
        auc = m.get("auc", float("nan"))
        sens = m.get("sensitivity", float("nan"))
        print(f"      practice {pid:>2}: n={n:<4} auc={auc:.3f} sensitivity={sens:.3f}")
    return out


def server_fn(context: Context) -> ServerAppComponents:
    rc = context.run_config
    num_rounds = int(rc["num-server-rounds"])
    fraction_fit = float(rc["fraction-fit"])
    num_practices = int(rc["num-practices"])

    # Initialize global parameters from a fresh model so all clients start from the same shapes.
    model = make_model(
        str(rc["model"]),
        class_weight_balanced=bool(rc["class-weight-balanced"]),
        seed=int(rc["seed"]),
    )
    model.initialize(NUM_FEATURES)
    initial_parameters = ndarrays_to_parameters(model.get_parameters())

    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_fit_clients=max(1, int(num_practices * fraction_fit)),
        min_evaluate_clients=num_practices,
        min_available_clients=num_practices,
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=weighted_and_worst,
    )
    return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=num_rounds))


app = ServerApp(server_fn=server_fn)
