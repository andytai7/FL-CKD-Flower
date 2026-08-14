"""Flower ServerApp — FedAvg with dual-level metric logging (Message API, flwr 1.33).

Per CLAUDE.md §5, a non-IID federation can look strong on the global aggregate while collapsing on
an outlier practice. So the evaluate-metrics aggregator reports BOTH the sample-weighted global
mean AND the worst (min) client.

Privacy note (CLAUDE.md §9, layer L5): naming a practice alongside its AUROC and cohort size is
itself a disclosure channel to whoever operates the SuperLink. Per-practice lines are therefore
suppressed for cohorts below `min-cohort-size`, and can be switched to fully anonymous reporting
with `metric-privacy = true` — which keeps the worst-practice number (the thing rule 5 exists to
protect) while dropping the identity that makes it re-identifying.
"""

from __future__ import annotations

import math

from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord, RecordDict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from data import NUM_FEATURES
from models import make_model

app = ServerApp()

_AGG_KEYS = ("accuracy", "sensitivity", "auc")

# Defaults for the L5 metric-hygiene layer; overridden from run_config in server_fn.
_METRIC_PRIVACY = {"enabled": False, "min_cohort_size": 0}


def configure_metric_privacy(*, enabled: bool, min_cohort_size: int) -> None:
    """Set the disclosure policy for per-practice metric logging (privacy layer L5)."""
    _METRIC_PRIVACY["enabled"] = enabled
    _METRIC_PRIVACY["min_cohort_size"] = int(min_cohort_size)


def weighted_and_worst(records: list[RecordDict], weighting_key: str = "num-examples") -> MetricRecord:
    """Aggregate per-client evaluate metrics into global weighted means + worst-client values.

    Signature matches flwr 1.33's `evaluate_metrics_aggr_fn`: it receives the reply RecordDicts and
    the key to weight by.
    """
    rows = []
    for record in records:
        metrics = next(iter(record.metric_records.values()))
        rows.append(dict(metrics))

    out = MetricRecord()
    for key in _AGG_KEYS:
        pairs = [
            (float(m[weighting_key]), float(m[key]))
            for m in rows
            if key in m and not math.isnan(float(m[key]))
        ]
        if pairs:
            total = sum(n for n, _ in pairs)
            out[key] = sum(n * v for n, v in pairs) / total
            out[f"{key}_worst"] = min(v for _, v in pairs)

    _log_per_practice(rows, weighting_key)
    return out


def _log_per_practice(rows: list[dict], weighting_key: str) -> None:
    """Dual-level visibility, subject to the L5 disclosure policy."""
    anonymous = _METRIC_PRIVACY["enabled"]
    floor = _METRIC_PRIVACY["min_cohort_size"]
    suppressed = 0

    for m in sorted(rows, key=lambda r: r.get("partition-id", -1)):
        n = int(float(m.get(weighting_key, 0)))
        if n < floor:
            suppressed += 1
            continue
        auc = float(m.get("auc", float("nan")))
        sens = float(m.get("sensitivity", float("nan")))
        if anonymous:
            print(f"      practice ··: n={'·' * 4} auc={auc:.3f} sensitivity={sens:.3f}")
        else:
            pid = int(float(m.get("partition-id", -1)))
            print(f"      practice {pid:>2}: n={n:<4} auc={auc:.3f} sensitivity={sens:.3f}")

    if suppressed:
        print(f"      [{suppressed} practice(s) suppressed: cohort < {floor} rows]")


@app.main()
def main(grid: Grid, context: Context) -> None:
    rc = context.run_config
    num_rounds = int(rc["num-server-rounds"])

    configure_metric_privacy(
        enabled=bool(rc.get("metric-privacy", False)),
        min_cohort_size=int(rc.get("min-cohort-size", 0)),
    )

    # Initialize global parameters from a fresh model so all clients start from the same shapes.
    model = make_model(
        str(rc["model"]),
        class_weight_balanced=bool(rc["class-weight-balanced"]),
        seed=int(rc["seed"]),
    )
    model.initialize(NUM_FEATURES)
    initial_arrays = ArrayRecord(model.get_parameters())

    strategy = FedAvg(
        fraction_train=float(rc["fraction-fit"]),
        fraction_evaluate=1.0,
        min_train_nodes=1,
        min_evaluate_nodes=1,
        min_available_nodes=int(rc["num-practices"]),
        evaluate_metrics_aggr_fn=weighted_and_worst,
    )

    strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=num_rounds,
        train_config=ConfigRecord({"local-epochs": int(rc["local-epochs"])}),
    )
