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
from logging import INFO

from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord, RecordDict
from flwr.common.logger import log
from flwr.compat.common.recorddict_compat import arrayrecord_to_parameters
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

    if bool(rc.get("secure-aggregation", False)):
        _run_secure_aggregation(grid, context, initial_arrays, num_rounds)
        return

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


def _run_secure_aggregation(grid: Grid, context: Context, initial_arrays, num_rounds: int) -> None:
    """Privacy layer L3: run the round under SecAgg+, so the server can only open the sum.

    This is the configuration that answers the question counsel put to the project — *does the
    aggregating party ever hold one practice's update in isolation?* Under SecAgg+ it does not:
    each practice masks its update with pairwise secrets that cancel only once enough contributions
    are combined, so the server obtains the aggregate and never an individual vector.

    **Why this is a second code path rather than a flag.** In flwr 1.33 SecAgg+ ships only in the
    legacy namespaces: `secaggplus_mod` is absent from `flwr.clientapp.mod`, and
    `SecAggPlusWorkflow.__call__` demands a `LegacyContext`, so it cannot compose with
    `strategy.start()`. It is reachable — that is what this function does — by driving
    `DefaultWorkflow` with a `LegacyContext` from inside this modern `ServerApp`. The client side
    switches on the same `secure-aggregation` config key, and its handlers accept the legacy record
    shape (`fitins.parameters`) as well as this project's `arrays` — see `client_app.py`.

    Three limits that belong next to any claim made about this path:

    1. **Semi-honest threat model.** The guarantee holds against a server that follows the protocol
       but inspects what it receives. It is not a guarantee against a server that deviates — for
       instance by running a round with a single practice, whose "aggregate" is that practice.
       `min_fit_clients` below is the control that makes such a round fail rather than succeed.
    2. **Participation stays visible.** The server always learns which practices took part, and the
       aggregate. Only the individual contribution is hidden.
    3. **The aggregate is still model parameters.** SecAgg+ answers who may see one practice's
       update; it does not by itself make the released model anonymous. That is what DP and the
       leakage audit are for.

    It also cannot protect the XGBoost path at all: `FedXgbBagging` transmits serialized decision
    trees, and there is no numeric vector to mask. A SecAgg-protected deployment is necessarily a
    logistic-regression deployment.
    """
    from flwr.server import LegacyContext, ServerConfig
    from flwr.server.strategy import FedAvg as LegacyFedAvg
    from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow

    rc = context.run_config
    num_practices = int(rc["num-practices"])
    num_shares = int(rc.get("secagg-num-shares", 3))
    threshold = int(rc.get("secagg-reconstruction-threshold", 2))

    log(
        INFO,
        "SecAgg+ enabled: %s shares, reconstruction threshold %s, %s practices required per round",
        num_shares,
        threshold,
        num_practices,
    )

    legacy = LegacyContext(
        context=context,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=LegacyFedAvg(
            fraction_fit=float(rc["fraction-fit"]),
            fraction_evaluate=1.0,
            # A round with too few practices would defeat the masking, so it must not be allowed
            # to run at all. This is the mitigation for limit (1) above.
            min_fit_clients=num_practices,
            min_evaluate_clients=1,
            min_available_clients=num_practices,
            initial_parameters=arrayrecord_to_parameters(initial_arrays, keep_input=True),
            evaluate_metrics_aggregation_fn=_legacy_weighted_and_worst,
        ),
    )

    workflow = DefaultWorkflow(
        fit_workflow=SecAggPlusWorkflow(
            num_shares=num_shares,
            reconstruction_threshold=threshold,
        )
    )
    workflow(grid, legacy)


def _legacy_weighted_and_worst(results: list[tuple[int, dict]]) -> dict:
    """`weighted_and_worst` for the legacy strategy's aggregation signature.

    The legacy path hands `[(num_examples, metrics), ...]` rather than reply RecordDicts, so this
    adapts the shape and delegates. The dual-level policy itself (CLAUDE.md §0 rule 5) — global
    sample-weighted mean plus worst practice, and the L5 disclosure controls — is defined once, in
    `weighted_and_worst`, and is not duplicated here.
    """
    records = [
        RecordDict({"metrics": MetricRecord({"num-examples": float(n), **{
            k: float(v) for k, v in m.items()
        }})})
        for n, m in results
    ]
    return dict(weighted_and_worst(records))
