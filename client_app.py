"""Flower ClientApp — one GP practice (Message API, flwr 1.33).

Model-agnostic: it asks the factory for whatever architecture the run is configured with, loads its
data partition, splits/scales locally (data never leaves the client), and runs warm-started local
training each round.

Message API contract (CLAUDE.md §6):
- incoming  `msg.content["arrays"]`  -> global model weights
- outgoing  `{"arrays": ArrayRecord, "metrics": MetricRecord}` where the MetricRecord carries
  `num-examples` — the key FedAvg weights the average by.

## The two privacy mechanisms that change what leaves the practice

Both are ordinary run config, OFF by default so the research baseline stays interpretable:

    local-dp-epsilon = <eps>    clip + Gaussian-noise the update INSIDE the SuperNode, before it
                                is transmitted. The only mechanism here that changes what the
                                *aggregating party receives*.
    secure-aggregation = true   join SecAgg+ masking, so the server can open only the sum.

`ClientApp` takes its mods at **construction** time and the app is built at import, before any
`run_config` exists — so each is installed as a small dispatching mod that reads the run config
when it is *called*. That keeps them configurable the ordinary way instead of through environment
variables, which matters because the simulation engine runs ClientApps in separate Ray worker
processes that do not inherit the launching shell's environment.

A SecAgg+ round is driven by the legacy `DefaultWorkflow` and therefore arrives in the legacy record
shape (`fitins.parameters`) rather than this module's `arrays`. The handlers below accept either,
translating through Flower's own compat bridge.
"""

from __future__ import annotations

import numpy as np
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from data import load_partition, load_practice_frame, to_xy
from models import make_model
from task import compute_metrics, fit_scaler

# δ is conventionally set below 1/n (CLAUDE.md §9 / privacy.py); the pilot cohort is ~3.5k patients.
LOCAL_DP_DELTA = 1e-5
LOCAL_DP_CLIPPING_NORM = 1.0


class CKDPractice:
    """One practice's local state: its data, its scaler, and its model."""

    def __init__(self, model, X_train, y_train, X_test, y_test, local_epochs, partition_id):
        self.model = model
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test
        self.local_epochs = local_epochs
        self.partition_id = partition_id

    def fit(self, ndarrays: list[np.ndarray]) -> tuple[list[np.ndarray], int]:
        self.model.set_parameters(ndarrays)
        self.model.fit(self.X_train, self.y_train, epochs=self.local_epochs)
        return self.model.get_parameters(), len(self.X_train)

    def evaluate(self, ndarrays: list[np.ndarray]) -> tuple[dict, int]:
        self.model.set_parameters(ndarrays)
        y_score = self.model.predict_proba(self.X_test)
        return compute_metrics(self.y_test, y_score), len(self.X_test)


def _local_split(X, y, seed, partition_id):
    """Reproducible local 80/20 split that always leaves >=1 train and >=1 test row."""
    n = len(X)
    rng = np.random.default_rng(seed + partition_id)
    perm = rng.permutation(n)
    X, y = X[perm], y[perm]
    if n < 2:
        return X, y, X, y
    split = min(n - 1, max(1, int(n * 0.8)))
    return X[:split], y[:split], X[split:], y[split:]


def build_client_from_frame(df, partition_id: int, run_config) -> CKDPractice:
    """Preprocess one practice's DataFrame locally and build its client (data never shared).

    The data-source-agnostic core: callers supply the rows (a Dirichlet partition of the flat CSV,
    one on-disk clinic file, or a FHIR query result) and this does the local split + scaling +
    model build identically. This is the ONLY client constructor — see SKILL.md.
    """
    seed = int(run_config["seed"])
    X, y = to_xy(df)
    X_train, y_train, X_test, y_test = _local_split(X, y, seed, partition_id)

    scaler = fit_scaler(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    model = make_model(
        str(run_config["model"]),
        class_weight_balanced=bool(run_config["class-weight-balanced"]),
        seed=seed,
    )
    model.initialize(X_train.shape[1])

    return CKDPractice(
        model, X_train, y_train, X_test, y_test, int(run_config["local-epochs"]), partition_id
    )


def build_client(context: Context) -> CKDPractice:
    """Build this node's client from whichever data source the run is configured with.

    `data-source = "csv"` (default) partitions the flat synthetic CSV; `"fhir"` queries this
    practice's own FHIR server. Either way the rows are loaded locally and never transmitted.
    """
    run_config = context.run_config
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])

    if str(run_config.get("data-source", "csv")) == "fhir":
        df = load_practice_frame(str(context.node_config["fhir-base-url"]))
    else:
        df = load_partition(
            partition_id,
            num_partitions,
            alpha=float(run_config["alpha"]),
            iid=bool(run_config["iid"]),
            seed=int(run_config["seed"]),
        )
    return build_client_from_frame(df, partition_id, run_config)


def _local_dp_dispatch(msg: Message, ctx: Context, call_next):
    """Privacy layer L4-local: noise the update *before it leaves the practice*.

    Local DP is the only mechanism here that changes what the aggregating party receives — central
    DP noises during aggregation, so the server sees every un-noised update first. Enabled by
    setting `local-dp-epsilon` in the run config; 0 (the default) means off.

    Why a dispatching mod rather than `ClientApp(mods=[LocalDpMod(...)])`: mods are attached at
    **construction** time, and the app object is built at import, before any run config exists. A
    mod that reads `ctx.run_config` when it is *called* is config-driven in the ordinary way and
    needs no environment variable — which also means it survives the SuperNode process boundary
    (the simulation engine runs ClientApps in separate Ray workers that do not inherit the
    launching shell's environment).

    `LocalDpMod` itself does the clipping and the noise; neither is reimplemented (rule 1).
    """
    epsilon = float(ctx.run_config.get("local-dp-epsilon", 0.0) or 0.0)
    if epsilon <= 0:
        return call_next(msg, ctx)

    from flwr.clientapp.mod import LocalDpMod

    mod = LocalDpMod(
        clipping_norm=LOCAL_DP_CLIPPING_NORM,
        # Updates are clipped to the clipping norm, which is therefore what bounds sensitivity.
        sensitivity=LOCAL_DP_CLIPPING_NORM,
        epsilon=epsilon,
        delta=LOCAL_DP_DELTA,
    )
    return mod(msg, ctx, call_next)


def _secagg_dispatch(msg: Message, ctx: Context, call_next):
    """Privacy layer L3: join SecAgg+ masking when the run is configured for it.

    Same reasoning as `_local_dp_dispatch` — read the run config at call time rather than pinning
    the decision at import. `secaggplus_mod` lives in the legacy `flwr.client.mod` namespace (it
    has not been ported to `flwr.clientapp.mod` in 1.33), which is why the import is local.
    """
    if not bool(ctx.run_config.get("secure-aggregation", False)):
        return call_next(msg, ctx)

    from flwr.client.mod import secaggplus_mod

    return secaggplus_mod(msg, ctx, call_next)


app = ClientApp(mods=[_local_dp_dispatch, _secagg_dispatch])


def _legacy_arrays(msg: Message):
    """The broadcast weights, whichever record shape this round is using.

    The Message API sends them under `arrays`. The SecAgg+ round is driven by the legacy
    `DefaultWorkflow`, which speaks the legacy shape (`fitins.parameters`) — so a practice must
    understand both to serve either path. Flower's own compat bridge does the translation; this
    only picks which one applies, and returns a flag so the reply can be built to match.
    """
    if "arrays" in msg.content.array_records:
        return msg.content["arrays"].to_numpy_ndarrays(), False

    from flwr.app import MessageType
    from flwr.compat.common.recorddict_compat import (
        recorddict_to_evaluateins,
        recorddict_to_fitins,
    )
    from flwr.common import parameters_to_ndarrays

    # Train and evaluate carry the weights under different legacy keys
    # (`fitins.parameters` / `evaluateins.parameters`), so the bridge is chosen by message type.
    to_ins = (
        recorddict_to_fitins
        if msg.metadata.message_type == MessageType.TRAIN
        else recorddict_to_evaluateins
    )
    return parameters_to_ndarrays(to_ins(msg.content, True).parameters), True


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Warm-start from the global weights, train locally, return the updated weights."""
    client = build_client(context)
    ndarrays, legacy = _legacy_arrays(msg)

    updated, num_examples = client.fit(ndarrays)

    if legacy:
        from flwr.compat.common.recorddict_compat import fitres_to_recorddict
        from flwr.compat.common.typing import Code, FitRes, Status
        from flwr.common import ndarrays_to_parameters

        content = fitres_to_recorddict(
            FitRes(
                status=Status(Code.OK, ""),
                parameters=ndarrays_to_parameters(updated),
                num_examples=num_examples,
                metrics={},
            ),
            True,
        )
        return Message(content=content, reply_to=msg)

    # Only `num-examples` goes in the train MetricRecord: FedAvg weight-averages every key it
    # finds, and averaging a partition id produces a meaningless number in the round summary.
    content = RecordDict({
        "arrays": ArrayRecord(updated),
        "metrics": MetricRecord({"num-examples": num_examples}),
    })
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Score the global model on this practice's local held-out split."""
    client = build_client(context)
    ndarrays, legacy = _legacy_arrays(msg)

    metrics, num_examples = client.evaluate(ndarrays)
    reported = {
        "num-examples": num_examples,
        "partition-id": float(client.partition_id),
        **{k: float(v) for k, v in metrics.items()},
    }

    if legacy:
        from flwr.compat.common.recorddict_compat import evaluateres_to_recorddict
        from flwr.compat.common.typing import Code, EvaluateRes, Status

        content = evaluateres_to_recorddict(
            EvaluateRes(
                status=Status(Code.OK, ""),
                # The legacy signature needs a loss; this project scores by AUROC and sensitivity,
                # so loss carries 1 - AUROC purely to satisfy it. The metrics below are what the
                # dual-level aggregator actually reads.
                loss=float(1.0 - metrics.get("auc", 0.0)),
                num_examples=num_examples,
                metrics={k: float(v) for k, v in reported.items() if k != "num-examples"},
            )
        )
        return Message(content=content, reply_to=msg)

    return Message(content=RecordDict({"metrics": MetricRecord(reported)}), reply_to=msg)
