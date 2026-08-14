"""Flower ClientApp — one GP practice (Message API, flwr 1.33).

Model-agnostic: it asks the factory for whatever architecture the run is configured with, loads its
data partition, splits/scales locally (data never leaves the client), and runs warm-started local
training each round.

Message API contract (CLAUDE.md §6):
- incoming  `msg.content["arrays"]`  -> global model weights
- outgoing  `{"arrays": ArrayRecord, "metrics": MetricRecord}` where the MetricRecord carries
  `num-examples` — the key FedAvg weights the average by.
"""

from __future__ import annotations

import numpy as np
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from data import load_partition, load_practice_frame, to_xy
from models import make_model
from task import compute_metrics, fit_scaler

app = ClientApp()


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


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Warm-start from the global weights, train locally, return the updated weights."""
    client = build_client(context)
    ndarrays = msg.content["arrays"].to_numpy_ndarrays()

    updated, num_examples = client.fit(ndarrays)

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
    ndarrays = msg.content["arrays"].to_numpy_ndarrays()

    metrics, num_examples = client.evaluate(ndarrays)

    content = RecordDict({
        "metrics": MetricRecord({
            "num-examples": num_examples,
            "partition-id": float(client.partition_id),
            **{k: float(v) for k, v in metrics.items()},
        }),
    })
    return Message(content=content, reply_to=msg)
