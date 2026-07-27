"""Flower ClientApp — one simulated GP practice.

Model-agnostic: it asks the factory for whatever architecture the run is configured with, loads
its non-IID data partition, splits/scales locally (data never leaves the client), and runs
warm-started local training each round.
"""

from __future__ import annotations

import numpy as np
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from data import load_partition, to_xy
from models import make_model
from task import compute_metrics, fit_scaler


class FlowerClient(NumPyClient):
    def __init__(self, model, X_train, y_train, X_test, y_test, local_epochs, partition_id):
        self.model = model
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test
        self.local_epochs = local_epochs
        self.partition_id = partition_id

    def fit(self, parameters, config):
        self.model.set_parameters(parameters)
        self.model.fit(self.X_train, self.y_train, epochs=self.local_epochs)
        return self.model.get_parameters(), len(self.X_train), {"partition_id": self.partition_id}

    def evaluate(self, parameters, config):
        self.model.set_parameters(parameters)
        y_score = self.model.predict_proba(self.X_test)
        metrics = compute_metrics(self.y_test, y_score)
        loss = 1.0 - metrics["accuracy"]  # proxy loss for the imbalanced task
        metrics["partition_id"] = float(self.partition_id)
        return float(loss), len(self.X_test), metrics


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


def build_client_from_frame(df, partition_id: int, run_config) -> FlowerClient:
    """Preprocess one practice's DataFrame locally and build its FlowerClient (data never shared).

    The data-source-agnostic core: callers supply the rows (a Dirichlet partition of the flat CSV,
    or one on-disk clinic file), and this does the local split + scaling + model build identically.
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

    return FlowerClient(
        model, X_train, y_train, X_test, y_test, int(run_config["local-epochs"]), partition_id
    )


def build_client(partition_id: int, num_partitions: int, run_config) -> FlowerClient:
    """Load one practice's Dirichlet/IID partition of the flat CSV, then build its FlowerClient.

    Used by the Flower ClientApp (`client_fn`) for `flwr run`; the Ray-free runner (`simulate.py`)
    calls `build_client_from_frame` directly so it can also feed in on-disk clinic files.
    """
    seed = int(run_config["seed"])
    df = load_partition(
        partition_id,
        num_partitions,
        alpha=float(run_config["alpha"]),
        iid=bool(run_config["iid"]),
        seed=seed,
    )
    return build_client_from_frame(df, partition_id, run_config)


def client_fn(context: Context):
    return build_client(
        int(context.node_config["partition-id"]),
        int(context.node_config["num-partitions"]),
        context.run_config,
    ).to_client()


app = ClientApp(client_fn=client_fn)
