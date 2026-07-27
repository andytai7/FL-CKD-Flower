"""Federated XGBoost via Flower's real ``FedXgbBagging`` strategy.

XGBoost can't FedAvg (there are no weight vectors to average), so Flower federates it by **bagging**:
each practice trains a few local boosting rounds on its own data and sends only the *new* trees it
grew; the server (``flwr.server.strategy.FedXgbBagging``) appends them into one growing global
ensemble. This is the genuine Flower federated-trees path — and the only tree model compatible with
Flower, which is why it's the single tree architecture this project keeps.

This module provides the per-practice local-training logic (mirroring Flower's official xgboost
client) plus a pooled-data trainer for the centralized "ceiling" reference. The in-process driver
that wires these into the real ``FedXgbBagging`` strategy lives in ``simulate.py``.
"""

from __future__ import annotations

import numpy as np
import xgboost as xgb

# Binary CKD classifier; histogram trees, single-threaded for reproducible in-process simulation.
XGB_PARAMS: dict = {
    "objective": "binary:logistic",
    "eta": 0.1,
    "max_depth": 3,
    "eval_metric": "logloss",
    "tree_method": "hist",
    "nthread": 1,
}


def _params(seed: int, y: np.ndarray | None, balanced: bool) -> dict:
    params = dict(XGB_PARAMS, seed=seed)
    if balanced and y is not None:
        pos = max(1, int(np.sum(y == 1)))
        neg = int(np.sum(y == 0))
        params["scale_pos_weight"] = neg / pos
    return params


class XgbPractice:
    """One practice's local XGBoost training for FedXgbBagging (mirrors Flower's xgboost client)."""

    def __init__(
        self, X_train, y_train, X_val, y_val, *,
        num_local_round: int = 2, class_weight_balanced: bool = True, seed: int = 42,
    ):
        self.dtrain = xgb.DMatrix(X_train, label=y_train)
        self.dval = xgb.DMatrix(X_val, label=y_val)
        self.y_val = np.asarray(y_val)
        self.num_examples = int(len(X_train))
        self.num_local_round = num_local_round
        self.params = _params(seed, y_train, class_weight_balanced)

    def local_trees(self, global_model: bytes | None) -> bytes:
        """Train locally and return only this round's NEW trees (serialized) for bagging."""
        if global_model is None:
            # Round 1: grow a fresh local ensemble.
            bst = xgb.train(self.params, self.dtrain, num_boost_round=self.num_local_round)
        else:
            # Later rounds: continue from the global ensemble, then keep only the new local trees.
            bst = xgb.Booster(params=self.params)
            bst.load_model(bytearray(global_model))
            for _ in range(self.num_local_round):
                bst.update(self.dtrain, bst.num_boosted_rounds())
            total = bst.num_boosted_rounds()
            bst = bst[total - self.num_local_round : total]
        return bytes(bst.save_raw("json"))

    def predict_proba(self, global_model: bytes) -> np.ndarray:
        """P(CKD positive) on the local validation set, using the full global ensemble."""
        bst = xgb.Booster(params=self.params)
        bst.load_model(bytearray(global_model))
        return bst.predict(self.dval)


def train_pooled(
    X: np.ndarray, y: np.ndarray, *, num_round: int = 50, class_weight_balanced: bool = True,
    seed: int = 42,
) -> xgb.Booster:
    """Centralized pooled-data XGBoost — the non-federated 'ceiling' reference."""
    dtrain = xgb.DMatrix(X, label=y)
    return xgb.train(_params(seed, y, class_weight_balanced), dtrain, num_boost_round=num_round)


def predict_pooled(bst: xgb.Booster, X: np.ndarray) -> np.ndarray:
    return bst.predict(xgb.DMatrix(X))
