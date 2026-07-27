"""Logistic regression baseline (FLIP-IT grant task T2.2 anchor).

Uses sklearn's ``SGDClassifier(loss="log_loss")`` rather than ``LogisticRegression`` because
``partial_fit`` supports warm-starting across federation rounds (the global weights become the
starting point for the next local update). Weight exchange = ``[coef_, intercept_]``.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import SGDClassifier

from .base import FederatedModel

_CLASSES = np.array([0, 1])


def _balanced_sample_weight(y: np.ndarray) -> np.ndarray | None:
    """Per-sample balanced weights (sklearn forbids class_weight='balanced' with partial_fit)."""
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return None  # can't rebalance a single-class batch
    n = len(y)
    weight_map = {c: n / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    return np.array([weight_map[v] for v in y], dtype="float64")


class LogRegModel(FederatedModel):
    def __init__(self, *, alpha: float = 1e-3, class_weight_balanced: bool = True, seed: int = 42):
        self.class_weight_balanced = class_weight_balanced
        self.model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=alpha,
            max_iter=1,                # epochs driven manually via partial_fit
            warm_start=True,
            random_state=seed,
        )
        self._n_features: int | None = None

    def initialize(self, n_features: int) -> None:
        # A single partial_fit on a tiny dummy batch allocates coef_/intercept_ of the right shape.
        # float64 throughout so coef_ and X dtypes always match (sklearn SGD is strict about this).
        self._n_features = n_features
        X0 = np.zeros((2, n_features), dtype="float64")
        y0 = np.array([0, 1])
        self.model.partial_fit(X0, y0, classes=_CLASSES)

    def get_parameters(self) -> list[np.ndarray]:
        return [self.model.coef_.flatten().astype("float32"), self.model.intercept_.astype("float32")]

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        coef, intercept = parameters
        self.model.coef_ = coef.reshape(1, -1).astype("float64")
        self.model.intercept_ = intercept.astype("float64")

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1) -> None:
        X = np.asarray(X, dtype="float64")
        sample_weight = _balanced_sample_weight(y) if self.class_weight_balanced else None
        for _ in range(epochs):
            self.model.partial_fit(X, y, classes=_CLASSES, sample_weight=sample_weight)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(np.asarray(X, dtype="float64"))[:, 1]
