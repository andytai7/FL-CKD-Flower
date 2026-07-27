"""Small MLP baseline (neural network).

Implemented with sklearn's ``MLPClassifier`` to keep the stack torch-free and lightweight
(CLAUDE.md §4 option). Weight exchange = the network's ``coefs_`` (one matrix per layer) followed
by its ``intercepts_`` (one bias vector per layer), flattened into a single ``list[np.ndarray]``.

Notes:
- ``partial_fit`` provides warm-starting across rounds, analogous to the logistic baseline.
- ``MLPClassifier`` has no ``class_weight``; imbalance is left to the data/threshold for now
  (a manual PyTorch MLP with weighted ``BCEWithLogitsLoss`` is the documented upgrade path).
- BatchNorm caveats from CLAUDE.md don't apply here (sklearn MLP has no batch norm).
"""

from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPClassifier

from .base import FederatedModel

_CLASSES = np.array([0, 1])


class MLPModel(FederatedModel):
    def __init__(
        self,
        *,
        hidden_layer_sizes: tuple[int, ...] = (32, 16),
        alpha: float = 1e-3,
        seed: int = 42,
        **_ignored,  # accept (and ignore) class_weight_balanced for a uniform factory signature
    ):
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            alpha=alpha,
            max_iter=1,
            warm_start=True,
            random_state=seed,
        )
        self._n_coefs: int | None = None

    def initialize(self, n_features: int) -> None:
        X0 = np.zeros((2, n_features), dtype="float64")
        y0 = np.array([0, 1])
        self.model.partial_fit(X0, y0, classes=_CLASSES)
        self._n_coefs = len(self.model.coefs_)

    def get_parameters(self) -> list[np.ndarray]:
        coefs = [c.astype("float32") for c in self.model.coefs_]
        intercepts = [b.astype("float32") for b in self.model.intercepts_]
        return coefs + intercepts

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        n = self._n_coefs if self._n_coefs is not None else len(parameters) // 2
        self.model.coefs_ = [c.astype("float64") for c in parameters[:n]]
        self.model.intercepts_ = [b.astype("float64") for b in parameters[n:]]

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1) -> None:
        X = np.asarray(X, dtype="float64")
        # MLPClassifier can't warm-start partial_fit on a single-class batch (it rejects a class set
        # that differs from the first call). A practice can legitimately have all-negative patients
        # under non-IID, so skip the local update there — it keeps the global weights, contributing
        # nothing rather than crashing. (Logreg is immune: SGD pins classes=[0, 1].)
        if len(np.unique(y)) < 2:
            return
        for _ in range(epochs):
            self.model.partial_fit(X, y, classes=_CLASSES)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(np.asarray(X, dtype="float64"))[:, 1]
