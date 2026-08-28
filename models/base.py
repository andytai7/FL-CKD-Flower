"""The weight-exchange contract for the project's single model, `LogRegModel`.

The logistic regression exposes its weights as a flat ``list[np.ndarray]`` — ``[coef_,
intercept_]`` — so the Flower client/server can exchange and average them without knowing the
model internals: the client warm-starts from the global arrays, trains locally, and hands the
updated arrays back to FedAvg.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class FederatedModel(ABC):
    """A model whose parameters can be get/set as NumPy arrays for FedAvg."""

    @abstractmethod
    def initialize(self, n_features: int) -> None:
        """Create parameter arrays of the correct shape (called once before round 1)."""

    @abstractmethod
    def get_parameters(self) -> list[np.ndarray]:
        """Return current weights as a list of NumPy arrays."""

    @abstractmethod
    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        """Load aggregated weights received from the server."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1) -> None:
        """Run ``epochs`` local warm-started passes over local data."""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(CKD positive) for each row."""
