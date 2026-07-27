"""Common interface for federated (FedAvg-compatible) models.

Both the logistic-regression and MLP baselines expose their weights as a flat ``list[np.ndarray]``
so the Flower client/server can exchange and average them without knowing the model type. This is
what makes the pipeline "model-agnostic": swap the model, keep the client/server untouched.
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
