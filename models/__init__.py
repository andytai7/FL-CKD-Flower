"""The model zoo is a single model: FedAvg-compatible warm-started logistic regression.

`LogRegModel` wraps sklearn's `SGDClassifier(loss="log_loss")` so `partial_fit` can warm-start
across federation rounds (the global weights become the starting point for the next local
update). Logistic regression is the only model this project trains — it is the first deployment
step, and the FedAvg / FedProx / FedMosaic protocols are aggregation strategies over it.
"""

from .base import FederatedModel
from .logreg import LogRegModel

__all__ = ["FederatedModel", "LogRegModel"]
