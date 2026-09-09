"""The model zoo is a single model: FedAvg-compatible warm-started logistic regression.

`LogRegModel` wraps sklearn's `SGDClassifier(loss="log_loss")` so `partial_fit` can warm-start
across federation rounds (the global weights become the starting point for the next local
update). Logistic regression is the only model this project trains — it is the first deployment
step, and the FedAvg / FedProx / FedMosaic protocols are aggregation strategies over it.

The published *rule-based* clinical baseline (Tangri KFRE) is deliberately NOT here: it has no
trainable/federated path and lives in the repo-root `kfre.py` as an evaluation reference only.
"""

from .base import FederatedModel
from .logreg import LogRegModel

__all__ = ["FederatedModel", "LogRegModel"]
