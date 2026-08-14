"""Model zoo: FedAvg-compatible logistic regression and MLP.

Federated XGBoost is not built here — it has no weight vector to average, so it lives in
`models/fedxgb.py` and federates through Flower's `FedXgbBagging` strategy instead.
"""

from .base import FederatedModel
from .logreg import LogRegModel
from .mlp import MLPModel

# Federated (FedAvg-compatible) architectures, selectable via the `model` run-config key.
FEDERATED_MODELS = {
    "logreg": LogRegModel,
    "mlp": MLPModel,
}


def make_model(name: str, **kwargs) -> FederatedModel:
    """Factory: build a federated model by name (`logreg` | `mlp`)."""
    try:
        return FEDERATED_MODELS[name](**kwargs)
    except KeyError as exc:
        raise ValueError(
            f"Unknown federated model {name!r}; choose from {sorted(FEDERATED_MODELS)}. "
            "('xgboost' has no weight vector to average — it federates via FedXgbBagging; "
            "see models/fedxgb.py.)"
        ) from exc


__all__ = ["FederatedModel", "LogRegModel", "MLPModel", "make_model", "FEDERATED_MODELS"]
