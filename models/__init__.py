"""Model zoo: FedAvg-compatible logistic regression and MLP, plus a centralized GBT reference."""

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
            "('gbt' is centralized-only — use the ckd-baseline script.)"
        ) from exc


__all__ = ["FederatedModel", "LogRegModel", "MLPModel", "make_model", "FEDERATED_MODELS"]
