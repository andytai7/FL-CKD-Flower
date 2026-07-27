"""Centralized (pooled-data) baselines — the non-federated performance ceiling.

These train on ALL practices' data combined, which the federated setup is not allowed to do.
Comparing federated (`ckd-simulate` / `flwr run`) numbers against these quantifies the price of
privacy: the federated-vs-centralized gap.

Every model here is the **same Flower-compatible architecture** as the federated path, just trained
on pooled data — there are no extra, non-federatable models (project rule: we only use models that
can run with Flower; see CLAUDE.md §0):
- `logreg`, `mlp`  -> standardized features, warm-started like the federated FedAvg path
- `xgboost`        -> the pooled counterpart of the federated FedXgbBagging trees

    uv run ckd-baseline --model all
    uv run ckd-baseline --model xgboost
    uv run ckd-baseline --model logreg
"""

from __future__ import annotations

import argparse

import numpy as np

from data import load_dataframe, to_xy
from models import make_model
from models.fedxgb import predict_pooled, train_pooled
from task import compute_metrics, fit_scaler

MODELS = ("logreg", "mlp", "xgboost")


def _split(X, y, seed, test_frac=0.2):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    split = int(len(X) * (1 - test_frac))
    return X[:split], y[:split], X[split:], y[split:]


def run_centralized(model_name: str, seed: int = 42, local_epochs: int = 50) -> dict:
    X, y = to_xy(load_dataframe())
    X_train, y_train, X_test, y_test = _split(X, y, seed)

    if model_name == "xgboost":
        # Trees use raw (unscaled) features — pooled counterpart of the federated FedXgbBagging.
        bst = train_pooled(X_train, y_train, seed=seed)
        y_score = predict_pooled(bst, X_test)
    else:
        # Linear / NN: standardized features + warm-started partial_fit (mirrors the FedAvg path).
        scaler = fit_scaler(X_train)
        X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)
        model = make_model(model_name, seed=seed)
        model.initialize(X_train_s.shape[1])
        model.fit(X_train_s, y_train, epochs=local_epochs)
        y_score = model.predict_proba(X_test_s)

    return compute_metrics(y_test, y_score)


def main() -> None:
    parser = argparse.ArgumentParser(description="Centralized CKD baselines (pooled-data ceiling)")
    parser.add_argument(
        "--model", default="all", choices=[*MODELS, "all"],
        help="Which centralized baseline(s) to run",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    names = list(MODELS) if args.model == "all" else [args.model]
    print("Centralized (pooled-data) baselines — performance ceiling\n" + "-" * 56)
    for name in names:
        m = run_centralized(name, seed=args.seed)
        print(
            f"{name:>9}:  AUROC={m['auc']:.3f}  sensitivity={m['sensitivity']:.3f}  "
            f"accuracy={m['accuracy']:.3f}"
        )


if __name__ == "__main__":
    main()
