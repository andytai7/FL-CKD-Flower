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
    uv run ckd-baseline --clinics            # pooled data/clinics/ — comparable to --clinics runs

⚠️ **Match the data source to the federated run you are comparing against.** The default reads the
flat `synthetic_ckd_data.csv`, which is a signal-less placeholder — its ceiling is AUROC ≈ 0.5 and it
is NOT a valid ceiling for `ckd-simulate --clinics`. Use `--clinics` here whenever the federated side
used `--clinics`, or the "price of privacy" gap you compute is just a dataset difference.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from data import load_clinic_frames, load_dataframe, to_xy
from models import make_model
from models.fedxgb import predict_pooled, train_pooled
from task import compute_metrics, fit_scaler

MODELS = ("logreg", "mlp", "xgboost")


def load_pooled(clinics: bool = False) -> pd.DataFrame:
    """The pooled training frame: every clinic stacked, or the flat synthetic CSV."""
    if clinics:
        return pd.concat(load_clinic_frames(), ignore_index=True)
    return load_dataframe()


def _split(X, y, seed, test_frac=0.2):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    split = int(len(X) * (1 - test_frac))
    return X[:split], y[:split], X[split:], y[split:]


def run_centralized(
    model_name: str, seed: int = 42, local_epochs: int = 50, *, clinics: bool = False
) -> dict:
    X, y = to_xy(load_pooled(clinics))
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
    parser.add_argument(
        "--clinics", action="store_true",
        help="pool the on-disk data/clinics/ CSVs instead of the flat synthetic CSV. Use this "
             "whenever you are comparing against `ckd-simulate --clinics`. Run ckd-clinics first.",
    )
    args = parser.parse_args()

    names = list(MODELS) if args.model == "all" else [args.model]
    source = "pooled data/clinics/" if args.clinics else "flat synthetic_ckd_data.csv"
    print(f"Centralized (pooled-data) baselines — performance ceiling [{source}]\n" + "-" * 64)
    for name in names:
        m = run_centralized(name, seed=args.seed, clinics=args.clinics)
        print(
            f"{name:>9}:  AUROC={m['auc']:.3f}  sensitivity={m['sensitivity']:.3f}  "
            f"accuracy={m['accuracy']:.3f}"
        )


if __name__ == "__main__":
    main()
