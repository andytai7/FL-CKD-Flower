"""Centralized (pooled-data) baseline — the non-federated performance ceiling.

This trains on ALL practices' data combined, which the federated setup is not allowed to do.
Comparing federated (`ckd-simulate` / `flwr run`) numbers against this quantifies the price of
privacy: the federated-vs-centralized gap.

The model here is the **same Flower-compatible architecture** as the federated path — the warm-
started logistic regression (the only model this project trains), on standardized features, warm-
started like the federated FedAvg path, just on pooled data (project rule: we only use models that
can run with Flower; see CLAUDE.md §0).

    uv run ckd-baseline
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
from models import LogRegModel
from task import compute_metrics, fit_scaler


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
    seed: int = 42,
    local_epochs: int = 50,
    *,
    clinics: bool = False,
    frame: pd.DataFrame | None = None,
) -> dict:
    """Train the pooled logistic regression and return its held-out metrics (the ceiling).

    Pass `frame` to raise the ceiling on an arbitrary (already-extracted) canonical frame —
    e.g. the pooled Helios cohort — instead of the on-disk clinics/flat data.
    """
    pooled = load_pooled(clinics) if frame is None else frame
    X, y = to_xy(pooled)
    X_train, y_train, X_test, y_test = _split(X, y, seed)

    # Standardized features + warm-started partial_fit (mirrors the FedAvg path).
    scaler = fit_scaler(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)
    model = LogRegModel(seed=seed)
    model.initialize(X_train_s.shape[1])
    model.fit(X_train_s, y_train, epochs=local_epochs)
    y_score = model.predict_proba(X_test_s)

    return compute_metrics(y_test, y_score)


def main() -> None:
    parser = argparse.ArgumentParser(description="Centralized CKD baseline (pooled-data ceiling)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--clinics", action="store_true",
        help="pool the on-disk data/clinics/ CSVs instead of the flat synthetic CSV. Use this "
             "whenever you are comparing against `ckd-simulate --clinics`. Run ckd-clinics first.",
    )
    args = parser.parse_args()

    source = "pooled data/clinics/" if args.clinics else "flat synthetic_ckd_data.csv"
    print(f"Centralized (pooled-data) baseline — performance ceiling [{source}]\n" + "-" * 64)
    m = run_centralized(seed=args.seed, clinics=args.clinics)
    print(
        f"   logreg:  AUROC={m['auc']:.3f}  sensitivity={m['sensitivity']:.3f}  "
        f"accuracy={m['accuracy']:.3f}"
    )


if __name__ == "__main__":
    main()
