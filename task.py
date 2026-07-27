"""Shared helpers: local preprocessing and the imbalanced-data metrics used everywhere.

Metrics prioritize AUROC and sensitivity (recall for CKD positives) over raw accuracy, because the
cohort is imbalanced.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """Fit a StandardScaler on local training data only (never shared)."""
    return StandardScaler().fit(X_train)


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    """Accuracy, sensitivity (recall+), and ROC-AUC. AUC is NaN if only one class is present."""
    y_pred = (y_score >= threshold).astype(int)
    accuracy = float(np.mean(y_pred == y_true))

    tp = float(np.sum((y_pred == 1) & (y_true == 1)))
    fn = float(np.sum((y_pred == 0) & (y_true == 1)))
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

    auc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else float("nan")
    return {"accuracy": accuracy, "sensitivity": sensitivity, "auc": auc}
