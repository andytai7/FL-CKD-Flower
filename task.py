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


def fairness_metrics(
    y_true: np.ndarray, y_score: np.ndarray, group: np.ndarray, threshold: float = 0.5
) -> dict:
    """The four group-fairness measures the Projektantrag names for T2.5.

    Quoted from the funding application (AP2, T2.5 Datenschutztests und Bias-Analyse): *"untersuchen
    wir demographic parity, equal opportunity, equal odds, und callibration by group"*.

    Each is reported as a **gap** — the largest absolute difference between any two groups — so 0.0
    is perfectly fair and larger is worse:

    - `demographic_parity_gap` — difference in the rate of being flagged positive
    - `equal_opportunity_gap`  — difference in true-positive rate (sensitivity) among actual cases
    - `equalized_odds_gap`     — the worse of the TPR gap and the FPR gap
    - `calibration_gap`        — difference between mean predicted risk and observed rate

    ⚠️ The Antrag specifies this analysis by **sex**, which the synthetic schema does not carry —
    `geschlecht` exists only in the real `extract_features.sql` contract (CLAUDE.md §3b). Pass
    whatever grouping the data supports (age band here) and treat the sex-based audit as pending
    real data.
    """
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    groups = [g for g in np.unique(group) if np.sum(group == g) > 0]

    positive_rate, tpr, fpr, calibration = [], [], [], []
    for g in groups:
        mask = group == g
        yt, yp, ys = y_true[mask], y_pred[mask], np.asarray(y_score)[mask]
        positive_rate.append(float(np.mean(yp)))
        calibration.append(float(np.mean(ys) - np.mean(yt)))
        if np.any(yt == 1):
            tpr.append(float(np.mean(yp[yt == 1])))
        if np.any(yt == 0):
            fpr.append(float(np.mean(yp[yt == 0])))

    def gap(values: list[float]) -> float:
        return float(max(values) - min(values)) if len(values) > 1 else float("nan")

    tpr_gap, fpr_gap = gap(tpr), gap(fpr)
    return {
        "demographic_parity_gap": gap(positive_rate),
        "equal_opportunity_gap": tpr_gap,
        "equalized_odds_gap": float(np.nanmax([tpr_gap, fpr_gap])),
        "calibration_gap": gap(calibration),
        "n_groups": float(len(groups)),
    }
