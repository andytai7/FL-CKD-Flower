"""Rule-based CKD baseline.

This baseline does not learn parameters from data.
It applies the fixed clinical rule supplied for the project:

    CKD = True if eGFR < 60 OR UACR >= 30 mg/g

If eGFR is not available directly, it can be calculated from
serum creatinine, age, and sex using the CKD-EPI 2021 equation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from data.external.nhanes_to_clinics import egfr_ckd_epi_2021
from task import compute_metrics


EGFR_THRESHOLD = 60.0
UACR_THRESHOLD = 30.0

# When eGFR is available, we can apply the fixed CKD rule directly.
def predict_ckd_from_egfr(
    egfr: np.ndarray,
    uacr: np.ndarray,
) -> np.ndarray:
    """Apply the fixed CKD rule.

    CKD positive when:
        eGFR < 60
        OR
        UACR >= 30 mg/g

    Returns:
        NumPy array containing 0 (negative) or 1 (positive).
    """
    egfr = np.asarray(egfr, dtype=float)
    uacr = np.asarray(uacr, dtype=float)

    return ((egfr < EGFR_THRESHOLD) | (uacr >= UACR_THRESHOLD)).astype(int)


# When eGFR is not available, we can calculate it from serum creatinine, age, and sex.
def predict_ckd_from_creatinine(
    serum_creatinine: np.ndarray,
    age: np.ndarray,
    female: np.ndarray,
    uacr: np.ndarray,
) -> np.ndarray:
    """Calculate eGFR first, then apply the CKD rule."""
    egfr = egfr_ckd_epi_2021(
        scr_mgdl=np.asarray(serum_creatinine, dtype=float),
        age=np.asarray(age, dtype=float),
        female=np.asarray(female, dtype=bool),
    )

    return predict_ckd_from_egfr(egfr, uacr)

def predict_ckd_dataframe(df: pd.DataFrame) -> np.ndarray:
    """Apply the CKD rule to all patients in a DataFrame.

    Required columns:
        egfr
        uacr

    Returns:
        NumPy array of 0/1 CKD predictions.
    """
    return predict_ckd_from_egfr(
        egfr=df["egfr"].to_numpy(),
        uacr=df["uacr"].to_numpy(),
    )


def evaluate_ckd_dataframe(
    df: pd.DataFrame,
    label_column: str = "true_ckd",
) -> dict:
    """Evaluate the rule-based baseline on one clinic DataFrame.

    The DataFrame must contain:
        egfr
        uacr
        true CKD label

    Returns the same core metrics used by the existing FL pipeline:
        accuracy
        sensitivity
        auc
    """
    predictions = predict_ckd_dataframe(df)

    y_true = df[label_column].to_numpy(dtype=int)

    return compute_metrics(
        y_true=y_true,
        y_score=predictions.astype(float),
    )

def evaluate_ckd_clinics(
    clinic_frames: list[pd.DataFrame],
    label_column: str = "true_ckd",
) -> list[dict]:
    """Evaluate the rule-based baseline separately at each clinic."""

    results = []

    for clinic_id, clinic in enumerate(clinic_frames):
        metrics = evaluate_ckd_dataframe(
            clinic,
            label_column=label_column,
        )

        results.append(
            {
                "clinic_id": clinic_id,
                "num_examples": len(clinic),
                **metrics,
            }
        )

    return results


def aggregate_ckd_clinic_results(
    results: list[dict],
) -> dict:
    """Aggregate rule-based results like the FL server.

    For each metric:
    - sample-size-weighted mean across clinics
    - worst (minimum) clinic value
    """

    output = {}

    metric_names = ("accuracy", "sensitivity", "auc")

    for metric in metric_names:
        valid_results = [
            result
            for result in results
            if metric in result and not np.isnan(result[metric])
        ]

        if not valid_results:
            continue

        total_examples = sum(
            result["num_examples"]
            for result in valid_results
        )

        weighted_mean = sum(
            result["num_examples"] * result[metric]
            for result in valid_results
        ) / total_examples

        worst = min(
            result[metric]
            for result in valid_results
        )

        output[metric] = weighted_mean
        output[f"{metric}_worst"] = worst

    return output
