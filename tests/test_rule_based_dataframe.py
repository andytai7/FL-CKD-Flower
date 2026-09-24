import pandas as pd

from models.rule_based import (
    predict_ckd_dataframe,
    evaluate_ckd_dataframe,
)


def test_rule_on_clinic_dataframe():
    clinic = pd.DataFrame(
        {
            "egfr": [55.0, 85.0, 85.0, 60.0, 70.0],
            "uacr": [10.0, 45.0, 10.0, 30.0, 29.0],
            "true_ckd": [1, 1, 0, 1, 1],
        }
    )

    predictions = predict_ckd_dataframe(clinic)

    expected = [1, 1, 0, 1, 0]

    assert predictions.tolist() == expected


def test_rule_metrics_on_clinic_dataframe():
    clinic = pd.DataFrame(
        {
            "egfr": [55.0, 85.0, 85.0, 60.0, 70.0],
            "uacr": [10.0, 45.0, 10.0, 30.0, 29.0],
            "true_ckd": [1, 1, 0, 1, 1],
        }
    )

    metrics = evaluate_ckd_dataframe(clinic)

    assert metrics["accuracy"] == 0.8
    assert metrics["sensitivity"] == 0.75