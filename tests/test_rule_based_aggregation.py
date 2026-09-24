import pandas as pd
import pytest

from models.rule_based import (
    evaluate_ckd_clinics,
    aggregate_ckd_clinic_results,
)


def test_aggregate_rule_based_clinic_results():
    clinic_0 = pd.DataFrame(
        {
            "egfr": [55.0, 85.0, 85.0, 70.0],
            "uacr": [10.0, 45.0, 10.0, 29.0],
            "true_ckd": [1, 1, 0, 1],
        }
    )

    clinic_1 = pd.DataFrame(
        {
            "egfr": [50.0, 90.0, 75.0, 40.0],
            "uacr": [5.0, 5.0, 50.0, 10.0],
            "true_ckd": [1, 0, 1, 1],
        }
    )

    clinic_results = evaluate_ckd_clinics(
        [clinic_0, clinic_1]
    )

    summary = aggregate_ckd_clinic_results(
        clinic_results
    )

    assert summary["accuracy"] == pytest.approx(0.875)
    assert summary["accuracy_worst"] == pytest.approx(0.75)

    assert summary["sensitivity"] == pytest.approx(5 / 6)
    assert summary["sensitivity_worst"] == pytest.approx(2 / 3)

    