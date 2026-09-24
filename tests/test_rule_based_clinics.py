import pandas as pd

from models.rule_based import evaluate_ckd_clinics


def test_rule_on_multiple_clinics():
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

    results = evaluate_ckd_clinics(
        [clinic_0, clinic_1]
    )

    assert len(results) == 2

    assert results[0]["clinic_id"] == 0
    assert results[0]["num_examples"] == 4

    assert results[1]["clinic_id"] == 1
    assert results[1]["num_examples"] == 4

    assert results[0]["accuracy"] == 0.75
    assert results[1]["accuracy"] == 1.0

    