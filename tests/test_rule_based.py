import numpy as np

from models.rule_based import predict_ckd_from_egfr


def test_low_egfr_is_positive():
    prediction = predict_ckd_from_egfr(
        egfr=np.array([55.0]),
        uacr=np.array([10.0]),
    )

    assert prediction[0] == 1


def test_high_uacr_is_positive():
    prediction = predict_ckd_from_egfr(
        egfr=np.array([85.0]),
        uacr=np.array([45.0]),
    )

    assert prediction[0] == 1


def test_normal_values_are_negative():
    prediction = predict_ckd_from_egfr(
        egfr=np.array([85.0]),
        uacr=np.array([10.0]),
    )

    assert prediction[0] == 0


def test_threshold_boundary():
    prediction = predict_ckd_from_egfr(
        egfr=np.array([60.0]),
        uacr=np.array([30.0]),
    )

    assert prediction[0] == 1