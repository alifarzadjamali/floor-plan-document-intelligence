from floorplan_di.evaluation.calibration import (
    confidence_band,
    expected_calibration_error,
    reliability_bins,
)


def test_reliability_bins_and_ece_are_deterministic() -> None:
    rows = reliability_bins([0.1, 0.9, 1.0], [False, True, False], bins=2)
    assert rows[0]["count"] == 1
    assert rows[1]["count"] == 2
    assert expected_calibration_error([0.1, 0.9, 1.0], [False, True, False], bins=2) == 0.333333


def test_confidence_bands_are_review_oriented() -> None:
    assert confidence_band(0.9) == "high"
    assert confidence_band(0.7) == "medium"
    assert confidence_band(0.3) == "low_review"
