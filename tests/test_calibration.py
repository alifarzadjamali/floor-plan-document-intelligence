import pytest

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


def test_reliability_bins_clamp_scores_to_valid_range() -> None:
    rows = reliability_bins([-0.2, 1.5], [True, False], bins=2)

    assert rows[0]["mean_confidence"] == 0.0
    assert rows[1]["mean_confidence"] == 1.0


def test_reliability_bins_reject_non_finite_scores() -> None:
    with pytest.raises(ValueError, match="finite"):
        reliability_bins([float("nan")], [True])
