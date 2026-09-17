"""Small, dependency-free confidence calibration utilities.

Calibration is deliberately based on held-out validation observations.  It does
not change the model probabilities; it turns them into review bands whose
empirical correctness is visible to a user.
"""

from __future__ import annotations

from collections.abc import Iterable


def reliability_bins(
    scores: Iterable[float],
    correct: Iterable[bool],
    bins: int = 10,
) -> list[dict[str, float | int | None]]:
    """Return equal-width reliability bins for confidence/correctness pairs."""
    score_values = [min(1.0, max(0.0, float(value))) for value in scores]
    correct_values = [bool(value) for value in correct]
    if len(score_values) != len(correct_values):
        raise ValueError("scores and correct must have the same length")
    if bins < 1:
        raise ValueError("bins must be positive")
    result: list[dict[str, float | int | None]] = []
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            position
            for position, score in enumerate(score_values)
            if (lower <= score < upper) or (index == bins - 1 and score == upper)
        ]
        result.append(
            {
                "lower": round(lower, 3),
                "upper": round(upper, 3),
                "count": len(members),
                "mean_confidence": round(sum(score_values[i] for i in members) / len(members), 4)
                if members
                else None,
                "empirical_accuracy": round(sum(correct_values[i] for i in members) / len(members), 4)
                if members
                else None,
            }
        )
    return result


def expected_calibration_error(
    scores: Iterable[float], correct: Iterable[bool], bins: int = 10
) -> float:
    """Calculate the standard equal-width expected calibration error."""
    rows = reliability_bins(scores, correct, bins)
    total = sum(int(row["count"]) for row in rows)
    if not total:
        return 0.0
    return round(
        sum(
            int(row["count"])
            * abs(float(row["mean_confidence"]) - float(row["empirical_accuracy"]))
            for row in rows
            if row["count"]
        )
        / total,
        6,
    )


def confidence_band(score: float) -> str:
    """Map a calibrated score to a conservative review label."""
    value = min(1.0, max(0.0, float(score)))
    if value >= 0.85:
        return "high"
    if value >= 0.60:
        return "medium"
    return "low_review"
