"""Evaluation of approximate structured output against CubiCasa SVG counts."""

from __future__ import annotations

from collections.abc import Iterable

COUNT_CATEGORIES = ("rooms", "doors", "windows")


def summarise_structured_records(records: Iterable[dict[str, object]]) -> dict[str, object]:
    """Aggregate count error and vectorisation validity without inventing links GT."""
    rows = list(records)
    if not rows:
        raise ValueError("At least one structured-evaluation record is required")
    count_error = {}
    for category in COUNT_CATEGORIES:
        total_error = sum(
            abs(int(row[f"predicted_{category}"]) - int(row[f"ground_truth_{category}"]))
            for row in rows
        )
        count_error[category] = total_error / len(rows)
    attempted = sum(int(row["attempted_entities"]) for row in rows)
    vectorized = sum(int(row["vectorized_entities"]) for row in rows)
    return {
        "plans": len(rows),
        "mean_absolute_count_error": count_error,
        "attempted_detected_entities": attempted,
        "successfully_vectorized_entities": vectorized,
        "vectorization_rate": vectorized / attempted if attempted else None,
        "per_plan": rows,
        "ground_truth_note": (
            "Counts come from CubiCasa SVG target polygons. Connectivity and OCR have no "
            "trustworthy ground truth here and are not scored."
        ),
    }
