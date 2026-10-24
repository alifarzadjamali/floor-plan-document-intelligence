"""Evaluation of approximate structured output against CubiCasa SVG counts."""

from __future__ import annotations

from collections.abc import Iterable

COUNT_CATEGORIES = ("rooms", "doors", "windows")


def summarise_structured_records(records: Iterable[dict[str, object]]) -> dict[str, object]:
    """Aggregate count error and vectorisation validity without inventing links GT."""
    rows = list(records)
    if not rows:
        raise ValueError("At least one structured-evaluation record is required")
    for row in rows:
        counts = [
            int(row[f"{source}_{category}"])
            for category in COUNT_CATEGORIES
            for source in ("predicted", "ground_truth")
        ]
        attempted = int(row["attempted_entities"])
        vectorized = int(row["vectorized_entities"])
        if any(count < 0 for count in counts) or not 0 <= vectorized <= attempted:
            raise ValueError(
                "Entity counts must be nonnegative and vectorized cannot exceed attempted"
            )
    count_error = {}
    count_bias = {}
    for category in COUNT_CATEGORIES:
        total_error = sum(
            abs(int(row[f"predicted_{category}"]) - int(row[f"ground_truth_{category}"]))
            for row in rows
        )
        count_error[category] = total_error / len(rows)
        count_bias[category] = sum(
            int(row[f"predicted_{category}"]) - int(row[f"ground_truth_{category}"])
            for row in rows
        ) / len(rows)
    attempted = sum(int(row["attempted_entities"]) for row in rows)
    vectorized = sum(int(row["vectorized_entities"]) for row in rows)
    return {
        "plans": len(rows),
        "mean_absolute_count_error": count_error,
        "mean_signed_count_error": count_bias,
        "attempted_detected_entities": attempted,
        "successfully_vectorized_entities": vectorized,
        "vectorization_rate": vectorized / attempted if attempted else None,
        "per_plan": rows,
        "ground_truth_note": (
            "Counts come from CubiCasa SVG target polygons. Connectivity and OCR have no "
            "trustworthy ground truth here and are not scored."
        ),
    }
