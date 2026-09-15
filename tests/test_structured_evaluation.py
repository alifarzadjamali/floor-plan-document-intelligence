from floorplan_di.evaluation.structured import summarise_structured_records


def test_structured_summary_reports_count_error_and_vectorization_rate() -> None:
    report = summarise_structured_records(
        [
            {
                "predicted_rooms": 3,
                "ground_truth_rooms": 5,
                "predicted_doors": 4,
                "ground_truth_doors": 3,
                "predicted_windows": 1,
                "ground_truth_windows": 1,
                "attempted_entities": 8,
                "vectorized_entities": 6,
            }
        ]
    )
    assert report["mean_absolute_count_error"] == {"rooms": 2.0, "doors": 1.0, "windows": 0.0}
    assert report["vectorization_rate"] == 0.75
