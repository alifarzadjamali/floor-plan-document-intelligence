from floorplan_di.evaluation.error_analysis import choose_error_examples


def test_error_examples_use_fixed_extrema() -> None:
    records = [
        {"sample_id": "a", "foreground_iou": 0.2, "missed_openings": 3, "false_openings": 1,
         "text_links": 2, "unassigned_text": 5, "resolved_doors": 1, "unresolved_doors": 4},
        {"sample_id": "b", "foreground_iou": 0.8, "missed_openings": 1, "false_openings": 7,
         "text_links": 8, "unassigned_text": 2, "resolved_doors": 6, "unresolved_doors": 1},
    ]
    examples = choose_error_examples(records)
    assert examples["strong_segmentation"]["sample_id"] == "b"
    assert examples["weak_segmentation"]["sample_id"] == "a"
    assert examples["false_door_or_window"]["sample_id"] == "b"
