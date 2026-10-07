import numpy as np
import pytest

from floorplan_di.evaluation.segmentation_metrics import ConfusionMatrix


def test_metrics_are_exact_for_perfect_prediction() -> None:
    target = np.asarray([[0, 1, 2, 3, 4]], dtype=np.uint8)
    metrics = ConfusionMatrix()
    metrics.update(target, target.copy())
    result = metrics.compute()

    assert result["mean_iou_all_classes"] == 1.0
    assert result["mean_dice_foreground"] == 1.0
    assert result["per_class"]["window"]["recall"] == 1.0


def test_metrics_accumulate_confusion() -> None:
    target = np.asarray([[1, 1, 2, 2]], dtype=np.uint8)
    prediction = np.asarray([[1, 0, 2, 1]], dtype=np.uint8)
    metrics = ConfusionMatrix()
    metrics.update(target, prediction)
    result = metrics.compute()

    assert result["per_class"]["room"]["precision"] == 0.5
    assert result["per_class"]["room"]["recall"] == 0.5
    assert result["per_class"]["wall"]["recall"] == 0.5


def test_metrics_reject_out_of_range_predictions() -> None:
    metrics = ConfusionMatrix()

    with pytest.raises(ValueError, match="prediction values"):
        metrics.update(np.asarray([0, 1]), np.asarray([0, 5]))


def test_metrics_ignore_predictions_at_ignored_target_pixels() -> None:
    metrics = ConfusionMatrix()
    metrics.update(np.asarray([0, 255]), np.asarray([0, 255]))

    assert metrics.matrix.sum() == 1


def test_metrics_support_reduced_class_sets() -> None:
    metrics = ConfusionMatrix(num_classes=2)
    metrics.update(np.asarray([0, 1]), np.asarray([0, 1]))

    result = metrics.compute()

    assert set(result["per_class"]) == {"background", "room"}
    assert result["mean_iou_foreground"] == 1.0


def test_single_class_metrics_report_undefined_foreground_as_none() -> None:
    metrics = ConfusionMatrix(num_classes=1)
    metrics.update(np.asarray([0]), np.asarray([0]))

    assert metrics.compute()["mean_iou_foreground"] is None


@pytest.mark.parametrize("field", ["target", "prediction"])
def test_metrics_reject_non_integer_class_ids(field: str) -> None:
    target = np.asarray([0.0, 1.0]) if field == "target" else np.asarray([0, 1])
    prediction = np.asarray([0.0, 1.0]) if field == "prediction" else np.asarray([0, 1])

    with pytest.raises(ValueError, match=f"{field} must contain integer"):
        ConfusionMatrix().update(target, prediction)


def test_empty_metrics_are_json_safe() -> None:
    result = ConfusionMatrix().compute()

    assert result["mean_iou_all_classes"] is None
    assert result["mean_dice_foreground"] is None
