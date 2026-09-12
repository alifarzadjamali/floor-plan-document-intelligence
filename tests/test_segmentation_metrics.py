import numpy as np

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
