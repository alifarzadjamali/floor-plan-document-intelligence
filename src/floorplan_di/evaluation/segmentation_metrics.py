from __future__ import annotations

import numpy as np

from floorplan_di.constants import CLASS_NAMES


class ConfusionMatrix:
    def __init__(self, num_classes: int = len(CLASS_NAMES)) -> None:
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, target: np.ndarray, prediction: np.ndarray) -> None:
        if target.shape != prediction.shape:
            raise ValueError(
                f"Shape mismatch: target={target.shape}, prediction={prediction.shape}"
            )
        valid = (target >= 0) & (target < self.num_classes)
        encoded = self.num_classes * target[valid].astype(np.int64) + prediction[valid]
        self.matrix += np.bincount(encoded, minlength=self.num_classes**2).reshape(
            self.num_classes, self.num_classes
        )

    def compute(self) -> dict[str, object]:
        true_positive = np.diag(self.matrix).astype(np.float64)
        target_total = self.matrix.sum(axis=1).astype(np.float64)
        predicted_total = self.matrix.sum(axis=0).astype(np.float64)
        union = target_total + predicted_total - true_positive
        iou = np.divide(true_positive, union, out=np.full_like(union, np.nan), where=union > 0)
        dice_denominator = target_total + predicted_total
        dice = np.divide(
            2 * true_positive,
            dice_denominator,
            out=np.full_like(union, np.nan),
            where=dice_denominator > 0,
        )
        precision = np.divide(
            true_positive,
            predicted_total,
            out=np.full_like(union, np.nan),
            where=predicted_total > 0,
        )
        recall = np.divide(
            true_positive,
            target_total,
            out=np.full_like(union, np.nan),
            where=target_total > 0,
        )

        per_class = {
            name: {
                "iou": _optional_float(iou[index]),
                "dice": _optional_float(dice[index]),
                "precision": _optional_float(precision[index]),
                "recall": _optional_float(recall[index]),
                "support_pixels": int(target_total[index]),
            }
            for index, name in enumerate(CLASS_NAMES)
        }
        foreground = np.arange(1, self.num_classes)
        return {
            "mean_iou_all_classes": float(np.nanmean(iou)),
            "mean_dice_all_classes": float(np.nanmean(dice)),
            "mean_iou_foreground": float(np.nanmean(iou[foreground])),
            "mean_dice_foreground": float(np.nanmean(dice[foreground])),
            "per_class": per_class,
            "confusion_matrix": self.matrix.tolist(),
        }


def _optional_float(value: np.floating) -> float | None:
    return None if np.isnan(value) else float(value)
