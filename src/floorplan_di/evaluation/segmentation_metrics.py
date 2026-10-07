from __future__ import annotations

import numpy as np

from floorplan_di.constants import CLASS_NAMES


class ConfusionMatrix:
    def __init__(self, num_classes: int = len(CLASS_NAMES)) -> None:
        if num_classes < 1:
            raise ValueError("num_classes must be positive")
        if num_classes > len(CLASS_NAMES):
            raise ValueError(f"num_classes cannot exceed {len(CLASS_NAMES)} known classes")
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, target: np.ndarray, prediction: np.ndarray) -> None:
        if target.shape != prediction.shape:
            raise ValueError(
                f"Shape mismatch: target={target.shape}, prediction={prediction.shape}"
            )
        if not np.issubdtype(target.dtype, np.integer):
            raise ValueError("target must contain integer class ids")
        if not np.issubdtype(prediction.dtype, np.integer):
            raise ValueError("prediction must contain integer class ids")
        valid = (target >= 0) & (target < self.num_classes)
        valid_prediction = prediction[valid]
        if np.any((valid_prediction < 0) | (valid_prediction >= self.num_classes)):
            raise ValueError(f"prediction values must be between 0 and {self.num_classes - 1}")
        encoded = self.num_classes * target[valid].astype(np.int64) + valid_prediction.astype(
            np.int64
        )
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
            for index, name in enumerate(CLASS_NAMES[: self.num_classes])
        }
        foreground = np.arange(1, self.num_classes)
        return {
            "mean_iou_all_classes": _optional_mean(iou),
            "mean_dice_all_classes": _optional_mean(dice),
            "mean_iou_foreground": _optional_mean(iou[foreground]),
            "mean_dice_foreground": _optional_mean(dice[foreground]),
            "per_class": per_class,
            "confusion_matrix": self.matrix.tolist(),
        }


def _optional_float(value: np.floating) -> float | None:
    return None if np.isnan(value) else float(value)


def _optional_mean(values: np.ndarray) -> float | None:
    finite = values[~np.isnan(values)]
    return float(finite.mean()) if finite.size else None
