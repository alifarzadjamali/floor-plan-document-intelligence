from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from floorplan_di.constants import PlanClass


@dataclass(frozen=True)
class BaselineConfig:
    adaptive_block_size: int = 31
    adaptive_c: float = 12
    line_kernel_fraction: float = 0.012
    minimum_line_kernel: int = 5
    wall_dilation_fraction: float = 0.0025
    closing_kernel: int = 3
    minimum_room_area_fraction: float = 0.0005
    room_barrier_dilation_fraction: float = 0.0015

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> BaselineConfig:
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown baseline settings: {sorted(unknown)}")
        return cls(**values)


def _odd(value: int) -> int:
    return max(3, value if value % 2 else value + 1)


def _connected_interior(free_space: np.ndarray, minimum_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(free_space, connectivity=4)
    if count <= 1:
        return np.zeros_like(free_space)
    border_labels = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    room = np.zeros_like(free_space)
    for label in range(1, count):
        if label not in border_labels and stats[label, cv2.CC_STAT_AREA] >= minimum_area:
            room[labels == label] = 255
    return room


def predict(image_rgb: np.ndarray, config: BaselineConfig) -> np.ndarray:
    """Infer room and wall masks using only image morphology."""
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape HxWx3")
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    short_side = min(gray.shape)
    block_size = _odd(min(config.adaptive_block_size, short_side - 1))
    ink = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        config.adaptive_c,
    )
    close_size = _odd(config.closing_kernel)
    ink = cv2.morphologyEx(
        ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (close_size,) * 2)
    )

    line_length = max(config.minimum_line_kernel, round(short_side * config.line_kernel_fraction))
    horizontal = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (line_length, 1))
    )
    vertical = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_length))
    )
    structural = cv2.bitwise_or(horizontal, vertical)
    dilation = max(1, round(short_side * config.wall_dilation_fraction))
    wall = cv2.dilate(structural, cv2.getStructuringElement(cv2.MORPH_RECT, (_odd(dilation),) * 2))

    barrier_dilation = max(1, round(short_side * config.room_barrier_dilation_fraction))
    barrier = cv2.dilate(
        structural, cv2.getStructuringElement(cv2.MORPH_RECT, (_odd(barrier_dilation),) * 2)
    )
    free_space = cv2.bitwise_not(barrier)
    minimum_area = max(16, round(gray.size * config.minimum_room_area_fraction))
    room = _connected_interior(free_space, minimum_area)

    prediction = np.full(gray.shape, int(PlanClass.BACKGROUND), dtype=np.uint8)
    prediction[room > 0] = int(PlanClass.ROOM)
    prediction[wall > 0] = int(PlanClass.WALL)
    return prediction
