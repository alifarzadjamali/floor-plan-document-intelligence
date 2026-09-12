import cv2
import numpy as np

from floorplan_di.baseline import BaselineConfig, predict
from floorplan_di.constants import PlanClass


def test_baseline_finds_enclosed_room_and_wall() -> None:
    image = np.full((160, 160, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (25, 25), (135, 135), (0, 0, 0), thickness=8)

    prediction = predict(
        image,
        BaselineConfig(
            adaptive_block_size=31,
            line_kernel_fraction=0.04,
            wall_dilation_fraction=0.005,
            room_barrier_dilation_fraction=0.005,
        ),
    )

    assert prediction[80, 80] == PlanClass.ROOM
    assert prediction[25, 80] == PlanClass.WALL
    assert PlanClass.DOOR not in prediction
    assert PlanClass.WINDOW not in prediction


def test_baseline_rejects_non_rgb_input() -> None:
    with np.testing.assert_raises(ValueError):
        predict(np.zeros((10, 10), dtype=np.uint8), BaselineConfig())
