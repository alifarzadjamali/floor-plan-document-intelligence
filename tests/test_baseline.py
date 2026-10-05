import cv2
import numpy as np

from floorplan_di.baseline import BaselineConfig, predict
from floorplan_di.baseline.classical_cv import _connected_interior
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


def test_connected_interior_filters_border_and_small_components() -> None:
    free_space = np.zeros((12, 16), dtype=np.uint8)
    free_space[0:4, 1:5] = 255
    free_space[3:9, 7:13] = 255
    free_space[10:12, 14:16] = 255

    room = _connected_interior(free_space, minimum_area=20)

    assert np.all(room[3:9, 7:13] == 255)
    assert np.all(room[0:4, 1:5] == 0)
    assert np.all(room[10:12, 14:16] == 0)
