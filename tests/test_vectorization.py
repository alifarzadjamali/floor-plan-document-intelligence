import numpy as np
import pytest

from floorplan_di.constants import PlanClass
from floorplan_di.geometry.vectorize import vectorize_mask


def test_vectorize_mask_emits_pixel_polygons_and_opening_orientation() -> None:
    mask = np.zeros((100, 120), dtype=np.uint8)
    mask[10:60, 10:50] = PlanClass.ROOM
    mask[65:95, 70:110] = PlanClass.ROOM
    mask[20:30, 55:65] = PlanClass.WALL
    mask[45:53, 48:64] = PlanClass.DOOR
    result = vectorize_mask(mask, room_min_area=20, wall_min_area=10, opening_min_area=5)
    assert [room["id"] for room in result["rooms"]] == ["R01", "R02"]
    assert result["doors"][0]["id"] == "D01"
    assert result["doors"][0]["orientation_degrees"] is not None
    assert len(result["rooms"][0]["polygon"]) >= 4
    assert result["rooms"][0]["confidence_band"] is None
    assert result["rooms"][0]["geometry_valid"] is True
    assert result["walls"][0]["centerline"] is not None


def test_vectorize_mask_labels_probability_confidence() -> None:
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[2:18, 2:18] = PlanClass.DOOR
    probability = np.zeros((5, 20, 20), dtype=np.float32)
    probability[PlanClass.DOOR, 2:18, 2:18] = 0.92
    result = vectorize_mask(mask, probability, opening_min_area=5)
    assert result["doors"][0]["confidence_band"] == "high"


def test_vectorize_mask_rejects_unknown_class_ids() -> None:
    mask = np.asarray([[0, 5]], dtype=np.uint8)

    with pytest.raises(ValueError, match="class ids"):
        vectorize_mask(mask)


def test_vectorize_mask_rejects_wrong_probability_channels() -> None:
    mask = np.zeros((4, 4), dtype=np.uint8)
    probability = np.zeros((4, 4, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="probability must have shape"):
        vectorize_mask(mask, probability)


def test_vectorize_mask_rejects_invalid_geometry_parameters() -> None:
    mask = np.zeros((4, 4), dtype=np.uint8)

    with pytest.raises(ValueError, match="component areas"):
        vectorize_mask(mask, opening_min_area=0)
    with pytest.raises(ValueError, match="simplify_fraction"):
        vectorize_mask(mask, simplify_fraction=1.1)
