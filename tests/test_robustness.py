import numpy as np
import pytest

from floorplan_di.evaluation.robustness import PERTURBATIONS, perturb


def test_perturbations_keep_image_shape_and_are_deterministic() -> None:
    image = np.full((24, 32, 3), 120, dtype=np.uint8)
    for name in PERTURBATIONS:
        first = perturb(image, name)
        assert first.shape == image.shape
        assert np.array_equal(first, perturb(image, name))


def test_mask_rotation_preserves_class_ids() -> None:
    mask = np.zeros((24, 32), dtype=np.uint8)
    mask[8:16, 10:20] = 4
    assert set(np.unique(perturb(mask, "rotation", is_mask=True))) <= {0, 4}


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((24, 32), dtype=np.uint8),
        np.zeros((24, 32, 3), dtype=np.float32),
        np.zeros((0, 32, 3), dtype=np.uint8),
    ],
)
def test_perturb_rejects_invalid_rgb_inputs(image: np.ndarray) -> None:
    with pytest.raises(ValueError):
        perturb(image, "clean")


def test_perturb_rejects_non_integer_masks() -> None:
    with pytest.raises(ValueError, match="2-D integer"):
        perturb(np.zeros((24, 32), dtype=np.float32), "rotation", is_mask=True)
