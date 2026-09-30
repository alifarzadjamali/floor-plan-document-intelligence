import numpy as np
import pytest

from floorplan_di.data.segmentation import letterbox_image_and_mask


def test_letterbox_preserves_classes_and_aspect_ratio() -> None:
    image = np.full((20, 40, 3), 20, dtype=np.uint8)
    mask = np.zeros((20, 40), dtype=np.uint8)
    mask[:, 10:20] = 3
    resized_image, resized_mask = letterbox_image_and_mask(image, mask, 80)

    assert resized_image.shape == (80, 80, 3)
    assert resized_mask.shape == (80, 80)
    assert set(np.unique(resized_mask)) == {0, 3}
    assert np.all(resized_image[:20] == 255)


def test_letterbox_rejects_mismatched_image_and_mask() -> None:
    image = np.zeros((20, 40, 3), dtype=np.uint8)
    mask = np.zeros((10, 40), dtype=np.uint8)

    with pytest.raises(ValueError, match="dimensions must match"):
        letterbox_image_and_mask(image, mask, 80)


@pytest.mark.parametrize("size", [0, -1])
def test_letterbox_rejects_non_positive_output_size(size: int) -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=np.uint8)

    with pytest.raises(ValueError, match="size must be positive"):
        letterbox_image_and_mask(image, mask, size)
