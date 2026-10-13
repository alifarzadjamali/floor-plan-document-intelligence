import numpy as np
import pytest

from floorplan_di.data.segmentation import (
    AugmentationConfig,
    augment_image_and_mask,
    letterbox_image_and_mask,
)


def test_letterbox_preserves_classes_and_aspect_ratio() -> None:
    image = np.full((20, 40, 3), 20, dtype=np.uint8)
    mask = np.zeros((20, 40), dtype=np.uint8)
    mask[:, 10:20] = 3
    resized_image, resized_mask = letterbox_image_and_mask(image, mask, 80)

    assert resized_image.shape == (80, 80, 3)
    assert resized_mask.shape == (80, 80)
    assert set(np.unique(resized_mask)) == {0, 3}
    assert np.all(resized_image[:20] == 255)


def test_jpeg_augmentation_preserves_rgb_channels_and_mask() -> None:
    image = np.full((32, 32, 3), (230, 50, 10), dtype=np.uint8)
    mask = np.ones((32, 32), dtype=np.uint8)
    config = AugmentationConfig(
        rotation_degrees=0,
        brightness_contrast_probability=0,
        gaussian_noise_probability=0,
        blur_probability=0,
        jpeg_probability=1,
        jpeg_quality_min=95,
        jpeg_quality_max=95,
    )

    augmented, augmented_mask = augment_image_and_mask(
        image, mask, config, np.random.default_rng(42)
    )

    np.testing.assert_allclose(augmented, image, atol=5)
    np.testing.assert_array_equal(augmented_mask, mask)


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
