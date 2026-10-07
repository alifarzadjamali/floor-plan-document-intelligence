import numpy as np
import pytest

from floorplan_di.ocr.preprocess import preprocess


def test_preprocess_upscales_valid_rgb_images() -> None:
    image = np.zeros((4, 6, 3), dtype=np.uint8)

    result = preprocess(image, "raw_upscaled")

    assert result.shape == (8, 12, 3)


@pytest.mark.parametrize(
    ("image", "message"),
    [
        (np.zeros((4, 6), dtype=np.uint8), "shape"),
        (np.zeros((0, 6, 3), dtype=np.uint8), "must not be empty"),
        (np.zeros((4, 6, 3), dtype=np.float32), "uint8"),
    ],
)
def test_preprocess_rejects_invalid_image_arrays(image: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        preprocess(image, "raw_upscaled")
