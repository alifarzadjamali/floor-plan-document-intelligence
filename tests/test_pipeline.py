import cv2
import numpy as np

from floorplan_di.inference.pipeline import _resize_probabilities


def test_probability_channels_are_resized_together_without_changing_values() -> None:
    probability = np.arange(5 * 4 * 7, dtype=np.float32).reshape(5, 4, 7)
    expected = np.stack(
        [
            cv2.resize(channel, (11, 9), interpolation=cv2.INTER_LINEAR)
            for channel in probability
        ]
    )

    resized = _resize_probabilities(probability, width=11, height=9)

    np.testing.assert_allclose(resized, expected, rtol=1e-6, atol=1e-6)
    assert resized.shape == (5, 9, 11)
