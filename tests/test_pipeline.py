from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from floorplan_di.inference.pipeline import FloorPlanPipeline, _resize_probabilities


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


def test_probability_resize_preserves_single_class_axis() -> None:
    probability = np.ones((1, 2, 3), dtype=np.float32)
    resized = _resize_probabilities(probability, width=7, height=5)
    assert resized.shape == (1, 5, 7)
    np.testing.assert_allclose(resized, 1.0)


@pytest.mark.parametrize("shape", [(1, 100), (100, 1), (7, 11)])
def test_segment_restores_extreme_aspect_ratios_without_empty_crops(shape: tuple[int, int]) -> None:
    # Bypass checkpoint and OCR setup; no external models are needed for this regression.
    pipeline = object.__new__(FloorPlanPipeline)
    pipeline.device = torch.device("cpu")
    pipeline.image_size = 8
    logits = torch.zeros(1, 5, 2, 2)
    logits[:, 2] = 3.0
    pipeline.model = lambda **kwargs: SimpleNamespace(logits=logits)

    mask, probabilities = pipeline.segment(np.full((*shape, 3), 255, dtype=np.uint8))

    assert mask.shape == shape
    assert probabilities.shape == (5, *shape)
    assert np.all(mask == 2)
    np.testing.assert_allclose(probabilities.sum(axis=0), 1.0, atol=1e-6)
