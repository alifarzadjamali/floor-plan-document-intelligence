import numpy as np
import pytest

from floorplan_di.ocr import tesseract_engine
from floorplan_di.ocr.tesseract_engine import OCRResult, recognise_crop


def test_recognise_crop_skips_duplicate_orientations(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_recognise(
        image: np.ndarray, method: str, page_segmentation_mode: int, orientation: int
    ) -> OCRResult:
        calls.append(orientation)
        return OCRResult(str(orientation), orientation / 100, orientation, ())

    monkeypatch.setattr(tesseract_engine, "_recognise_oriented", fake_recognise)

    result = recognise_crop(np.zeros((4, 4, 3), dtype=np.uint8), "raw", orientations=(0, 0, 90))

    assert calls == [0, 90]
    assert result.orientation == 90


@pytest.mark.parametrize("orientations", [(), (0, 45)])
def test_recognise_crop_rejects_invalid_orientations_before_ocr(
    monkeypatch: pytest.MonkeyPatch, orientations: tuple[int, ...]
) -> None:
    def unexpected_call(*args: object, **kwargs: object) -> OCRResult:
        raise AssertionError("OCR should not run for invalid orientations")

    monkeypatch.setattr(tesseract_engine, "_recognise_oriented", unexpected_call)

    with pytest.raises(ValueError, match="orientation"):
        recognise_crop(np.zeros((4, 4, 3), dtype=np.uint8), "raw", orientations=orientations)
