from __future__ import annotations

import os
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

from floorplan_di.ocr.preprocess import preprocess


@dataclass(frozen=True)
class OCRToken:
    text: str
    bbox: tuple[int, int, int, int]
    confidence: float


@dataclass(frozen=True)
class OCRResult:
    text: str
    confidence: float
    orientation: int
    tokens: tuple[OCRToken, ...]

    def as_dict(self) -> dict[str, object]:
        return {**asdict(self), "tokens": [asdict(token) for token in self.tokens]}


def configure_tesseract(command: str | Path | None = None) -> str:
    candidates = [
        command,
        os.environ.get("TESSERACT_CMD"),
        shutil.which("tesseract"),
        Path(sys.prefix) / "Tesseract-OCR" / "tesseract.exe",
    ]
    selected = next((Path(item) for item in candidates if item and Path(item).is_file()), None)
    if selected is None:
        raise FileNotFoundError(
            "Tesseract 5 executable was not found. Set TESSERACT_CMD or install it under .venv/Tesseract-OCR."
        )
    pytesseract.pytesseract.tesseract_cmd = str(selected)
    return str(selected)


def _rotate(image: np.ndarray, orientation: int) -> np.ndarray:
    if orientation == 0:
        return image
    if orientation == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if orientation == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if orientation == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Unsupported orientation: {orientation}")


def _unrotate_bbox(
    bbox: tuple[int, int, int, int], original_shape: tuple[int, int], orientation: int
) -> tuple[int, int, int, int]:
    x, y, width, height = bbox
    original_height, original_width = original_shape
    points = np.asarray(
        [[x, y], [x + width - 1, y], [x, y + height - 1], [x + width - 1, y + height - 1]]
    )
    if orientation == 90:
        points = np.column_stack((points[:, 1], original_height - 1 - points[:, 0]))
    elif orientation == 180:
        points = np.column_stack(
            (original_width - 1 - points[:, 0], original_height - 1 - points[:, 1])
        )
    elif orientation == 270:
        points = np.column_stack((original_width - 1 - points[:, 1], points[:, 0]))
    left, top = points.min(axis=0)
    right, bottom = points.max(axis=0)
    return int(left), int(top), int(right - left + 1), int(bottom - top + 1)


def _recognise_oriented(
    image_rgb: np.ndarray, method: str, page_segmentation_mode: int, orientation: int
) -> OCRResult:
    rotated = _rotate(image_rgb, orientation)
    processed = preprocess(rotated, method)
    data = pytesseract.image_to_data(
        processed,
        output_type=Output.DICT,
        config=f"--oem 1 --psm {page_segmentation_mode}",
    )
    scale = 2
    tokens: list[OCRToken] = []
    for index, raw_text in enumerate(data["text"]):
        text = raw_text.strip()
        confidence = float(data["conf"][index])
        if text and confidence >= 0:
            x = round(data["left"][index] / scale)
            y = round(data["top"][index] / scale)
            width = max(1, round(data["width"][index] / scale))
            height = max(1, round(data["height"][index] / scale))
            bbox = _unrotate_bbox((x, y, width, height), image_rgb.shape[:2], orientation)
            tokens.append(OCRToken(text, bbox, confidence / 100.0))
    mean_confidence = float(np.mean([token.confidence for token in tokens])) if tokens else 0.0
    return OCRResult(
        " ".join(token.text for token in tokens), mean_confidence, orientation, tuple(tokens)
    )


def recognise_crop(
    image_rgb: np.ndarray,
    method: str,
    page_segmentation_mode: int = 7,
    orientations: tuple[int, ...] = (0, 90, 270),
) -> OCRResult:
    candidates = [
        _recognise_oriented(image_rgb, method, page_segmentation_mode, orientation)
        for orientation in orientations
    ]
    return max(candidates, key=lambda result: (result.confidence, len(result.text)))


def recognise_page(
    image_rgb: np.ndarray,
    method: str,
    orientations: tuple[int, ...] = (0, 90, 270),
) -> OCRResult:
    """Run whole-page OCR under horizontal and vertical orientation hypotheses."""
    return recognise_crop(image_rgb, method, page_segmentation_mode=11, orientations=orientations)
