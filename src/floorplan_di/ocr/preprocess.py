from __future__ import annotations

import cv2
import numpy as np


def _validate_rgb_image(image_rgb: np.ndarray) -> None:
    if not isinstance(image_rgb, np.ndarray):
        raise TypeError("image_rgb must be a NumPy array")
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must have shape (height, width, 3)")
    if image_rgb.shape[0] == 0 or image_rgb.shape[1] == 0:
        raise ValueError("image_rgb must not be empty")
    if image_rgb.dtype != np.uint8:
        raise ValueError("image_rgb must use uint8 values")


def _upscale(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.resize(image_rgb, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)


def preprocess(image_rgb: np.ndarray, method: str) -> np.ndarray:
    """Prepare an RGB floor-plan image for the configured OCR method."""
    _validate_rgb_image(image_rgb)
    if method == "raw_upscaled":
        # Keep the source colours while giving Tesseract a larger glyph image.
        return _upscale(image_rgb)
    if method == "adaptive_threshold":
        # Thresholding works on grayscale after the same 2x enlargement.
        enlarged = _upscale(image_rgb)
        gray = cv2.cvtColor(enlarged, cv2.COLOR_RGB2GRAY)
        denoised = cv2.fastNlMeansDenoising(
            gray, None, h=6, templateWindowSize=7, searchWindowSize=21
        )
        return cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            9,
        )
    raise ValueError(f"Unknown OCR preprocessing method: {method}")
