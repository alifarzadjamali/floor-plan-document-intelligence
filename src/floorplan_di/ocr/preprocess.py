from __future__ import annotations

import cv2
import numpy as np


def preprocess(image_rgb: np.ndarray, method: str) -> np.ndarray:
    """Prepare an RGB floor-plan image for the configured OCR method."""
    if method == "raw_upscaled":
        # Keep the source colours while giving Tesseract a larger glyph image.
        return cv2.resize(image_rgb, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    if method == "adaptive_threshold":
        # Thresholding works on grayscale after the same 2x enlargement.
        enlarged = cv2.resize(image_rgb, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
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
