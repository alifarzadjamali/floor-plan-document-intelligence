"""Deterministic, realistic document degradation transforms."""

from __future__ import annotations

import cv2
import numpy as np

PERTURBATIONS = (
    "clean",
    "brightness",
    "contrast",
    "blur",
    "noise",
    "jpeg",
    "rotation",
    "down_up",
)


def perturb(image: np.ndarray, name: str, seed: int = 42, is_mask: bool = False) -> np.ndarray:
    """Apply one mild transform; masks only receive the geometric rotation."""
    if name not in PERTURBATIONS:
        raise ValueError(f"Unknown perturbation: {name}")
    if name == "clean" or (is_mask and name != "rotation"):
        return image.copy()
    if name == "brightness":
        return np.clip(image.astype(np.int16) + 20, 0, 255).astype(np.uint8)
    if name == "contrast":
        return cv2.convertScaleAbs(image, alpha=1.18, beta=-18)
    if name == "blur":
        return cv2.GaussianBlur(image, (3, 3), 0.7)
    if name == "noise":
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, 5, image.shape)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if name == "jpeg":
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [
            cv2.IMWRITE_JPEG_QUALITY,
            80,
        ])
        if not ok:  # pragma: no cover - OpenCV encoding failure is environmental
            raise RuntimeError("Could not apply JPEG compression")
        return cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if name == "rotation":
        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 3.0, 1.0)
        border = 0 if is_mask else (255, 255, 255)
        interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        return cv2.warpAffine(
            image, matrix, (width, height), flags=interpolation, borderValue=border
        )
    height, width = image.shape[:2]
    smaller = cv2.resize(
        image, (max(1, width // 2), max(1, height // 2)), interpolation=cv2.INTER_AREA
    )
    return cv2.resize(smaller, (width, height), interpolation=cv2.INTER_CUBIC)
