from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from floorplan_di.data.cubicasa import CubiCasaSample
from floorplan_di.data.masks import rasterize_annotation

IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


@dataclass(frozen=True)
class AugmentationConfig:
    brightness_contrast_probability: float = 0.5
    gaussian_noise_probability: float = 0.15
    blur_probability: float = 0.1
    rotation_degrees: float = 5.0
    jpeg_probability: float = 0.15
    jpeg_quality_min: int = 75
    jpeg_quality_max: int = 95


def letterbox_image_and_mask(
    image: np.ndarray, mask: np.ndarray, size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Resize without geometric distortion and pad with white/background."""
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    target_width = max(1, round(width * scale))
    target_height = max(1, round(height * scale))
    resized_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    canvas_image = np.full((size, size, 3), 255, dtype=np.uint8)
    canvas_mask = np.zeros((size, size), dtype=np.uint8)
    top = (size - target_height) // 2
    left = (size - target_width) // 2
    canvas_image[top : top + target_height, left : left + target_width] = resized_image
    canvas_mask[top : top + target_height, left : left + target_width] = resized_mask
    return canvas_image, canvas_mask


def augment_image_and_mask(
    image: np.ndarray, mask: np.ndarray, config: AugmentationConfig, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    result_image = image.copy()
    result_mask = mask.copy()
    height, width = image.shape[:2]
    angle = float(rng.uniform(-config.rotation_degrees, config.rotation_degrees))
    if abs(angle) > 0.05:
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        result_image = cv2.warpAffine(
            result_image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        result_mask = cv2.warpAffine(
            result_mask,
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    if rng.random() < config.brightness_contrast_probability:
        alpha = float(rng.uniform(0.86, 1.14))
        beta = float(rng.uniform(-16, 16))
        result_image = cv2.convertScaleAbs(result_image, alpha=alpha, beta=beta)
    if rng.random() < config.gaussian_noise_probability:
        noise = rng.normal(0, 5, size=result_image.shape).astype(np.float32)
        result_image = np.clip(result_image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < config.blur_probability:
        result_image = cv2.GaussianBlur(result_image, (3, 3), sigmaX=0.7)
    if rng.random() < config.jpeg_probability:
        quality = int(rng.integers(config.jpeg_quality_min, config.jpeg_quality_max + 1))
        encoded, payload = cv2.imencode(".jpg", result_image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if encoded:
            result_image = cv2.imdecode(payload, cv2.IMREAD_COLOR)
            result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    return result_image, result_mask


def normalise_image(image: np.ndarray) -> torch.Tensor:
    values = image.astype(np.float32) / 255.0
    values = (values - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.moveaxis(values, -1, 0).copy())


def cache_sample(
    sample: CubiCasaSample, image_size: int, image_path: Path, mask_path: Path
) -> dict[str, Any]:
    source_bgr = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
    if source_bgr is None:
        raise OSError(f"Could not read {sample.image_path}")
    image = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
    annotation = rasterize_annotation(sample.annotation_path, image.shape[0], image.shape[1])
    letterboxed_image, letterboxed_mask = letterbox_image_and_mask(
        image, annotation.mask, image_size
    )
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(image_path), cv2.cvtColor(letterboxed_image, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not write {image_path}")
    if not cv2.imwrite(str(mask_path), letterboxed_mask):
        raise OSError(f"Could not write {mask_path}")
    return {
        "id": sample.sample_id,
        "image": image_path.as_posix(),
        "mask": mask_path.as_posix(),
        "object_counts": annotation.object_counts,
        "pixel_counts": np.bincount(letterboxed_mask.ravel(), minlength=5).tolist(),
    }


class CachedSegmentationDataset(Dataset[dict[str, torch.Tensor | str]]):
    def __init__(
        self,
        manifest_path: Path,
        training: bool = False,
        augmentation: AugmentationConfig | None = None,
        seed: int = 42,
    ) -> None:
        self.records = [
            json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()
        ]
        if not self.records:
            raise ValueError(f"No cached records in {manifest_path}")
        self.training = training
        self.augmentation = augmentation or AugmentationConfig()
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        image_bgr = cv2.imread(record["image"], cv2.IMREAD_COLOR)
        mask = cv2.imread(record["mask"], cv2.IMREAD_GRAYSCALE)
        if image_bgr is None or mask is None:
            raise OSError(f"Could not load cached sample {record['id']}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if self.training:
            rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index)
            image, mask = augment_image_and_mask(image, mask, self.augmentation, rng)
        return {
            "pixel_values": normalise_image(image),
            "labels": torch.from_numpy(mask.astype(np.int64)),
            "id": record["id"],
        }


def select_training_samples(
    samples: list[CubiCasaSample], size: int, seed: int
) -> list[CubiCasaSample]:
    if size > len(samples):
        raise ValueError(f"Requested {size} training plans but split contains only {len(samples)}")
    selected = random.Random(seed).sample(samples, size)
    return sorted(selected, key=lambda sample: sample.sample_id)
