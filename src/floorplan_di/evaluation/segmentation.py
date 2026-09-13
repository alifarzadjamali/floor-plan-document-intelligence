from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader
from tqdm import tqdm

from floorplan_di.data.segmentation import CachedSegmentationDataset
from floorplan_di.evaluation.segmentation_metrics import ConfusionMatrix
from floorplan_di.models.segformer import build_segformer
from floorplan_di.visualization import colourize_mask, overlay_mask, save_contact_sheet


def load_checkpoint(
    checkpoint_path: Path, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_segformer(checkpoint["config"]["model_name"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), checkpoint


def evaluate_checkpoint(
    checkpoint_path: Path,
    manifest_path: Path,
    output: Path,
    batch_size: int,
    num_workers: int,
    examples: int = 8,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    dataset = CachedSegmentationDataset(manifest_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        shuffle=False,
    )
    confusion = ConfusionMatrix()
    durations: list[float] = []
    output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for batch_index, batch in enumerate(tqdm(loader, desc="Evaluating segmentation")):
            images = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"]
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = perf_counter()
            logits = model(pixel_values=images).logits
            logits = functional.interpolate(
                logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            durations.append((perf_counter() - start) / images.shape[0])
            predictions = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
            targets = labels.numpy().astype(np.uint8)
            confusion.update(targets, predictions)
            if batch_index * batch_size < examples:
                start_index = batch_index * batch_size
                for offset, (target, prediction) in enumerate(
                    zip(targets, predictions, strict=True)
                ):
                    absolute_index = start_index + offset
                    if absolute_index >= examples:
                        break
                    record = dataset.records[absolute_index]
                    source_bgr = cv2.imread(record["image"], cv2.IMREAD_COLOR)
                    image = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
                    save_contact_sheet(
                        [
                            ("Input", image),
                            ("Ground truth", colourize_mask(target)),
                            ("SegFormer-B0", colourize_mask(prediction)),
                            ("Prediction overlay", overlay_mask(image, prediction)),
                        ],
                        output / "examples" / f"{absolute_index:03d}.png",
                    )
    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint["epoch"],
        "samples": len(dataset),
        "device": str(device),
        "runtime_seconds_per_plan": {
            "mean": float(np.mean(durations)),
            "median": float(np.median(durations)),
            "includes_image_io": False,
        },
        "metrics": confusion.compute(),
    }
    return result
