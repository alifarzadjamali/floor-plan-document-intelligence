from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

from floorplan_di.baseline import BaselineConfig, predict
from floorplan_di.data import CubiCasaDataset, rasterize_annotation
from floorplan_di.evaluation.segmentation_metrics import ConfusionMatrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Small validation-only baseline parameter search")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data/cubicasa5k"))
    parser.add_argument("--output", type=Path, default=Path("results/baseline/tuning.json"))
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()

    base = BaselineConfig.from_dict(yaml.safe_load(args.config.read_text(encoding="utf-8")))
    candidates = [
        replace(
            base,
            line_kernel_fraction=line_fraction,
            wall_dilation_fraction=wall_dilation,
            room_barrier_dilation_fraction=barrier_dilation,
        )
        for line_fraction, wall_dilation, barrier_dilation in itertools.product(
            (0.008, 0.012, 0.018),
            (0.0015, 0.0025, 0.004),
            (0.0015, 0.003),
        )
    ]
    matrices = [ConfusionMatrix() for _ in candidates]
    samples = CubiCasaDataset(args.data_root, "val").samples[: args.max_samples]
    for sample in tqdm(samples, desc="Tuning on validation only"):
        image = np.asarray(Image.open(sample.image_path).convert("RGB"))
        target = rasterize_annotation(sample.annotation_path, image.shape[0], image.shape[1]).mask
        for config, matrix in zip(candidates, matrices, strict=True):
            matrix.update(target, predict(image, config))

    results = []
    for config, matrix in zip(candidates, matrices, strict=True):
        metrics = matrix.compute()
        room_iou = metrics["per_class"]["room"]["iou"]
        wall_iou = metrics["per_class"]["wall"]["iou"]
        results.append(
            {
                "config": asdict(config),
                "selection_score_room_wall_mean_iou": (room_iou + wall_iou) / 2,
                "metrics": metrics,
            }
        )
    results.sort(key=lambda item: item["selection_score_room_wall_mean_iou"], reverse=True)
    report = {
        "selection_split": "val",
        "samples": len(samples),
        "selection_rule": "maximum unweighted mean of room IoU and wall IoU",
        "best": results[0],
        "candidates": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key != "candidates"}, indent=2))


if __name__ == "__main__":
    main()
