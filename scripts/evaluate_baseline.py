from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

from floorplan_di.baseline import BaselineConfig, predict
from floorplan_di.data import CubiCasaDataset, rasterize_annotation
from floorplan_di.evaluation.segmentation_metrics import ConfusionMatrix
from floorplan_di.visualization import colourize_mask, overlay_mask, save_contact_sheet


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the frozen classical CV baseline")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data/cubicasa5k"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--output", type=Path, default=Path("results/baseline"))
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args()
    if args.split == "test" and not args.allow_test:
        raise SystemExit("Refusing to read test data without --allow-test")

    config = BaselineConfig.from_dict(yaml.safe_load(args.config.read_text(encoding="utf-8")))
    dataset = CubiCasaDataset(args.data_root, args.split)
    samples = dataset.samples[: args.max_samples]
    metrics = ConfusionMatrix()
    durations: list[float] = []
    example_dir = args.output / "examples"

    for index, sample in enumerate(tqdm(samples, desc=f"Evaluating {args.split}")):
        image = np.asarray(Image.open(sample.image_path).convert("RGB"))
        target = rasterize_annotation(sample.annotation_path, image.shape[0], image.shape[1]).mask
        start = time.perf_counter()
        prediction = predict(image, config)
        durations.append(time.perf_counter() - start)
        metrics.update(target, prediction)
        if index < args.examples:
            save_contact_sheet(
                [
                    ("Original", image),
                    ("Ground truth", colourize_mask(target)),
                    ("Baseline", colourize_mask(prediction)),
                    ("Baseline overlay", overlay_mask(image, prediction)),
                ],
                example_dir / f"{args.split}_{index:03d}.png",
            )

    result = {
        "method": "classical_cv_v1",
        "split": args.split,
        "samples": len(samples),
        "official_split_size": len(dataset),
        "config_sha256": _sha256(args.config),
        "split_file_sha256": _sha256(args.data_root / f"{args.split}.txt"),
        "runtime_seconds_per_plan": {
            "mean": float(np.mean(durations)),
            "median": float(np.median(durations)),
            "device": "CPU",
        },
        "metrics": metrics.compute(),
        "limitations": [
            "The baseline does not predict door or window classes.",
            "Furniture and text can be mistaken for wall structure.",
            "Open doorways can merge rooms with exterior whitespace.",
            "Curved and diagonal walls are weakly represented by axis-aligned morphology.",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / f"{args.split}_metrics.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
