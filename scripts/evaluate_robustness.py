from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from floorplan_di.data.cubicasa import load_official_split
from floorplan_di.data.masks import rasterize_annotation
from floorplan_di.evaluation.robustness import PERTURBATIONS, perturb
from floorplan_di.evaluation.segmentation_metrics import ConfusionMatrix
from floorplan_di.inference import FloorPlanPipeline
from floorplan_di.ocr.metrics import benchmark_metrics
from floorplan_di.ocr.tesseract_engine import recognise_crop


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure mild degradation robustness")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/cubicasa5k"))
    parser.add_argument("--ocr-root", type=Path, default=Path("data/synthetic_ocr"))
    parser.add_argument("--plans", type=int, default=12)
    parser.add_argument("--ocr-samples", type=int, default=24)
    parser.add_argument("--output", type=Path, default=Path("results/phase5/robustness"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    pipeline = FloorPlanPipeline(
        args.checkpoint, Path("configs/segmentation.yaml"), Path("configs/ocr.yaml"), args.device
    )
    plans = load_official_split(args.dataset_root, "test")[: args.plans]
    manifest = args.ocr_root / "manifests" / "test.jsonl"
    ocr_records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()][
        : args.ocr_samples
    ]
    report: dict[str, object] = {
        "scope": {
            "plans": len(plans),
            "ocr_samples": len(ocr_records),
            "split": "official segmentation test / synthetic OCR test",
            "severity": "mild, fixed parameters recorded in source",
        },
        "perturbations": {},
    }
    for name in PERTURBATIONS:
        matrix = ConfusionMatrix()
        for index, sample in enumerate(plans):
            image = np.asarray(Image.open(sample.image_path).convert("RGB"))
            truth = rasterize_annotation(
                sample.annotation_path, image.shape[0], image.shape[1]
            ).mask
            predicted, _ = pipeline.segment(perturb(image, name, seed=42 + index))
            matrix.update(perturb(truth, name, is_mask=True), predicted)
        pairs = []
        for index, record in enumerate(ocr_records):
            image = np.asarray(Image.open(record["image"]).convert("RGB"))
            result = recognise_crop(perturb(image, name, seed=500 + index), pipeline.ocr_method)
            pairs.append((record["text"], result.text))
        report["perturbations"][name] = {
            "segmentation": matrix.compute(),
            "ocr": benchmark_metrics(pairs),
        }
        print(f"completed {name}")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
