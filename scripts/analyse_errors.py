from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from floorplan_di.data.cubicasa import load_official_split
from floorplan_di.data.masks import rasterize_annotation
from floorplan_di.evaluation.error_analysis import choose_error_examples
from floorplan_di.evaluation.segmentation_metrics import ConfusionMatrix
from floorplan_di.inference import FloorPlanPipeline
from floorplan_di.inference.visualization import structured_overlay
from floorplan_di.visualization import colourize_mask, save_contact_sheet


def _foreground_iou(truth: np.ndarray, prediction: np.ndarray) -> float:
    metrics = ConfusionMatrix()
    metrics.update(truth, prediction)
    return float(metrics.compute()["mean_iou_foreground"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create fixed-rule Phase 5 error-analysis examples"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/cubicasa5k"))
    parser.add_argument("--plans", type=int, default=24)
    parser.add_argument("--output", type=Path, default=Path("results/phase5/error_analysis"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    pipeline = FloorPlanPipeline(
        args.checkpoint, Path("configs/segmentation.yaml"), Path("configs/ocr.yaml"), args.device
    )
    records: list[dict[str, object]] = []
    artifacts: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]] = {}
    for sample in load_official_split(args.dataset_root, "test")[: args.plans]:
        document, prediction, image = pipeline.run(sample.image_path)
        annotation = rasterize_annotation(sample.annotation_path, image.shape[0], image.shape[1])
        resolved = sum(bool(link["resolved"]) for link in document["door_to_room_links"])
        record = {
            "sample_id": sample.sample_id,
            "foreground_iou": round(_foreground_iou(annotation.mask, prediction), 4),
            "missed_openings": max(
                0,
                annotation.object_counts["door"]
                + annotation.object_counts["window"]
                - len(document["doors"])
                - len(document["windows"]),
            ),
            "false_openings": max(
                0,
                len(document["doors"])
                + len(document["windows"])
                - annotation.object_counts["door"]
                - annotation.object_counts["window"],
            ),
            "text_links": len(document["text_to_room_links"]),
            "unassigned_text": len(document["text_entities"]) - len(document["text_to_room_links"]),
            "resolved_doors": resolved,
            "unresolved_doors": len(document["door_to_room_links"]) - resolved,
        }
        records.append(record)
        artifacts[sample.sample_id] = (image, annotation.mask, prediction, document)
    chosen = choose_error_examples(records)
    args.output.mkdir(parents=True, exist_ok=True)
    for category, record in chosen.items():
        image, truth, prediction, document = artifacts[str(record["sample_id"])]
        save_contact_sheet(
            [
                ("Original", image),
                ("Ground-truth segmentation", colourize_mask(truth)),
                ("Predicted segmentation", colourize_mask(prediction)),
                ("OCR + structured overlay", structured_overlay(image, prediction, document)),
            ],
            args.output / f"{category}.png",
        )
    report = {
        "plans": len(records),
        "selection_rule": "fixed extrema over the first official held-out split entries",
        "examples": chosen,
        "all_records": records,
        "limitations": [
            "Opening count differences are proxies, not object matching metrics.",
            "OCR and link examples are qualitative because CubiCasa has no text transcription "
            "ground truth.",
        ],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
