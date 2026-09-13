from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

from floorplan_di.ocr.metrics import benchmark_metrics
from floorplan_di.ocr.tesseract_engine import configure_tesseract, recognise_crop


def evaluate(manifest: Path, method: str, config: dict[str, object]) -> dict[str, object]:
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    settings = config["tesseract"]
    predictions = []
    for record in tqdm(records, desc=f"OCR {method}"):
        image = np.asarray(Image.open(record["image"]).convert("RGB"))
        result = recognise_crop(
            image,
            method,
            int(settings["page_segmentation_mode"]),
            tuple(int(angle) for angle in settings["orientation_candidates"]),
        )
        predictions.append(
            {
                "id": record["id"],
                "reference": record["text"],
                "prediction": result.text,
                "ocr_confidence": result.confidence,
                "orientation": result.orientation,
            }
        )
    return {
        "method": method,
        "metrics": benchmark_metrics((row["reference"], row["prediction"]) for row in predictions),
        "predictions": predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select and evaluate Tesseract preprocessing on synthetic OCR data"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/ocr.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data/synthetic_ocr"))
    parser.add_argument("--output", type=Path, default=Path("results/ocr"))
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--method")
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args()
    if args.split == "test" and not args.allow_test:
        raise SystemExit("Refusing to evaluate the synthetic test split without --allow-test")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    configure_tesseract()
    methods = [args.method] if args.method else list(config["preprocessing_candidates"])
    reports = [
        evaluate(args.data_root / "manifests" / f"{args.split}.jsonl", method, config)
        for method in methods
    ]
    reports.sort(
        key=lambda report: (
            report["metrics"]["exact_match_accuracy"],
            -report["metrics"]["cer"],
        ),
        reverse=True,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    filename = (
        "preprocessing_selection.json"
        if args.split == "val" and args.method is None
        else f"{args.split}_metrics.json"
    )
    compact = [{"method": report["method"], "metrics": report["metrics"]} for report in reports]
    result = {"split": args.split, "best_method": reports[0]["method"], "reports": compact}
    (args.output / filename).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    prediction_path = args.output / f"{args.split}_{reports[0]['method']}_predictions.jsonl"
    prediction_path.write_text(
        "".join(json.dumps(row) + "\n" for row in reports[0]["predictions"]), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
