from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from floorplan_di.data.cubicasa import load_official_split
from floorplan_di.data.masks import rasterize_annotation
from floorplan_di.evaluation.structured import summarise_structured_records
from floorplan_di.inference import FloorPlanPipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen end-to-end structured extraction on official held-out plans"
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data/cubicasa5k"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--max-samples", type=int, default=24)
    parser.add_argument("--output", type=Path, default=Path("results/phase5/structured_evaluation"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if args.split == "test" and not args.allow_test:
        raise SystemExit("Refusing to read the held-out test split without --allow-test")
    if args.max_samples < 1:
        raise SystemExit("--max-samples must be positive")

    pipeline = FloorPlanPipeline(
        args.checkpoint, Path("configs/segmentation.yaml"), Path("configs/ocr.yaml"), args.device
    )
    samples = load_official_split(args.dataset_root, args.split)[: args.max_samples]
    records: list[dict[str, object]] = []
    total_seconds = 0.0
    for index, sample in enumerate(samples, start=1):
        started = time.perf_counter()
        document, _, image = pipeline.run(sample.image_path)
        elapsed = time.perf_counter() - started
        total_seconds += elapsed
        truth = rasterize_annotation(sample.annotation_path, image.shape[0], image.shape[1])
        entities = [
            item for category in ("rooms", "doors", "windows") for item in document[category]
        ]
        vectorized = sum(len(item["polygon"]) >= 3 for item in entities)
        records.append(
            {
                "sample_id": sample.sample_id,
                **{
                    f"predicted_{category}": len(document[category])
                    for category in ("rooms", "doors", "windows")
                },
                **{
                    f"ground_truth_{plural}": truth.object_counts[singular]
                    for singular, plural in (
                        ("room", "rooms"),
                        ("door", "doors"),
                        ("window", "windows"),
                    )
                },
                "attempted_entities": len(entities),
                "vectorized_entities": vectorized,
                "runtime_seconds": round(elapsed, 3),
                "review_warnings": len(document["review_warnings"]),
            }
        )
        print(f"[{index}/{len(samples)}] {sample.sample_id}: {elapsed:.2f}s")
    report = summarise_structured_records(records)
    report["runtime"] = {
        "mean_end_to_end_seconds_per_plan": total_seconds / len(samples),
        "device": str(pipeline.device),
        "includes": "image load, segmentation, OCR, geometry, linking, and JSON construction",
    }
    report["evaluation_scope"] = {
        "split": args.split,
        "sample_selection": f"first {len(samples)} official split entries; no tuning performed",
        "checkpoint": str(args.checkpoint),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
