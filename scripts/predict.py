from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from floorplan_di.inference import FloorPlanPipeline
from floorplan_di.inference.visualization import structured_overlay


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert one floor-plan raster into approximate structured geometry"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/prediction"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("results/segmentation/phase2_dev/checkpoints/best.pt"),
    )
    parser.add_argument(
        "--segmentation-config", type=Path, default=Path("configs/segmentation.yaml")
    )
    parser.add_argument("--ocr-config", type=Path, default=Path("configs/ocr.yaml"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    pipeline = FloorPlanPipeline(
        args.checkpoint, args.segmentation_config, args.ocr_config, args.device
    )
    document, mask = pipeline.run(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "structured_output.json").write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf-8"
    )
    image = __import__("numpy").asarray(Image.open(args.input).convert("RGB"))
    Image.fromarray(structured_overlay(image, mask, document)).save(args.output / "overlay.png")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "rooms": len(document["rooms"]),
                "doors": len(document["doors"]),
                "windows": len(document["windows"]),
                "text_entities": len(document["text_entities"]),
                "warnings": len(document["review_warnings"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
