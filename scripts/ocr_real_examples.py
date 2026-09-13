from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

from floorplan_di.data import CubiCasaDataset
from floorplan_di.ocr.entities import classify_entity
from floorplan_di.ocr.tesseract_engine import configure_tesseract, recognise_page


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create qualitative Tesseract results for real validation plans"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/ocr.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data/cubicasa5k"))
    parser.add_argument("--output", type=Path, default=Path("results/ocr/real_examples"))
    parser.add_argument("--count", type=int, default=3)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    selected_method = json.loads(
        Path("results/ocr/preprocessing_selection.json").read_text(encoding="utf-8")
    )["best_method"]
    configure_tesseract()
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    for index, sample in enumerate(CubiCasaDataset(args.data_root, "val").samples[: args.count]):
        image = np.asarray(Image.open(sample.image_path).convert("RGB"))
        result = recognise_page(
            image,
            selected_method,
            tuple(int(angle) for angle in config["tesseract"]["orientation_candidates"]),
        )
        canvas = Image.fromarray(image)
        draw = ImageDraw.Draw(canvas)
        for token in result.tokens:
            x, y, width, height = token.bbox
            draw.rectangle(
                (x, y, x + width, y + height), outline="red", width=max(1, image.shape[0] // 800)
            )
            draw.text((x, max(0, y - 12)), token.text, fill="red")
        canvas.save(args.output / f"{index:02d}.png")
        reports.append(
            {
                "sample_id": sample.sample_id,
                "method": selected_method,
                "result": result.as_dict(),
                "entity_types": [classify_entity(token.text) for token in result.tokens],
            }
        )
    (args.output / "results.json").write_text(
        json.dumps(reports, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
