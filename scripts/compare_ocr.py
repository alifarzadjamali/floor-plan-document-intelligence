"""Compare optional PaddleOCR output with the frozen Tesseract output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from floorplan_di.ocr.paddle_engine import paddle_available, recognise_page
from floorplan_di.ocr.tesseract_engine import configure_tesseract
from floorplan_di.ocr.tesseract_engine import recognise_page as tesseract_page


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/phase6/ocr_comparison.json"))
    args = parser.parse_args()
    image = np.asarray(Image.open(args.input).convert("RGB"))
    configure_tesseract()
    tesseract = tesseract_page(image, "raw_upscaled")
    report: dict[str, object] = {
        "input": str(args.input),
        "tesseract": {"tokens": len(tesseract.tokens), "confidence": tesseract.confidence},
        "paddleocr": {"available": paddle_available()},
    }
    if paddle_available():
        rows = recognise_page(image)
        report["paddleocr"] = {
            "available": True,
            "tokens": len(rows),
            "mean_confidence": sum(float(row["confidence"]) for row in rows) / len(rows)
            if rows
            else 0.0,
        }
    else:
        report["paddleocr"]["installation"] = "uv pip install -e '.[paddle]' plus PaddlePaddle"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
