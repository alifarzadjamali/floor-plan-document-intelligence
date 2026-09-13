from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from floorplan_di.ocr.synthetic_benchmark import generate_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the reproducible synthetic technical-plan OCR benchmark"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/ocr.yaml"))
    parser.add_argument("--output", type=Path, default=Path("data/synthetic_ocr"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    print(json.dumps(generate_benchmark(config, args.output), indent=2))


if __name__ == "__main__":
    main()
