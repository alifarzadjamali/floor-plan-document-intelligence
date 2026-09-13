from __future__ import annotations

import argparse
import json
from pathlib import Path

from floorplan_di.training.train import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SegFormer-B0 on cached CubiCasa5K masks")
    parser.add_argument("--config", type=Path, default=Path("configs/segmentation.yaml"))
    parser.add_argument("--cache-root", type=Path, default=Path("data/cache/segmentation_512"))
    parser.add_argument("--output", type=Path, default=Path("results/segmentation/phase2_dev"))
    args = parser.parse_args()
    print(json.dumps(train(args.config, args.cache_root, args.output), indent=2))


if __name__ == "__main__":
    main()
