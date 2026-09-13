from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from floorplan_di.evaluation.segmentation import evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a frozen SegFormer checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/segmentation.yaml"))
    parser.add_argument("--cache-root", type=Path, default=Path("data/cache/segmentation_512"))
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output", type=Path, default=Path("results/segmentation/evaluation"))
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args()
    if args.split == "test" and not args.allow_test:
        raise SystemExit("Refusing to evaluate the test split without --allow-test")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = args.output / args.split
    result = evaluate_checkpoint(
        args.checkpoint,
        args.cache_root / "manifests" / f"{args.split}.jsonl",
        output,
        config["batch_size"],
        config["num_workers"],
        args.examples,
    )
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
