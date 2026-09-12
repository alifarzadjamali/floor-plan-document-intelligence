from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

from floorplan_di.data import CubiCasaDataset, rasterize_annotation


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact five-class CubiCasa masks")
    parser.add_argument("--data-root", type=Path, default=Path("data/cubicasa5k"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/masks"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args()
    if args.split == "test" and not args.allow_test:
        raise SystemExit("Refusing to read test data without --allow-test")

    dataset = CubiCasaDataset(args.data_root, args.split)
    samples = dataset.samples[: args.max_samples]
    for sample in tqdm(samples, desc=f"Building {args.split} masks"):
        width, height = sample.image_size()
        result = rasterize_annotation(sample.annotation_path, height, width)
        target = args.output / args.split / f"{sample.sample_id}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(target), result.mask):
            raise OSError(f"Could not write {target}")


if __name__ == "__main__":
    main()
