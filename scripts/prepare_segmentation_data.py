from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from floorplan_di.data.cubicasa import CubiCasaDataset
from floorplan_di.data.segmentation import cache_sample, select_training_samples


def _write_manifest(records: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def _prepare_split(
    samples: list, split: str, image_size: int, cache_root: Path, force: bool
) -> list[dict[str, object]]:
    records = []
    for sample in tqdm(samples, desc=f"Caching {split}"):
        stem = Path(sample.sample_id)
        image_path = cache_root / "images" / split / stem.with_suffix(".jpg")
        mask_path = cache_root / "masks" / split / stem.with_suffix(".png")
        if not force and image_path.is_file() and mask_path.is_file():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise OSError(f"Could not read existing cache {mask_path}")
            records.append(
                {
                    "id": sample.sample_id,
                    "image": image_path.as_posix(),
                    "mask": mask_path.as_posix(),
                    "pixel_counts": np.bincount(mask.ravel(), minlength=5).tolist(),
                }
            )
        else:
            records.append(cache_sample(sample, image_size, image_path, mask_path))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache letterboxed CubiCasa images and five-class masks"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/segmentation.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data/cubicasa5k"))
    parser.add_argument("--cache-root", type=Path, default=Path("data/cache/segmentation_512"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.include_test and not args.allow_test:
        raise SystemExit("Refusing to prepare the test split without --allow-test")
    train = CubiCasaDataset(args.data_root, "train").samples
    validation = CubiCasaDataset(args.data_root, "val").samples
    selected_train = select_training_samples(train, config["train_subset"], config["seed"])
    train_records = _prepare_split(
        selected_train, "train", config["image_size"], args.cache_root, args.force
    )
    val_records = _prepare_split(
        validation, "val", config["image_size"], args.cache_root, args.force
    )
    _write_manifest(train_records, args.cache_root / "manifests" / "train.jsonl")
    _write_manifest(val_records, args.cache_root / "manifests" / "val.jsonl")
    test_records: list[dict[str, object]] = []
    if args.include_test:
        test = CubiCasaDataset(args.data_root, "test").samples
        test_records = _prepare_split(
            test, "test", config["image_size"], args.cache_root, args.force
        )
        _write_manifest(test_records, args.cache_root / "manifests" / "test.jsonl")
    summary = {
        "image_size": config["image_size"],
        "seed": config["seed"],
        "train_samples": len(train_records),
        "validation_samples": len(val_records),
        "test_samples": len(test_records),
        "train_pixel_counts": np.asarray([record["pixel_counts"] for record in train_records])
        .sum(axis=0)
        .tolist(),
    }
    (args.cache_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
