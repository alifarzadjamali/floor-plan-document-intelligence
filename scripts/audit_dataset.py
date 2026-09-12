from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

from floorplan_di.constants import CLASS_NAMES
from floorplan_di.data import CubiCasaDataset, rasterize_annotation
from floorplan_di.visualization import (
    colourize_mask,
    overlay_mask,
    save_contact_sheet,
    source_annotation_image,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    data = np.asarray(values, dtype=np.float64)
    return {
        "min": float(data.min()),
        "p25": float(np.percentile(data, 25)),
        "median": float(np.median(data)),
        "p75": float(np.percentile(data, 75)),
        "max": float(data.max()),
        "mean": float(data.mean()),
    }


def _choose_examples(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    valid = [row for row in rows if not row.get("error")]
    if not valid:
        return {}
    return {
        "simple": min(
            (row for row in valid if row["room_objects"] > 0),
            key=lambda row: row["room_objects"],
            default=min(valid, key=lambda row: row["room_objects"]),
        ),
        "dense": max(valid, key=lambda row: row["room_objects"]),
        "high_resolution": max(valid, key=lambda row: row["pixels"]),
        "low_resolution": min(valid, key=lambda row: row["pixels"]),
        "many_openings": max(valid, key=lambda row: row["door_objects"] + row["window_objects"]),
        "unusual_aspect": max(valid, key=lambda row: abs(np.log(row["aspect_ratio"]))),
    }


def _save_example(
    data_root: Path, split: str, label: str, row: dict[str, Any], output: Path
) -> None:
    sample = next(item for item in CubiCasaDataset(data_root, split) if item.sample_id == row["id"])
    image = np.asarray(Image.open(sample.image_path).convert("RGB"))
    annotation = rasterize_annotation(sample.annotation_path, image.shape[0], image.shape[1])
    save_contact_sheet(
        [
            ("Original", image),
            ("Source polygons", source_annotation_image(image, annotation.source_polygons)),
            ("Collapsed mask", colourize_mask(annotation.mask)),
            ("Overlay", overlay_mask(image, annotation.mask)),
        ],
        output / "examples" / f"{split}_{label}.png",
    )


def audit_split(
    data_root: Path, split: str, output: Path, max_samples: int | None
) -> dict[str, Any]:
    dataset = CubiCasaDataset(data_root, split)
    samples = dataset.samples[:max_samples]
    rows: list[dict[str, Any]] = []
    total_pixels = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    ignored = Counter()

    for sample in tqdm(samples, desc=f"Auditing {split}"):
        row: dict[str, Any] = {"id": sample.sample_id}
        try:
            image_exists = sample.image_path.is_file()
            svg_exists = sample.annotation_path.is_file()
            row.update(image_present=image_exists, svg_present=svg_exists)
            if not image_exists or not svg_exists:
                raise FileNotFoundError("F1_scaled.png or model.svg is missing")
            with Image.open(sample.image_path) as image:
                image.verify()
            with Image.open(sample.image_path) as image:
                width, height = image.size
            annotation = rasterize_annotation(sample.annotation_path, height, width)
            counts = np.bincount(annotation.mask.ravel(), minlength=len(CLASS_NAMES))
            total_pixels += counts
            ignored.update(annotation.ignored_counts)
            row.update(
                width=width,
                height=height,
                pixels=width * height,
                aspect_ratio=width / height,
                room_objects=annotation.object_counts["room"],
                wall_objects=annotation.object_counts["wall"],
                door_objects=annotation.object_counts["door"],
                window_objects=annotation.object_counts["window"],
                **{f"{name}_pixels": int(counts[index]) for index, name in enumerate(CLASS_NAMES)},
            )
        except Exception as exc:  # audit must retain every malformed record
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    valid = [row for row in rows if not row.get("error")]
    failures = [row for row in rows if row.get("error")]
    pixel_sum = int(total_pixels.sum())
    report = {
        "split": split,
        "official_split_size": len(dataset),
        "audited_samples": len(samples),
        "valid_samples": len(valid),
        "failures": failures,
        "source_files": {
            "images_present": sum(bool(row.get("image_present")) for row in rows),
            "annotations_present": sum(bool(row.get("svg_present")) for row in rows),
        },
        "samples_without_target_pixels": sum(
            row.get("room_pixels", 0)
            + row.get("wall_pixels", 0)
            + row.get("door_pixels", 0)
            + row.get("window_pixels", 0)
            == 0
            for row in valid
        ),
        "split_file_sha256": _sha256(data_root / f"{split}.txt"),
        "image_width": _distribution([row["width"] for row in valid]),
        "image_height": _distribution([row["height"] for row in valid]),
        "aspect_ratio": _distribution([row["aspect_ratio"] for row in valid]),
        "room_count": _distribution([row["room_objects"] for row in valid]),
        "wall_count": _distribution([row["wall_objects"] for row in valid]),
        "door_count": _distribution([row["door_objects"] for row in valid]),
        "window_count": _distribution([row["window_objects"] for row in valid]),
        "object_totals": {
            category: int(sum(row[f"{category}_objects"] for row in valid))
            for category in ("room", "wall", "door", "window")
        },
        "pixel_totals": {name: int(total_pixels[index]) for index, name in enumerate(CLASS_NAMES)},
        "pixel_frequencies": {
            name: float(total_pixels[index] / pixel_sum) if pixel_sum else 0.0
            for index, name in enumerate(CLASS_NAMES)
        },
        "ignored_source_objects": dict(ignored),
        "selected_examples": {label: row["id"] for label, row in _choose_examples(rows).items()},
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{split}_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    if rows:
        columns = sorted({key for row in rows for key in row})
        with (output / f"{split}_samples.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    for label, row in _choose_examples(rows).items():
        _save_example(data_root, split, label, row, output)
    return report


def _write_overview(reports: list[dict[str, Any]], output: Path) -> None:
    lines = [
        "# CubiCasa5K dataset audit",
        "",
        "Generated from the official split files and the verified five-class mapping.",
        "",
        "| Split | Official | Audited | Valid | Failures | Rooms | Walls | Doors | Windows |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for report in reports:
        totals = report["object_totals"]
        lines.append(
            f"| {report['split']} | {report['official_split_size']} | {report['audited_samples']} "
            f"| {report['valid_samples']} | {len(report['failures'])} | {totals['room']} "
            f"| {totals['wall']} | {totals['door']} | {totals['window']} |"
        )
    lines.append("")
    if any(report["split"] == "test" for report in reports):
        lines.append("Test was audited only after the baseline configuration was frozen.")
        lines.append("")
    lines.extend(
        [
            "Detailed distributions, class frequencies, checksums, failures and selected "
            "example IDs are in the JSON reports.",
            "",
        ]
    )
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit official CubiCasa5K splits")
    parser.add_argument("--data-root", type=Path, default=Path("data/cubicasa5k"))
    parser.add_argument("--output", type=Path, default=Path("results/dataset_audit"))
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "val", "test"), default=["train", "val"]
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args()
    if "test" in args.splits and not args.allow_test:
        raise SystemExit("Refusing to read test data without --allow-test")
    reports = [
        audit_split(args.data_root, split, args.output, args.max_samples) for split in args.splits
    ]
    _write_overview(reports, args.output)


if __name__ == "__main__":
    main()
