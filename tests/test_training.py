import json
from pathlib import Path

import pytest
import torch

from floorplan_di.training.train import calculate_class_weights


def test_class_weights_are_accumulated_from_manifest_stream(tmp_path: Path) -> None:
    manifest = tmp_path / "train.jsonl"
    records = [
        {"pixel_counts": [10, 20, 30, 40, 50]},
        {"pixel_counts": [5, 10, 15, 20, 25]},
    ]
    manifest.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    weights = calculate_class_weights(manifest, cap=10.0)

    assert weights.dtype == torch.float32
    assert weights.mean().item() == pytest.approx(1.0)
    assert weights.tolist() == pytest.approx([2.189781, 1.094891, 0.729927, 0.547445, 0.437956])


def test_class_weights_reject_empty_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "train.jsonl"
    manifest.write_text("\n", encoding="utf-8")

    with pytest.raises(ValueError, match="No labelled pixels"):
        calculate_class_weights(manifest, cap=10.0)
