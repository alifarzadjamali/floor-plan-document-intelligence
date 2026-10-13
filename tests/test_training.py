import json
from pathlib import Path

import pytest
import torch

from floorplan_di.training.train import _MeanLoss, calculate_class_weights


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


def test_epoch_loss_weights_short_final_batch_without_retaining_graph() -> None:
    losses = _MeanLoss(torch.device("cpu"))
    losses.update(torch.tensor(2.0, requires_grad=True), 4)
    losses.update(torch.tensor(7.0, requires_grad=True), 1)

    assert losses.compute() == pytest.approx(3.0)
    assert not losses.total.requires_grad


def test_epoch_loss_rejects_empty_loader() -> None:
    with pytest.raises(ValueError, match="empty loader"):
        _MeanLoss(torch.device("cpu")).compute()


def test_class_weights_handle_most_classes_absent(tmp_path: Path) -> None:
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(json.dumps({"pixel_counts": [100, 10, 0, 0, 0]}), encoding="utf-8")
    weights = calculate_class_weights(manifest, cap=8.0)

    assert torch.isfinite(weights).all()
    assert (weights > 0).all()
    assert weights.mean().item() == pytest.approx(1.0)
    assert weights[0] < weights[1]


@pytest.mark.parametrize("cap", [0, -1, float("nan"), float("inf")])
def test_class_weights_reject_invalid_cap(tmp_path: Path, cap: float) -> None:
    with pytest.raises(ValueError, match="cap must be finite and positive"):
        calculate_class_weights(tmp_path / "unused.jsonl", cap=cap)


@pytest.mark.parametrize("count", [float("nan"), float("inf"), -1])
def test_class_weights_reject_invalid_counts(tmp_path: Path, count: float) -> None:
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(json.dumps({"pixel_counts": [count, 1, 2, 3, 4]}), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid pixel_counts"):
        calculate_class_weights(manifest, cap=8.0)
