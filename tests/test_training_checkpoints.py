from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch.optim import AdamW

from floorplan_di.models.checkpoints import load_checkpoint_data
from floorplan_di.training.train import _save_checkpoint, _scheduler


def test_checkpoint_round_trips_with_restricted_loader(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = AdamW(model.parameters())
    scheduler = _scheduler(optimizer, 1, 10)
    path = tmp_path / "last.pt"
    _save_checkpoint(path, model, optimizer, 3, {}, torch.ones(5), 0.7, scheduler)

    saved = load_checkpoint_data(path)
    assert saved["best_score"] == 0.7
    assert saved["epoch"] == 3
    assert saved["scheduler_state_dict"] == scheduler.state_dict()
    assert type(saved["torch_version"]) is str
    assert not list(tmp_path.glob("*.tmp"))


def test_failed_checkpoint_write_preserves_previous_file(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = AdamW(model.parameters())
    path = tmp_path / "last.pt"
    path.write_bytes(b"previous checkpoint")

    with patch("floorplan_di.training.train.torch.save", side_effect=OSError("disk full")):
        with pytest.raises(OSError, match="disk full"):
            _save_checkpoint(path, model, optimizer, 3, {}, torch.ones(5), 0.7)

    assert path.read_bytes() == b"previous checkpoint"
    assert not list(tmp_path.glob("*.tmp"))
