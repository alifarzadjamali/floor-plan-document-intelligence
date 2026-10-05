from pathlib import Path

import pytest
import torch

from floorplan_di.models.checkpoints import load_checkpoint_data


def test_checkpoint_loader_uses_restricted_torch_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_load(path: Path, **kwargs: object) -> dict[str, object]:
        calls.append((path, kwargs))
        return {"model_state_dict": {}}

    monkeypatch.setattr(torch, "load", fake_load)
    path = Path("model.pt")

    assert load_checkpoint_data(path) == {"model_state_dict": {}}
    assert calls == [(path, {"map_location": "cpu", "weights_only": True})]


def test_checkpoint_loader_rejects_missing_model_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {"epoch": 1})

    with pytest.raises(ValueError, match="Invalid model checkpoint"):
        load_checkpoint_data(Path("model.pt"))
