from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_checkpoint_data(
    path: Path, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    """Load a project checkpoint without allowing arbitrary pickled objects."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Invalid model checkpoint: {path}")
    return checkpoint
