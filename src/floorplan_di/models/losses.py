from __future__ import annotations

import torch


def multiclass_dice_loss(
    logits: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6
) -> torch.Tensor:
    # Accumulate in float32 even under autocast; avoid an N x C x H x W target.
    probabilities = logits.softmax(dim=1, dtype=torch.float32)
    classes = logits.shape[1]
    flat_target = target.reshape(-1)
    matched = probabilities.gather(1, target.unsqueeze(1)).reshape(-1)
    intersection = probabilities.new_zeros(classes).scatter_add(0, flat_target, matched)
    target_counts = torch.bincount(flat_target, minlength=classes).to(probabilities.dtype)
    cardinality = probabilities.sum(dim=(0, 2, 3)) + target_counts
    return 1 - ((2 * intersection + epsilon) / (cardinality + epsilon)).mean()
