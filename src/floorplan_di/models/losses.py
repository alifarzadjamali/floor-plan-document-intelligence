from __future__ import annotations

import torch
import torch.nn.functional as functional


def multiclass_dice_loss(
    logits: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6
) -> torch.Tensor:
    probabilities = logits.softmax(dim=1)
    target_one_hot = functional.one_hot(target, num_classes=logits.shape[1]).movedim(-1, 1).float()
    intersection = (probabilities * target_one_hot).sum(dim=(0, 2, 3))
    cardinality = (probabilities + target_one_hot).sum(dim=(0, 2, 3))
    return 1 - ((2 * intersection + epsilon) / (cardinality + epsilon)).mean()
