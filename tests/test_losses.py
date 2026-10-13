import torch
import torch.nn.functional as functional

from floorplan_di.models.losses import multiclass_dice_loss


def test_dice_matches_one_hot_loss_and_gradients() -> None:
    logits = torch.linspace(-2, 2, 120).reshape(2, 5, 3, 4).requires_grad_()
    target = torch.arange(24).reshape(2, 3, 4) % 4  # Include an absent class.
    probabilities = logits.softmax(dim=1)
    one_hot = functional.one_hot(target, 5).movedim(-1, 1).float()
    intersection = (probabilities * one_hot).sum(dim=(0, 2, 3))
    cardinality = (probabilities + one_hot).sum(dim=(0, 2, 3))
    expected = 1 - ((2 * intersection + 1e-6) / (cardinality + 1e-6)).mean()
    actual = multiclass_dice_loss(logits, target)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, logits)[0], torch.autograd.grad(expected, logits)[0]
    )


def test_dice_half_precision_large_masks_remain_finite() -> None:
    logits = torch.zeros(1, 5, 256, 256, dtype=torch.float16, requires_grad=True)
    target = torch.zeros(1, 256, 256, dtype=torch.long)
    loss = multiclass_dice_loss(logits, target)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
