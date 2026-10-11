"""Combine ProbIoU with a custom function for already aligned packed boxes."""

import torch
from torch import Tensor

from obb_loss import (
    BoxLossBuilder,
    Err,
    LossConfig,
    Ok,
    PackedBoxes,
    Reduction,
    stable_probiou_loss,
)


def center_loss(preds: Tensor, targets: Tensor) -> Tensor:
    """Return one squared center distance per aligned pair."""
    return (preds[:, :2] - targets[:, :2]).square().sum(dim=-1)


def main() -> None:
    """Build and evaluate a weighted composite with one final mean reduction."""
    predictions = torch.tensor([[0.2, 0.1, 2.0, 1.0, 1.0, 0.0]], requires_grad=True)
    targets = torch.tensor([[0.0, 0.0, 2.0, 1.0, 1.0, 0.0]])
    offsets = torch.tensor([0, 1], dtype=torch.int64)
    builder = (
        BoxLossBuilder()
        .add("probiou", stable_probiou_loss)
        .add("center", center_loss, weight=0.1)
    )
    match builder.build(LossConfig(reduction=Reduction.MEAN)):
        case Err(error=error):
            print(error)
            return
        case Ok(value=criterion):
            pass
    match criterion.checked(
        PackedBoxes(predictions, offsets), PackedBoxes(targets, offsets)
    ):
        case Err(error=error):
            print(error)
            return
        case Ok(value=loss):
            loss.backward()
    print(f"composite loss: {loss.detach().item():.6f}")
    print(f"prediction gradients: {predictions.grad}")


if __name__ == "__main__":
    main()
