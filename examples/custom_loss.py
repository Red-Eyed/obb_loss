"""Combine ProbIoU with a custom function for already aligned packed boxes."""

import torch
from torch import Tensor

import obb_loss as obb


def center_loss(preds: Tensor, targets: Tensor) -> Tensor:
    """Return one squared center distance per aligned pair."""
    return (preds[:, :2] - targets[:, :2]).square().sum(dim=-1)


def main() -> None:
    """Build and evaluate a weighted composite with one final mean reduction."""
    predictions = torch.tensor([[0.2, 0.1, 2.0, 1.0, 1.0, 0.0]], requires_grad=True)
    targets = torch.tensor([[0.0, 0.0, 2.0, 1.0, 1.0, 0.0]])
    offsets = torch.tensor([0, 1], dtype=torch.int64)
    builder = (
        obb.BoxLossBuilder()
        .add("probiou", obb.stable_probiou_loss)
        .add("center", center_loss, weight=0.1)
    )
    match builder.build(obb.LossConfig(reduction=obb.Reduction.MEAN)):
        case obb.Err(error=error):
            print(error)
            return
        case obb.Ok(value=criterion):
            pass
    match criterion.checked(
        obb.PackedBoxes(predictions, offsets),
        obb.PackedBoxes(targets, offsets),
    ):
        case obb.Err(error=error):
            print(error)
            return
        case obb.Ok(value=loss):
            loss.backward()
    print(f"composite loss: {loss.detach().item():.6f}")
    print(f"prediction gradients: {predictions.grad}")


if __name__ == "__main__":
    main()
