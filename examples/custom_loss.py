"""Combine ProbIoU with a custom function for already aligned packed boxes."""

import torch
from torch import Tensor

import obb_loss as obb


def center_loss(context: obb.BoxLossContext) -> obb.Result[Tensor, obb.InputError]:
    """Return one squared center distance per aligned pair."""
    return obb.Ok(
        (context.preds.boxes[:, :2].double() - context.targets.boxes[:, :2].double())
        .square()
        .sum(dim=-1)
    )


def main() -> None:
    """Build and evaluate a weighted composite with one final mean reduction."""
    predictions = torch.tensor([[0.2, 0.1, 2.0, 1.0, 1.0, 0.0]], requires_grad=True)
    targets = torch.tensor([[0.0, 0.0, 2.0, 1.0, 1.0, 0.0]])
    offsets = torch.tensor([0, 1], dtype=torch.int64)
    criterion = obb.CompositeBoxLoss(
        losses={
            "probiou": obb.box_probiou_loss,
            "center": obb.WeightedLoss(center_loss, obb.WeightConfig(weight=0.1)),
        }
    )
    match criterion(
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
