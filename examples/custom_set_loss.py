"""Extend a configured criterion with an all-query direction regularizer."""

from collections.abc import Mapping

import torch
from torch import Tensor
from typing_extensions import override

import obb_loss as obb


class DirectionCriterion(obb.SetCriterion):
    """Add a unit-direction penalty, including unmatched queries."""

    @override
    def compute_losses(
        self,
        preds: obb.Predictions,
        targets: obb.Targets,
        assignment: obb.Assignment,
    ) -> obb.Result[Mapping[str, Tensor], obb.InputError]:
        """Retain configured box losses and add weighted per-sample direction sums.

        Args:
            preds: All queries; the regularizer also supervises unmatched boxes.
            targets: Packed reference geometry and class labels.
            assignment: Matches selected for the built-in box terms.
        """
        match super().compute_losses(preds, targets, assignment):
            case obb.Err() as failure:
                return failure
            case obb.Ok(value=terms):
                residual = preds.boxes.values[:, 4:6].double().square().sum(-1) - 1
                direction = obb.sum_samples(residual.square(), preds.boxes.cu_seqlens)
                return obb.Ok({**terms, "direction": 0.5 * direction})


def main() -> None:
    """Backpropagate a subclass-defined term through packed predictions."""
    boxes = torch.tensor([[0.1, 0, 2, 1, 1, 0], [4, 0, 2, 1, 2, 0]], requires_grad=True)
    offsets = torch.tensor([0, 2])
    preds = obb.Predictions(
        obb.PackedBatch(boxes, offsets), torch.zeros(2, 1, requires_grad=True)
    )
    targets = obb.Targets(
        obb.PackedBatch(torch.tensor([[0.0, 0, 2, 1, 1, 0]]), torch.tensor([0, 1])),
        torch.tensor([0]),
    )
    criterion = DirectionCriterion(obb.SetCriterionConfig(classification="none"))
    match criterion(preds, targets):
        case obb.Err(error=error):
            print(error)
        case obb.Ok(value=losses):
            losses.total.backward()
            assert boxes.grad is not None
            assert boxes.grad[1, 4].item() != 0
            print({name: value.item() for name, value in losses.terms.items()})


if __name__ == "__main__":
    main()
