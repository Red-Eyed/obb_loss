"""Add a caller-defined direction constraint to the standard set loss callbacks."""

import torch
from torch import Tensor

import obb_loss as obb


def direction_penalty(
    context: obb.SetLossContext,
) -> obb.Result[Tensor, obb.InputError]:
    """Penalize nonunit directions on every query, including unmatched queries.

    Args:
        context: Validated packed sets; predictions store (cos(theta), sin(theta))
            in the last two coordinates. No target or assignment is needed here.

    Returns:
        Ok(float64 raw per-sample sums of (cos² + sin² - 1)²), without weighting,
        target-count normalization, or batch reduction.
    """
    directions = context.preds.geometry.boxes[:, 4:6].double()
    values = (directions.square().sum(-1) - 1).square()
    return obb.Ok(obb.sum_samples(values, context.preds.geometry.cu_seqlens))


def main() -> None:
    """Compose built-ins and a custom callback, then inspect its gradient."""
    boxes = torch.tensor(
        [[[0.1, 0, 2, 1, 1.2, 0], [3, 0, 2, 1, 0, 0.5]]], requires_grad=True
    )
    logits = torch.zeros(1, 2, 2, requires_grad=True)
    match obb.pack_predictions(boxes, logits):
        case obb.Err(error=error):
            print(error)
            return
        case obb.Ok(value=preds):
            pass
    match obb.pack_targets(
        [
            obb.TargetSample(
                torch.tensor([[0, 0, 2, 1, 1, 0]], dtype=boxes.dtype),
                torch.tensor([0]),
                torch.tensor([0.8]),
            )
        ],
        device=boxes.device,
        dtype=boxes.dtype,
    ):
        case obb.Err(error=error):
            print(error)
            return
        case obb.Ok(value=targets):
            pass
    criterion = obb.SetCriterion(
        losses={
            **obb.default_set_losses(),
            "direction": obb.WeightedLoss(
                direction_penalty, obb.WeightConfig(weight=0.5)
            ),
        },
    )
    match criterion(preds, targets, reduction=obb.Reduction.NONE, normalizer=1.0):
        case obb.Err(error=error):
            print(error)
            return
        case obb.Ok(value=losses):
            losses.total.sum().backward()
            assert boxes.grad is not None
            assert boxes.grad[0, 1, 5].item() != 0
            print(
                {name: value.detach().tolist() for name, value in losses.terms.items()}
            )
            print("The custom constraint also supervises the unmatched query.")


if __name__ == "__main__":
    main()
