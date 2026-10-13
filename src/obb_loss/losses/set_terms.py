"""Pure set-prediction costs, loss projections and normalization policies.

Matching combines class probability and geometry as in Carion et al.,
https://arxiv.org/abs/2005.12872; sigmoid probabilities here are an explicit
adaptation for quality-classification losses rather than DETR's softmax cost.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from einops import rearrange
from torch import Tensor

from obb_loss.criterion_core import validate_tensor
from obb_loss.losses.stable_probiou import stable_probiou_loss
from obb_loss.result import Err, Ok, Result

if TYPE_CHECKING:
    from shape_extensions import IntVar

    from obb_loss.criterion_types import ImagePredictions, LossContext, Targets
    from obb_loss.result import InputError

    N = IntVar("N")


def class_probability_cost(
    predictions: ImagePredictions, targets: Targets
) -> Result[Tensor, InputError]:
    """Return negative sigmoid class probability [queries, objects].

    Inputs have validated metadata; labels must be in range. Cost gradients are
    preserved here; the criterion disables them for assignment.
    """
    with torch.autocast(predictions.logits.device.type, enabled=False):
        return Ok(-predictions.logits.double().sigmoid()[:, targets.labels])


def pairwise_box_l1_cost(
    predictions: ImagePredictions, targets: Targets
) -> Result[Tensor, InputError]:
    """Return [queries, objects] center/side L1 in supplied coordinate units."""
    with torch.autocast(predictions.boxes.device.type, enabled=False):
        predicted = rearrange(predictions.boxes[:, :4].double(), "q c -> q 1 c")
        target = rearrange(targets.boxes[:, :4].double(), "n c -> 1 n c")
        return Ok((predicted - target).abs().sum(dim=-1))


def pairwise_probiou_cost(
    predictions: ImagePredictions, targets: Targets
) -> Result[Tensor, InputError]:
    """Return pairwise ProbIoU loss with the existing primitive's numerical domain."""
    predicted = rearrange(predictions.boxes, "q c -> q 1 c")
    target = rearrange(targets.boxes, "n c -> 1 n c")
    return stable_probiou_loss(predicted, target)


def matched_box_inputs(
    context: LossContext,
) -> Result[tuple[Tensor[[N, 6]], Tensor[[N, 6]]], InputError]:
    """Borrow aligned pairs prepared by the core; preserve both gradient paths."""
    return Ok((context.predicted_boxes, context.target_boxes))


def one_hot_classification_inputs(
    context: LossContext,
) -> Result[tuple[Tensor, Tensor], InputError]:
    """Return flattened logits and one-hot quality targets for every query.

    Matched class entries are 1 and all others 0, including unmatched queries.
    Integer labels must be in range. No prediction-derived quality is inferred.
    """
    logits = context.predictions.logits
    quality = torch.zeros_like(logits, dtype=torch.float64)
    for image, (target, assignment) in enumerate(
        zip(context.targets, context.matches, strict=True)
    ):
        quality[image, assignment.predictions, target.labels[assignment.targets]] = 1
    return Ok(
        (rearrange(logits, "b q c -> (b q) c"), rearrange(quality, "b q c -> (b q) c"))
    )


def supplied_quality_inputs(
    context: LossContext, *, quality: Tensor
) -> Result[tuple[Tensor, Tensor], InputError]:
    """Project caller-supplied [B,Q,C] quality, preserving its gradients.

    Bind quality with functools.partial. Values must be in [0,1]; metadata only
    is checked. No detach is applied, so prediction-derived quality policy is
    explicitly owned by the caller.
    """
    logits = context.predictions.logits
    match validate_tensor(quality, logits, "quality", tuple(logits.shape)):
        case Err() as failure:
            return failure
        case Ok():
            return Ok(
                (
                    rearrange(logits, "b q c -> (b q) c"),
                    rearrange(quality, "b q c -> (b q) c"),
                )
            )


def target_count(context: LossContext) -> Tensor:
    """Divide by max(total target count, 1), using metadata without synchronization."""
    count = max(sum(target.boxes.shape[0] for target in context.targets), 1)
    return context.predictions.boxes.new_tensor(count, dtype=torch.float64)


def query_count(context: LossContext) -> Tensor:
    """Divide by max(total query count, 1), including unmatched queries."""
    boxes = context.predictions.boxes
    return boxes.new_tensor(
        max(boxes.shape[0] * boxes.shape[1], 1), dtype=torch.float64
    )


def identity(context: LossContext) -> Tensor:
    """Leave a summed or already-normalized loss unscaled."""
    return context.predictions.boxes.new_tensor(1, dtype=torch.float64)
