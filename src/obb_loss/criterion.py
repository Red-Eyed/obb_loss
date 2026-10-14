"""Thin assignment shell around a pure set-prediction loss core."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from obb_loss.criterion_core import (
    build_cost_matrix,
    empty_match,
    evaluate_losses,
    prepare_context,
    reduced_zero,
    validate_declaration,
    validate_inputs,
)
from obb_loss.criterion_types import (
    Criterion,
    CriterionFailure,
    ImagePredictions,
    LossOutput,
    Match,
    Predictions,
    Targets,
)
from obb_loss.result import Err, Ok, Result

if TYPE_CHECKING:
    from collections.abc import Sequence


def criterion_loss(
    criterion: Criterion, predictions: Predictions, targets: Sequence[Targets]
) -> Result[LossOutput, CriterionFailure]:
    """Match each image and evaluate its declaratively specified batch objective.

    Boxes use (cx,cy,width,height,cos,sin). Labels and assignments must be valid;
    tensor values are not scanned. Matching is nondifferentiable; training losses
    preserve original prediction and target gradients. The selected solver may
    synchronize CUDA and transfer costs to CPU. Empty assignment problems bypass
    it. All-disabled losses skip matching and return a connected float64 zero.
    Reduction controls scalar or [batch] outputs while retaining batch divisors.
    No inputs are mutated. Expected metadata, declaration and backend failures
    return Err; unexpected bugs in user functions propagate.
    """
    for outcome in (
        validate_declaration(criterion),
        validate_inputs(predictions, targets),
    ):
        match outcome:
            case Err() as failure:
                return Err(failure.error)
            case Ok():
                pass
    if not any(loss.weight > 0 for loss in criterion.losses):
        matches = tuple(empty_match(predictions.boxes) for _ in targets)
        return Ok(
            LossOutput(reduced_zero(predictions, criterion.reduction), {}, {}, matches)
        )
    assignments: list[Match] = []
    with torch.no_grad():
        for image, target in enumerate(targets):
            if target.boxes.shape[0] == 0 or predictions.boxes.shape[1] == 0:
                assignments.append(empty_match(predictions.boxes))
                continue
            image_predictions = ImagePredictions(
                predictions.boxes[image], predictions.logits[image]
            )
            match build_cost_matrix(criterion.matcher.costs, image_predictions, target):
                case Err() as failure:
                    return Err(failure.error)
                case Ok(costs):
                    pass
            match criterion.matcher.solver(costs):
                case Err() as failure:
                    return Err(failure.error)
                case Ok(assignment):
                    assignments.append(assignment)
    match prepare_context(predictions, targets, assignments):
        case Err() as failure:
            return Err(failure.error)
        case Ok(context):
            match evaluate_losses(criterion.losses, context, criterion.reduction):
                case Err(error):
                    return Err(error)
                case Ok(output):
                    return Ok(output)
