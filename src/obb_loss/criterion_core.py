"""Pure metadata validation, matching costs and differentiable loss composition."""

from __future__ import annotations

import math
from itertools import chain
from typing import TYPE_CHECKING

import torch
from einops import rearrange
from torch import Tensor

from obb_loss.criterion_types import (
    Cost,
    Criterion,
    CriterionError,
    CriterionErrorKind,
    CriterionFailure,
    ImagePredictions,
    LossContext,
    LossOutput,
    LossTerm,
    Match,
    Predictions,
    Reduction,
    Targets,
)
from obb_loss.result import Err, ErrorKind, InputError, Ok, Result

if TYPE_CHECKING:
    from collections.abc import Sequence


def validate_declaration(criterion: Criterion) -> Result[None, CriterionFailure]:
    """Reject duplicate names, invalid weights and absent active matching costs."""
    match validate_reduction(criterion.reduction):
        case Err() as failure:
            return failure
        case Ok():
            pass
    for group in (criterion.matcher.costs, criterion.losses):
        names: set[str] = set()
        for term in group:
            if not term.name or term.name in names:
                return Err(
                    CriterionError(
                        CriterionErrorKind.SPECIFICATION,
                        "name",
                        "nonempty unique names required",
                    )
                )
            names.add(term.name)
            if not math.isfinite(term.weight) or term.weight < 0:
                return Err(
                    CriterionError(
                        CriterionErrorKind.SPECIFICATION,
                        term.name,
                        "finite nonnegative weight required",
                    )
                )
    if any(term.weight > 0 for term in criterion.losses) and not any(
        cost.weight > 0 for cost in criterion.matcher.costs
    ):
        return Err(
            CriterionError(
                CriterionErrorKind.SPECIFICATION,
                "costs",
                "an enabled matching cost is required",
            )
        )
    return Ok(None)


def validate_reduction(reduction: Reduction) -> Result[None, CriterionFailure]:
    """Reject unsupported reduction declarations before tensor work."""
    if reduction not in ("none", "sum", "mean"):
        return Err(
            CriterionError(
                CriterionErrorKind.SPECIFICATION,
                "reduction",
                "expected none, sum or mean",
            )
        )
    return Ok(None)


def reduced_zero(predictions: Predictions, reduction: Reduction) -> Tensor:
    """Retain zero gradients and return an empty vector for an empty image batch."""
    if reduction == "none":
        return predictions.boxes[..., :0].double().sum(dim=(1, 2)) + predictions.logits[
            ..., :0
        ].double().sum(dim=(1, 2))
    return connected_zero(predictions)


def reduce_values(
    loss: LossTerm, values: Tensor, context: LossContext, reduction: Reduction
) -> Result[Tensor, CriterionFailure]:
    """Group loss rows by explicit image ownership, without scanning index values."""
    batch = context.predictions.boxes.shape[0]
    if reduction != "none":
        total = values.double().sum()
        return Ok(total / max(batch, 1) if reduction == "mean" else total)
    match loss.samples(context):
        case Err() as failure:
            return failure
        case Ok(indices):
            pass
    field = f"{loss.name}.sample_indices"
    if values.ndim == 0 or indices.ndim != 1 or indices.shape[0] != values.shape[0]:
        return Err(
            InputError(
                ErrorKind.SHAPE, field, "int64 [rows] required for non-scalar values"
            )
        )
    if indices.dtype != torch.int64:
        return Err(InputError(ErrorKind.DTYPE, field, "int64 required"))
    if indices.device != context.predictions.boxes.device:
        return Err(InputError(ErrorKind.DEVICE, field, "prediction device required"))
    rows = (
        values.double().sum(dim=tuple(range(1, values.ndim)))
        if values.ndim > 1
        else values.double()
    )
    return Ok(values.new_zeros(batch, dtype=torch.float64).index_add(0, indices, rows))


def validate_inputs(
    predictions: Predictions, targets: Sequence[Targets]
) -> Result[None, InputError]:
    """Validate batch geometry, classes and per-image metadata without value scans."""
    boxes, logits = predictions.boxes, predictions.logits
    if (
        boxes.ndim != 3
        or boxes.shape[-1] != 6
        or logits.ndim != 3
        or logits.shape[:2] != boxes.shape[:2]
        or logits.shape[-1] == 0
        or len(targets) != boxes.shape[0]
    ):
        return Err(
            InputError(
                ErrorKind.SHAPE,
                "predictions",
                "expected boxes [B,Q,6], logits [B,Q,C>0], B targets",
            )
        )
    floating: list[Tensor] = [boxes, logits]
    for target in targets:
        if (
            target.boxes.ndim != 2
            or target.boxes.shape[-1] != 6
            or target.labels.ndim != 1
            or target.labels.shape[0] != target.boxes.shape[0]
        ):
            return Err(
                InputError(
                    ErrorKind.SHAPE, "targets", "expected boxes [N,6] and labels [N]"
                )
            )
        if target.labels.dtype != torch.int64:
            return Err(InputError(ErrorKind.DTYPE, "labels", "int64 required"))
        floating.append(target.boxes)
    if any(not tensor.is_floating_point() for tensor in floating):
        return Err(InputError(ErrorKind.DTYPE, "inputs", "floating tensors required"))
    tensors = chain(floating, (target.labels for target in targets))
    if boxes.device.type not in ("cpu", "cuda") or any(
        tensor.device != boxes.device for tensor in tensors
    ):
        return Err(
            InputError(ErrorKind.DEVICE, "inputs", "same CPU/CUDA device required")
        )
    return Ok(None)


def validate_tensor(
    tensor: Tensor, reference: Tensor, field: str, shape: tuple[int, ...] | None = None
) -> Result[None, InputError]:
    """Check extension output metadata; numerical domains remain caller obligations."""
    if shape is not None and tensor.shape != shape:
        return Err(InputError(ErrorKind.SHAPE, field, f"expected shape {shape}"))
    if not tensor.is_floating_point():
        return Err(InputError(ErrorKind.DTYPE, field, "floating tensor required"))
    if tensor.device != reference.device:
        return Err(InputError(ErrorKind.DEVICE, field, "prediction device required"))
    return Ok(None)


def build_cost_matrix(
    costs: Sequence[Cost], predictions: ImagePredictions, targets: Targets
) -> Result[Tensor, InputError]:
    """Sum declared pairwise costs; caller disables gradients for assignment.

    Predictions and targets must have validated metadata. Cost values must be
    finite. Zero-weight cost functions are never invoked.
    """
    shape = (predictions.boxes.shape[0], targets.boxes.shape[0])
    total = predictions.boxes.new_zeros(shape, dtype=torch.float64)
    with torch.autocast(predictions.boxes.device.type, enabled=False):
        for cost in costs:
            if cost.weight == 0:
                continue
            match cost.compute(predictions, targets):
                case Err() as failure:
                    return failure
                case Ok(values):
                    pass
            match validate_tensor(values, predictions.boxes, cost.name, shape):
                case Err() as failure:
                    return failure
                case Ok():
                    total = total + cost.weight * values.double()
    return Ok(total)


def empty_match(reference: Tensor) -> Match:
    """Create empty int64 assignments on the prediction device."""
    indices = torch.empty(0, dtype=torch.int64, device=reference.device)
    return Match(indices, indices)


def connected_zero(predictions: Predictions) -> Tensor:
    """Return zero gradients even when unused prediction values are nonfinite."""
    return (
        predictions.boxes[..., :0].double().sum()
        + predictions.logits[..., :0].double().sum()
    )


def _empty_box_pairs(boxes: Tensor) -> Tensor[[int, 6]]:
    """Return an empty aligned view of validated batch boxes, retaining autograd."""
    return rearrange(boxes, "b q c -> (b q) c")[:0]


def prepare_context(
    predictions: Predictions, targets: Sequence[Targets], matches: Sequence[Match]
) -> Result[LossContext, InputError]:
    """Gather aligned pairs after metadata validation, preserving target gradients.

    Match indices must be in range and unique within each image; only metadata
    is checked. Public core use does not require an assignment backend.
    """
    match validate_inputs(predictions, targets):
        case Err() as failure:
            return failure
        case Ok():
            pass
    if len(matches) != len(targets):
        return Err(
            InputError(ErrorKind.SHAPE, "matches", "one match per image required")
        )
    predicted: list[Tensor] = []
    expected: list[Tensor] = []
    for image, (target, assignment) in enumerate(zip(targets, matches, strict=True)):
        indices, target_indices = assignment.predictions, assignment.targets
        if indices.ndim != 1 or indices.shape != target_indices.shape:
            return Err(
                InputError(ErrorKind.SHAPE, "matches", "aligned [pairs] required")
            )
        if indices.dtype != torch.int64 or target_indices.dtype != torch.int64:
            return Err(InputError(ErrorKind.DTYPE, "matches", "int64 required"))
        if (
            indices.device != predictions.boxes.device
            or target_indices.device != predictions.boxes.device
        ):
            return Err(
                InputError(ErrorKind.DEVICE, "matches", "prediction device required")
            )
        predicted.append(predictions.boxes[image, indices])
        expected.append(target.boxes[target_indices])
    predicted_boxes = (
        torch.cat(predicted) if predicted else _empty_box_pairs(predictions.boxes)
    )
    target_boxes = torch.cat(expected) if expected else predicted_boxes
    return Ok(
        LossContext(
            predictions, tuple(targets), tuple(matches), predicted_boxes, target_boxes
        )
    )


def evaluate_losses(
    losses: Sequence[LossTerm], context: LossContext, reduction: Reduction = "sum"
) -> Result[LossOutput, CriterionFailure]:
    """Reduce declared losses without assignment, mutation or implicit detachment.

    Custom divisors must be finite positive scalars. Only divisor metadata is
    checked, allowing supplied device scalars without CPU synchronization.
    "none" groups first-axis rows by declared sample indices and sums trailing
    axes. "mean" divides the batch sum by max(batch size, 1). Every reduction
    retains the declared batch divisor; an empty batch returns connected zeros.
    """
    match validate_reduction(reduction):
        case Err() as failure:
            return Err(failure.error)
        case Ok():
            pass
    raw: dict[str, Tensor] = {}
    weighted: dict[str, Tensor] = {}
    total = reduced_zero(context.predictions, reduction)
    reference = context.predictions.boxes
    with torch.autocast(reference.device.type, enabled=False):
        for loss in losses:
            if loss.weight == 0:
                continue
            match loss.values(context):
                case Err() as failure:
                    return Err(failure.error)
                case Ok(values):
                    pass
            match validate_tensor(values, reference, loss.name):
                case Err() as failure:
                    return Err(failure.error)
                case Ok():
                    pass
            divisor = loss.divisor(context)
            match validate_tensor(divisor, reference, f"{loss.name}.divisor", ()):
                case Err() as failure:
                    return Err(failure.error)
                case Ok():
                    pass
            match reduce_values(loss, values, context, reduction):
                case Err() as failure:
                    return Err(failure.error)
                case Ok(numerator):
                    pass
            raw[loss.name] = numerator / divisor.double()
            weighted[loss.name] = loss.weight * raw[loss.name]
            total = total + weighted[loss.name]
    return Ok(LossOutput(total, raw, weighted, context.matches))
