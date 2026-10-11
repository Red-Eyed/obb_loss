"""Packed predictions and labeled targets for a single set-prediction layer."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch
from einops import rearrange
from torch import Tensor

from obb_loss.adapters import pack_boxes, pack_dense_boxes
from obb_loss.packed import ErrorKind, InputError, PackedBatch, make_packed_boxes
from obb_loss.result import Err, Ok, Result

if TYPE_CHECKING:
    from collections.abc import Sequence


class Predictions(NamedTuple):
    """Borrow one layer of predictions with shared sample boundaries.

    Attributes:
        boxes: Packed [queries, 6] oriented-box geometry.
        logits: Floating [queries, classes] pre-sigmoid scores in box order.
    """

    boxes: PackedBatch
    logits: Tensor[[int, int]]


class Targets(NamedTuple):
    """Borrow reference annotations with shared sample boundaries.

    Attributes:
        boxes: Packed [targets, 6] oriented-box geometry.
        labels: Int64 [targets] zero-based class IDs in box order.
    """

    boxes: PackedBatch
    labels: Tensor[[int]]


class TargetSample(NamedTuple):
    """Represent annotations for one sample before packing.

    Attributes:
        boxes: Floating [count, 6] reference boxes in (cx, cy, w, h, cos(theta),
            sin(theta)) order.
        labels: Same-device int64 [count] zero-based class IDs, aligned with boxes.
    """

    boxes: Tensor[[int, 6]]
    labels: Tensor[[int]]


def make_packed_predictions(
    geometry: PackedBatch, logits: Tensor
) -> Result[Predictions, InputError]:
    """Validate and borrow one layer of packed box predictions and logits.

    Args:
        geometry: Valid packed query boxes; offsets define sample ownership.
        logits: Finite floating [queries, classes] scores before sigmoid, in
            geometry order and on its device; at least one class and no background
            column.

    Returns:
        Ok(Predictions) borrowing both fields, or Err(InputError) for
        geometry, layout, shape, dtype, device, or nonfinite logits.
    """
    match make_packed_boxes(*geometry):
        case Err() as failure:
            return failure
        case Ok():
            pass
    if (
        logits.ndim != 2
        or logits.shape[0] != geometry.values.shape[0]
        or logits.shape[1] == 0
    ):
        return Err(
            InputError(ErrorKind.SHAPE, "logits", "expected [queries, classes>0]")
        )
    if logits.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        return Err(
            InputError(ErrorKind.DTYPE, "logits", "supported floating dtype required")
        )
    if logits.device != geometry.values.device:
        return Err(InputError(ErrorKind.DEVICE, "logits", "must share geometry device"))
    if not bool(torch.isfinite(logits).all()):
        return Err(InputError(ErrorKind.DOMAIN, "logits", "must be finite"))
    return Ok(Predictions(geometry, logits))


def _validate_target_fields(
    boxes: Tensor, labels: Tensor
) -> Result[Tensor, InputError]:
    """Require aligned class labels without silently transferring tensors."""
    if labels.shape != (boxes.shape[0],):
        return Err(InputError(ErrorKind.SHAPE, "labels", "expected [count]"))
    if labels.dtype != torch.int64:
        return Err(InputError(ErrorKind.DTYPE, "labels", "int64 labels required"))
    if labels.device != boxes.device:
        return Err(InputError(ErrorKind.DEVICE, "labels", "must share geometry device"))
    if not bool((labels >= 0).all()):
        return Err(
            InputError(ErrorKind.DOMAIN, "labels", "class labels must be nonnegative")
        )
    return Ok(labels)


def make_packed_targets(
    geometry: PackedBatch, labels: Tensor
) -> Result[Targets, InputError]:
    """Validate and borrow packed reference geometry and labels.

    Args:
        geometry: Valid packed reference boxes; offsets define sample ownership.
        labels: Same-device int64 [targets] zero-based class IDs, ordered with
            geometry; upper bounds are checked against prediction logits during
            matching.

    Returns:
        Ok(Targets) borrowing every field, or Err(InputError) for invalid
        geometry, layout, label or device contracts.
    """
    match make_packed_boxes(*geometry):
        case Err() as failure:
            return failure
        case Ok():
            pass
    match _validate_target_fields(geometry.values, labels):
        case Err() as failure:
            return failure
        case Ok():
            return Ok(Targets(geometry, labels))


def validate_sets(
    preds: Predictions, targets: Targets
) -> Result[tuple[Predictions, Targets], InputError]:
    """Validate current packed set records against one another.

    Args:
        preds: Packed geometry and [queries, classes] logits for one decoder layer,
            sharing box order and device.
        targets: Packed reference geometry and int64 [targets] labels following box
            order; classification quality is supplied separately after matching.

    Returns:
        Ok(the borrowed prediction/target pair) or Err(InputError) for invalid
        fields, sample counts, devices, or out-of-range labels.
    """
    match make_packed_predictions(preds.boxes, preds.logits):
        case Err() as failure:
            return failure
        case Ok():
            pass
    match make_packed_targets(targets.boxes, targets.labels):
        case Err() as failure:
            return failure
        case Ok():
            pass
    if preds.boxes.cu_seqlens.shape != targets.boxes.cu_seqlens.shape:
        return Err(InputError(ErrorKind.ALIGNMENT, "batch", "sample counts must match"))
    if preds.logits.device != targets.labels.device:
        return Err(
            InputError(
                ErrorKind.DEVICE, "targets", "prediction/target device must match"
            )
        )
    if not bool((targets.labels < preds.logits.shape[1]).all()):
        return Err(
            InputError(
                ErrorKind.DOMAIN, "labels", "label exceeds prediction class count"
            )
        )
    return Ok((preds, targets))


def pack_predictions(boxes: Tensor, logits: Tensor) -> Result[Predictions, InputError]:
    """Pack dense box and classification decoder outputs together.

    Autograd is preserved. Rearrangement borrows a view when possible and otherwise copies.

    Args:
        boxes: Floating [batch, queries, 6] geometry in (cx, cy, w, h, cos(theta),
            sin(theta)) order.
        logits: Same-device floating [batch, queries, classes] scores before
            sigmoid; sample/query order must match boxes.

    Returns:
        Ok(Predictions) with flattened sample/query order, or Err(InputError)
        for invalid fields.
    """
    match pack_dense_boxes(boxes):
        case Err() as failure:
            return failure
        case Ok(value=geometry):
            pass
    if logits.ndim != 3 or logits.shape[:2] != boxes.shape[:2]:
        return Err(
            InputError(ErrorKind.SHAPE, "logits", "expected [batch, queries, classes]")
        )
    return make_packed_predictions(geometry, rearrange(logits, "b q c -> (b q) c"))


def pack_targets(
    samples: Sequence[TargetSample], *, device: torch.device, dtype: torch.dtype
) -> Result[Targets, InputError]:
    """Pack variable-length labeled targets without losing empty samples.

    Concatenation allocates new storage and preserves box gradients.

    Args:
        samples: TargetSample records in batch order, each holding [count, 6] boxes,
            and aligned int64 [count] class labels.
        device: Required shared device for all input fields and output storage; no
            tensor is moved.
        dtype: Required box dtype, including empty storage for a zero-sample batch.

    Returns:
        Ok(Targets) in sample order or Err(InputError) for invalid
        box/label/device contracts.
    """
    match pack_boxes([sample.boxes for sample in samples], device=device, dtype=dtype):
        case Err() as failure:
            return failure
        case Ok(value=geometry):
            pass
    for sample in samples:
        match _validate_target_fields(*sample):
            case Err() as failure:
                return failure
            case Ok():
                pass
    labels = (
        torch.cat([sample.labels for sample in samples])
        if samples
        else torch.empty(0, dtype=torch.int64, device=device)
    )
    return Ok(Targets(geometry, labels))
