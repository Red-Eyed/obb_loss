"""Packed predictions and labeled targets for a single set-prediction layer."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch
from einops import rearrange
from torch import Tensor

from obb_loss.adapters import pack_boxes, pack_dense_boxes
from obb_loss.packed import ErrorKind, InputError, PackedBoxes, make_packed_boxes
from obb_loss.result import Err, Ok, Result

if TYPE_CHECKING:
    from collections.abc import Sequence

    from jaxtyping import Float, Int


class PackedPredictions(NamedTuple):
    """Borrow one decoder layer of packed geometry and class logits.

    Attributes:
        geometry: Packed query boxes and sample offsets.
        logits: Floating [queries, classes] pre-sigmoid scores, sharing geometry
            order/device. No background column.
    """

    geometry: PackedBoxes
    logits: Float[Tensor, "n c"]  # noqa: F722


class PackedTargets(NamedTuple):
    """Borrow packed geometry with labeled caller-supplied quality.

    Attributes:
        geometry: Packed reference boxes and sample offsets.
        labels: Same-device int64 [targets] zero-based class indices in geometry
            order.
        quality: Same-device floating [targets] values in [0,1], assigned to matched
            classes. Zero is negative in VFL; gradients are preserved.
    """

    geometry: PackedBoxes
    labels: Int[Tensor, "m"]  # noqa: F821, UP037
    quality: Float[Tensor, "m"]  # noqa: F821, UP037


class TargetSample(NamedTuple):
    """Represent annotations for one sample before packing.

    Attributes:
        boxes: Floating [count, 6] reference boxes in (cx, cy, w, h, cos(theta),
            sin(theta)) order.
        labels: Same-device int64 [count] zero-based class IDs, aligned with boxes.
        quality: Same-device floating [count] values in [0,1] supplied by the
            caller, aligned with boxes.
    """

    boxes: Float[Tensor, "m 6"]  # noqa: F722
    labels: Int[Tensor, "m"]  # noqa: F821, UP037
    quality: Float[Tensor, "m"]  # noqa: F821, UP037


def make_packed_predictions(
    geometry: PackedBoxes, logits: Tensor
) -> Result[PackedPredictions, InputError]:
    """Validate and borrow one layer of packed box predictions and logits.

    Args:
        geometry: Valid packed query boxes; offsets define sample ownership.
        logits: Finite floating [queries, classes] scores before sigmoid, in
            geometry order and on its device; at least one class and no background
            column.

    Returns:
        Ok(PackedPredictions) borrowing both fields, or Err(InputError) for
        geometry, layout, shape, dtype, device, or nonfinite logits.
    """
    match make_packed_boxes(*geometry):
        case Err() as failure:
            return failure
        case Ok():
            pass
    if (
        logits.ndim != 2
        or logits.shape[0] != geometry.boxes.shape[0]
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
    if logits.device != geometry.boxes.device:
        return Err(InputError(ErrorKind.DEVICE, "logits", "must share geometry device"))
    if not bool(torch.isfinite(logits).all()):
        return Err(InputError(ErrorKind.DOMAIN, "logits", "must be finite"))
    return Ok(PackedPredictions(geometry, logits))


def _validate_target_fields(
    boxes: Tensor, labels: Tensor, quality: Tensor
) -> Result[Tensor, InputError]:
    """Require aligned labels and quality without silently transferring tensors."""
    if labels.shape != (boxes.shape[0],) or quality.shape != labels.shape:
        return Err(
            InputError(ErrorKind.SHAPE, "targets", "labels and quality must be [count]")
        )
    if labels.dtype != torch.int64 or quality.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        return Err(
            InputError(
                ErrorKind.DTYPE, "targets", "int64 labels and floating quality required"
            )
        )
    if labels.device != boxes.device or quality.device != boxes.device:
        return Err(
            InputError(
                ErrorKind.DEVICE, "targets", "all target fields must share device"
            )
        )
    if not bool((labels >= 0).all()):
        return Err(
            InputError(ErrorKind.DOMAIN, "labels", "class labels must be nonnegative")
        )
    if not bool((torch.isfinite(quality) & (quality >= 0) & (quality <= 1)).all()):
        return Err(
            InputError(ErrorKind.DOMAIN, "quality", "must be finite and in [0,1]")
        )
    return Ok(quality)


def make_packed_targets(
    geometry: PackedBoxes, labels: Tensor, quality: Tensor
) -> Result[PackedTargets, InputError]:
    """Validate and borrow packed labeled quality targets.

    Args:
        geometry: Valid packed reference boxes; offsets define sample ownership.
        labels: Same-device int64 [targets] zero-based class IDs, ordered with
            geometry; upper bounds are checked against prediction logits during
            matching.
        quality: Same-device floating [targets] confidence/quality in [0, 1]
            supplied by the caller; never synthesized or detached.

    Returns:
        Ok(PackedTargets) borrowing every field, or Err(InputError) for invalid
        geometry, layout, label, quality, or device contracts.
    """
    match make_packed_boxes(*geometry):
        case Err() as failure:
            return failure
        case Ok():
            pass
    match _validate_target_fields(geometry.boxes, labels, quality):
        case Err() as failure:
            return failure
        case Ok():
            return Ok(PackedTargets(geometry, labels, quality))


def validate_sets(
    preds: PackedPredictions, targets: PackedTargets
) -> Result[tuple[PackedPredictions, PackedTargets], InputError]:
    """Validate current packed set records against one another.

    Args:
        preds: Packed geometry and [queries, classes] logits for one decoder layer,
            sharing box order and device.
        targets: Packed reference geometry, int64 class labels, and caller-supplied
            [targets] quality in [0, 1]; labels and quality follow box order.

    Returns:
        Ok(the borrowed prediction/target pair) or Err(InputError) for invalid
        fields, sample counts, devices, or out-of-range labels.
    """
    match make_packed_predictions(*preds):
        case Err() as failure:
            return failure
        case Ok():
            pass
    match make_packed_targets(*targets):
        case Err() as failure:
            return failure
        case Ok():
            pass
    if preds.geometry.cu_seqlens.shape != targets.geometry.cu_seqlens.shape:
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


def pack_predictions(
    boxes: Tensor, logits: Tensor
) -> Result[PackedPredictions, InputError]:
    """Pack dense box and classification decoder outputs together.

    Autograd is preserved. Rearrangement borrows a view when possible and otherwise copies.

    Args:
        boxes: Floating [batch, queries, 6] geometry in (cx, cy, w, h, cos(theta),
            sin(theta)) order.
        logits: Same-device floating [batch, queries, classes] scores before
            sigmoid; sample/query order must match boxes.

    Returns:
        Ok(PackedPredictions) with flattened sample/query order, or Err(InputError)
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
) -> Result[PackedTargets, InputError]:
    """Pack variable-length labeled targets without losing empty samples.

    Concatenation allocates new storage and preserves gradients, including quality gradients.

    Args:
        samples: TargetSample records in batch order, each holding [count, 6] boxes,
            int64 [count] labels, and caller-supplied [count] quality in [0, 1].
        device: Required shared device for all input fields and output storage; no
            tensor is moved.
        dtype: Required box dtype; also selects empty quality storage for a zero-
            sample batch. Nonempty quality tensors retain their floating dtype,
            promoted on concatenation if needed.

    Returns:
        Ok(PackedTargets) in sample order or Err(InputError) for invalid
        box/label/quality/device contracts.
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
    quality = (
        torch.cat([sample.quality for sample in samples])
        if samples
        else torch.empty(0, dtype=dtype, device=device)
    )
    return Ok(PackedTargets(geometry, labels, quality))
