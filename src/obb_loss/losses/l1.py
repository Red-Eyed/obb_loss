"""DETR box L1: Carion et al., ECCV 2020, https://arxiv.org/abs/2005.12872."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from obb_loss.result import (
    Err,
    ErrorKind,
    InputError,
    Ok,
    Result,
)

if TYPE_CHECKING:
    from shape_extensions import IntVar

    N = IntVar("N")


def _validate_inputs(preds: Tensor, targets: Tensor) -> Result[None, InputError]:
    """Check aligned six-coordinate boxes, floating dtypes and shared device."""
    if preds.ndim != 2 or preds.shape != targets.shape or preds.shape[1] != 6:
        return Err(
            InputError(ErrorKind.SHAPE, "boxes", "expected aligned [pairs, 6] tensors")
        )
    if not preds.is_floating_point() or not targets.is_floating_point():
        return Err(InputError(ErrorKind.DTYPE, "inputs", "floating tensors required"))
    if preds.device != targets.device or preds.device.type not in ("cpu", "cuda"):
        return Err(
            InputError(ErrorKind.DEVICE, "inputs", "same CPU/CUDA device required")
        )
    return Ok(None)


def box_l1_loss(
    preds: Tensor[[N, 6]],
    targets: Tensor[[N, 6]],
) -> Result[Tensor[[N]], InputError]:
    """Compute center and side L1 error for each matched box pair.

    Direction coordinates are excluded. Autocast is disabled locally and
    gradients flow to both tensors. Only tensor metadata is validated.

    Args:
        preds: Predicted aligned boxes in (cx, cy, width, height, cos(theta),
            sin(theta)) order; positive sides and nonzero directions, in shared
            coordinate units. Shape [pairs, 6].
        targets: Reference boxes in the same order and units as preds, paired
            elementwise with predictions. Same [pairs, 6] shape and CPU/CUDA device.

    Returns:
        Ok(float64 [pairs] sums of center/side absolute errors), or Err for tensor
        metadata mismatch. Values use the supplied coordinate units.
    """
    match _validate_inputs(preds, targets):
        case Err() as failure:
            return failure
        case Ok():
            pass
    with torch.autocast(preds.device.type, enabled=False):
        return Ok((preds[:, :4].double() - targets[:, :4].double()).abs().sum(dim=-1))
