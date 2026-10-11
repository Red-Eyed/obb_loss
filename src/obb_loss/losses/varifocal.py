"""Varifocal loss: Zhang et al., CVPR 2021, https://arxiv.org/abs/2008.13367."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
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
    C = IntVar("C")


def _validate_inputs(logits: Tensor, quality: Tensor) -> Result[None, InputError]:
    """Check aligned class tensors, floating dtypes and shared device."""
    if logits.ndim != 2 or quality.shape != logits.shape or logits.shape[1] == 0:
        return Err(
            InputError(
                ErrorKind.SHAPE, "quality", "expected equal [items, classes>0] tensors"
            )
        )
    if not logits.is_floating_point() or not quality.is_floating_point():
        return Err(InputError(ErrorKind.DTYPE, "inputs", "floating tensors required"))
    if logits.device != quality.device or logits.device.type not in ("cpu", "cuda"):
        return Err(
            InputError(ErrorKind.DEVICE, "inputs", "same CPU/CUDA device required")
        )
    return Ok(None)


def varifocal_loss(
    logits: Tensor[[N, C]],
    quality: Tensor[[N, C]],
    *,
    alpha: float = 0.75,
    gamma: float = 2.0,
    quality_weighted: bool = True,
) -> Result[Tensor[[N, C]], InputError]:
    """Compute unreduced Varifocal loss with supplied quality targets.

    Inputs must share a CPU/CUDA device and be finite. Autocast is disabled
    locally. The logsigmoid form keeps fractional negative modulation finite
    at probability underflow. Only tensor metadata is validated.

    Args:
        logits: Floating [queries, classes] scores before sigmoid; no background
            column.
        quality: Floating [queries, classes] quality in [0, 1], aligned with logits;
            zero marks a negative class. Target gradients are preserved.
        alpha: Finite nonnegative multiplier for negative-class BCE weights.
        gamma: Finite nonnegative exponent on predicted probability for negative
            classes; larger values suppress easy negatives.
        quality_weighted: Whether positive-class BCE is weighted by its quality
            target; false uses unit positive weight.

    Returns:
        Ok(float64 [queries, classes] losses), or Err for shape/dtype/device
        mismatch. Values are not reduced or weighted globally.
    """
    match _validate_inputs(logits, quality):
        case Err() as failure:
            return failure
        case Ok():
            pass
    with torch.autocast(logits.device.type, enabled=False):
        scores, targets = logits.double(), quality.double()
        negative_weight = alpha * (F.logsigmoid(scores) * gamma).exp()
        positive_weight = targets if quality_weighted else torch.ones_like(targets)
        weight = torch.where(targets > 0, positive_weight, negative_weight)
        return Ok(
            F.binary_cross_entropy_with_logits(scores, targets, reduction="none")
            * weight
        )
