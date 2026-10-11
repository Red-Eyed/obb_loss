"""Quality focal loss: Li et al., NeurIPS 2020, https://arxiv.org/abs/2006.04388."""

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


def quality_focal_loss(
    logits: Tensor[[N, C]],
    quality: Tensor[[N, C]],
    *,
    beta: float = 2.0,
) -> Result[Tensor[[N, C]], InputError]:
    """Compute unreduced Quality Focal loss with supplied quality targets.

    Inputs must share a CPU/CUDA device and be finite. Autocast is disabled
    locally; target gradients remain connected. beta >=1 avoids singular
    gradients at equal quality. Only tensor metadata is validated.

    Args:
        logits: Floating [queries, classes] scores before sigmoid; no background
            column.
        quality: Floating [queries, classes] quality in [0, 1], aligned with logits;
            zero marks a negative class. Target gradients are preserved.
        beta: Finite exponent >=1 on abs(quality - sigmoid(logits)); larger values
            suppress predictions close to target quality.

    Returns:
        Ok(float64 [queries, classes] modulated BCE losses), or Err for tensor
        metadata mismatch. Quality range and scalar parameters are trusted.
    """
    match _validate_inputs(logits, quality):
        case Err() as failure:
            return failure
        case Ok():
            pass
    with torch.autocast(logits.device.type, enabled=False):
        scores, targets = logits.double(), quality.double()
        modulation = (targets - scores.sigmoid()).abs().pow(beta)
        return Ok(
            F.binary_cross_entropy_with_logits(scores, targets, reduction="none")
            * modulation
        )
