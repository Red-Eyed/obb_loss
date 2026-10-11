"""Admission of caller-supplied classification quality, independent of assignment."""

import torch
from torch import Tensor

from obb_loss.packed import ErrorKind, InputError
from obb_loss.result import Err, Ok, Result


def validate_quality(
    logits: Tensor, quality: Tensor
) -> Result[tuple[Tensor, Tensor], InputError]:
    """Require equal floating `[n, c]` tensors with finite logits and quality in [0,1].

    Zero is a negative class. No target is synthesized or detached. Value admission
    synchronizes accelerator tensors. Different floating dtypes are promoted internally.
    """
    if logits.ndim != 2 or quality.shape != logits.shape:
        return Err(
            InputError(ErrorKind.SHAPE, "quality", "expected equal [n, c] tensors")
        )
    if logits.shape[1] == 0:
        return Err(InputError(ErrorKind.SHAPE, "logits", "at least one class required"))
    if not logits.is_floating_point() or not quality.is_floating_point():
        return Err(InputError(ErrorKind.DTYPE, "quality", "floating tensors required"))
    if logits.device != quality.device or logits.device.type not in ("cpu", "cuda"):
        return Err(
            InputError(ErrorKind.DEVICE, "quality", "same CPU/CUDA device required")
        )
    if not bool(torch.isfinite(logits).all()):
        return Err(InputError(ErrorKind.DOMAIN, "logits", "must be finite"))
    if not bool((torch.isfinite(quality) & (quality >= 0) & (quality <= 1)).all()):
        return Err(
            InputError(ErrorKind.DOMAIN, "quality", "must be finite and in [0,1]")
        )
    return Ok((logits, quality))
