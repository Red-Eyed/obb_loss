"""Borrowed packed tensors and box-specific admission checks."""

from __future__ import annotations

from enum import Enum
from typing import NamedTuple

import torch
from torch import Tensor

from obb_loss.result import Err, Ok, Result


class ErrorKind(Enum):
    """Recoverable input contract violations."""

    SHAPE = "shape"
    DTYPE = "dtype"
    DEVICE = "device"
    OFFSETS = "offsets"
    GEOMETRY = "geometry"
    DOMAIN = "domain"
    ALIGNMENT = "alignment"
    CONFIG = "config"
    CALLBACK = "callback"


class InputError(NamedTuple):
    """Describe a recoverable input-contract rejection.

    Attributes:
        kind: Category of contract violation used for typed recovery.
        field: Rejected parameter or field path.
        detail: Human-readable reason describing the required contract.
    """

    kind: ErrorKind
    field: str
    detail: str


class PackedBatch(NamedTuple):
    """Borrow packed tensor values and their sample boundaries.

    Attributes:
        values: Tensor with its first axis indexing packed items; box APIs require
            floating [count, 6] geometry in (cx, cy, w, h, cos(theta), sin(theta)) order.
        cu_seqlens: Same-device int32/int64 [samples + 1] cumulative item counts.
            Starts at zero, ends at count; repeated offsets preserve empty samples.
    """

    values: Tensor
    cu_seqlens: Tensor[[int]]


def validate_geometry(
    boxes: Tensor, field: str = "boxes"
) -> Result[Tensor, InputError]:
    """Validate six-coordinate oriented-box geometry.

    Args:
        boxes: Floating [count, 6] geometry; finite centers, positive sides, and
            nonzero direction vectors on CPU/CUDA.
        field: Parameter path used to identify rejected geometry in diagnostics.

    Returns:
        Ok(the borrowed tensor) or Err(InputError) identifying a geometry, shape,
        dtype, or device violation.
    """
    if boxes.ndim != 2 or boxes.shape[1] != 6:
        return Err(InputError(ErrorKind.SHAPE, field, "expected [n, 6]"))
    if boxes.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        return Err(InputError(ErrorKind.DTYPE, field, "expected floating geometry"))
    if boxes.device.type not in ("cpu", "cuda"):
        return Err(InputError(ErrorKind.DEVICE, field, "float64 requires CPU or CUDA"))
    if not bool(torch.isfinite(boxes).all()):
        return Err(InputError(ErrorKind.GEOMETRY, field, "coordinates must be finite"))
    if not bool((boxes[:, 2:4] > 0).all()):
        return Err(InputError(ErrorKind.GEOMETRY, field, "sides must be positive"))
    if not bool((boxes[:, 4:6].abs().amax(dim=-1) > 0).all()):
        return Err(InputError(ErrorKind.GEOMETRY, field, "directions must be nonzero"))
    return Ok(boxes)


def validate_layout(
    offsets: Tensor, total: int, device: torch.device
) -> Result[Tensor, InputError]:
    """Validate cumulative packed sample boundaries.

    Args:
        offsets: Int32/int64 [batch + 1] cumulative counts; monotonic, beginning at
            zero and ending at total.
        total: Number of packed items the final offset must equal.
        device: Required device of offsets, shared with the packed values.

    Returns:
        Ok(the borrowed offsets) or Err(InputError) for invalid boundaries, dtype,
        or device.
    """
    if offsets.ndim != 1 or offsets.numel() == 0:
        return Err(InputError(ErrorKind.SHAPE, "cu_seqlens", "expected nonempty [b+1]"))
    if offsets.dtype not in (torch.int32, torch.int64):
        return Err(InputError(ErrorKind.DTYPE, "cu_seqlens", "expected int32 or int64"))
    if offsets.device != device:
        return Err(
            InputError(ErrorKind.DEVICE, "cu_seqlens", "must match boxes device")
        )
    if int(offsets[0]) != 0 or int(offsets[-1]) != total:
        return Err(
            InputError(ErrorKind.OFFSETS, "cu_seqlens", "endpoints must be 0 and n")
        )
    if not bool((offsets[1:] >= offsets[:-1]).all()):
        return Err(InputError(ErrorKind.OFFSETS, "cu_seqlens", "must be monotonic"))
    return Ok(offsets)


def make_packed_boxes(
    boxes: Tensor,
    cu_seqlens: Tensor,
) -> Result[PackedBatch, InputError]:
    """Validate and borrow packed box tensors.

    Value checks synchronize accelerators. Admitted tensors must remain valid while used.

    Args:
        boxes: Floating [count, 6] geometry (cx, cy, w, h, cos(theta), sin(theta));
            finite values, positive sides, and nonzero directions are required.
        cu_seqlens: Same-device int32/int64 [batch + 1] offsets, starting at zero
            and ending at count. Repeated offsets represent empty samples.

    Returns:
        Ok(PackedBatch) borrowing the tensors, or Err(InputError) identifying an
        invalid geometry/layout field.
    """
    match validate_geometry(boxes):
        case Err() as failure:
            return failure
        case Ok():
            pass
    match validate_layout(cu_seqlens, boxes.shape[0], boxes.device):
        case Err() as failure:
            return failure
        case Ok():
            return Ok(PackedBatch(boxes, cu_seqlens))


def sample_indices(packed: PackedBatch) -> Tensor[[int]]:
    """Map packed boxes to their owning samples.

    Args:
        packed: Valid packed geometry whose offsets define each sample segment.

    Returns:
        Same-device int64 [count] sample indices, including gaps for empty samples.
    """
    cu = packed.cu_seqlens
    return torch.repeat_interleave(
        torch.arange(cu.shape[0] - 1, device=cu.device),
        cu[1:] - cu[:-1],
        output_size=packed.values.shape[0],
    )
