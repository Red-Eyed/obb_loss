"""Borrowed packed geometry and checked admission outside tensor execution."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, NamedTuple

import torch
from torch import Tensor

from obb_loss.result import Err, Ok, Result

if TYPE_CHECKING:
    from jaxtyping import Float, Int


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
    """Identify the rejected field and its actionable reason."""

    kind: ErrorKind
    field: str
    detail: str


class PackedBoxes(NamedTuple):
    """Borrow `[n, 6]` geometry and `[b+1]` image offsets without mutating them.

    Geometry is cx, cy, positive width/height, and a nonzero single-angle direction.
    Centers and sides use one shared coordinate frame/unit. Records are immutable;
    tensor storage is not. Callers must preserve admitted contents during execution.
    Repeated offsets retain empty images. Direct construction is the trusted path.
    """

    # Jaxtyping dimensions are strings, not Python expressions.
    boxes: Float[Tensor, "n 6"]  # noqa: F722
    cu_seqlens: Int[Tensor, "bp1"]  # noqa: F821, UP037


def validate_geometry(
    boxes: Tensor, field: str = "boxes"
) -> Result[Tensor, InputError]:
    """Check six-coordinate geometry, synchronizing value checks on accelerators."""
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
    """Admit monotonic same-device offsets, including a zero-image `[0]` layout."""
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
    boxes: Tensor, cu_seqlens: Tensor
) -> Result[PackedBoxes, InputError]:
    """Borrow admitted geometry and offsets; rejection never manufactures a box."""
    match validate_geometry(boxes):
        case Err() as failure:
            return failure
        case Ok():
            pass
    match validate_layout(cu_seqlens, boxes.shape[0], boxes.device):
        case Err() as failure:
            return failure
        case Ok():
            return Ok(PackedBoxes(boxes, cu_seqlens))


def image_indices(packed: PackedBoxes) -> Int[Tensor, "n"]:  # noqa: F821, UP037
    """Map each box to its image without lists or data-dependent output sizing."""
    cu = packed.cu_seqlens
    return torch.repeat_interleave(
        torch.arange(cu.shape[0] - 1, device=cu.device),
        cu[1:] - cu[:-1],
        output_size=packed.boxes.shape[0],
    )
