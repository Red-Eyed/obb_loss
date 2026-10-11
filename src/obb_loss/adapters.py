"""Explicit boundary conversion from ordinary detector tensors to packed boxes."""

from collections.abc import Sequence

import torch
from einops import rearrange
from torch import Tensor

from obb_loss.packed import ErrorKind, InputError, PackedBoxes, make_packed_boxes
from obb_loss.result import Err, Ok, Result


def pack_boxes(
    boxes: Sequence[Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Result[PackedBoxes, InputError]:
    """Concatenate per-image `[n_i, 6]` tensors, preserving gradients and empty images.

    Device/dtype are explicit so even zero-image input has a defined representation.
    No input is moved, detached, or mutated. Concatenation allocates packed storage.
    """
    counts = [0]
    for index, item in enumerate(boxes):
        if item.device != device or item.dtype != dtype:
            return Err(
                InputError(ErrorKind.DTYPE, f"boxes[{index}]", "device/dtype mismatch")
            )
        if item.ndim != 2 or item.shape[1] != 6:
            return Err(
                InputError(ErrorKind.SHAPE, f"boxes[{index}]", "expected [n_i, 6]")
            )
        counts.append(counts[-1] + item.shape[0])
    values = (
        torch.cat(tuple(boxes))
        if boxes
        else torch.empty((0, 6), device=device, dtype=dtype)
    )
    cu = torch.tensor(counts, device=device, dtype=torch.int64)
    return make_packed_boxes(values, cu)


def pack_dense_boxes(boxes: Tensor) -> Result[PackedBoxes, InputError]:
    """Flatten `[b, q, 6]` DETR geometry, retaining autograd and regular image offsets.

    Rearrangement borrows a view where strides allow it and otherwise copies. This
    adapter validates geometry; it does not match predictions or orchestrate layers.
    """
    if boxes.ndim != 3 or boxes.shape[2] != 6:
        return Err(InputError(ErrorKind.SHAPE, "boxes", "expected [b, q, 6]"))
    flat = rearrange(boxes, "b q c -> (b q) c")
    cu = torch.arange(boxes.shape[0] + 1, device=boxes.device) * boxes.shape[1]
    return make_packed_boxes(flat, cu)


def boxes_from_angles(boxes: Tensor) -> Result[Tensor, InputError]:
    """Convert explicit `[n, 5]` cx/cy/w/h/radians geometry to direction vectors.

    No unit conversion or canonicalization is implicit; gradients pass through
    sine/cosine. Full geometry admission is performed by packing helpers.
    """
    if boxes.ndim != 2 or boxes.shape[1] != 5 or not boxes.is_floating_point():
        return Err(InputError(ErrorKind.SHAPE, "boxes", "expected floating [n, 5]"))
    angle = boxes[:, 4:5]
    return Ok(torch.cat((boxes[:, :4], angle.cos(), angle.sin()), dim=-1))
