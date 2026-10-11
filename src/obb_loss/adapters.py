"""Explicit boundary conversion from ordinary detector tensors to packed boxes."""

from collections.abc import Sequence

import torch
from einops import rearrange
from torch import Tensor

from obb_loss.packed import ErrorKind, InputError, PackedBatch, make_packed_boxes
from obb_loss.result import Err, Ok, Result


def pack_boxes(
    boxes: Sequence[Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Result[PackedBatch, InputError]:
    """Pack variable-length box tensors in sample order.

    Concatenation preserves autograd and never mutates input tensors.

    Args:
        boxes: One [count, 6] tensor per sample, in (cx, cy, w, h, cos(theta),
            sin(theta)) order. Empty samples retain their position.
        device: Required device for every input and the packed output; inputs are
            never moved. Defines storage for a zero-sample batch.
        dtype: Required floating geometry dtype for every input and output; no
            implicit conversion.

    Returns:
        Ok(PackedBatch) with newly concatenated geometry and int64 offsets, or
        Err(InputError) for invalid shape, geometry, device, or dtype.
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


def pack_dense_boxes(boxes: Tensor) -> Result[PackedBatch, InputError]:
    """Pack dense decoder box outputs in sample/query order.

    Autograd is preserved. Packing borrows a view when strides allow and otherwise copies.

    Args:
        boxes: Floating [batch, queries, 6] geometry in (cx, cy, w, h, cos(theta),
            sin(theta)) order, with positive sides and nonzero directions.

    Returns:
        Ok(PackedBatch) with [batch * queries, 6] geometry and regular sample
        offsets, or Err(InputError) for invalid inputs.
    """
    if boxes.ndim != 3 or boxes.shape[2] != 6:
        return Err(InputError(ErrorKind.SHAPE, "boxes", "expected [b, q, 6]"))
    flat = rearrange(boxes, "b q c -> (b q) c")
    cu = torch.arange(boxes.shape[0] + 1, device=boxes.device) * boxes.shape[1]
    return make_packed_boxes(flat, cu)


def boxes_from_angles(boxes: Tensor) -> Result[Tensor, InputError]:
    """Convert angle-based boxes to the six-coordinate geometry representation.

    Angles are not canonicalized. Conversion preserves gradients; packing performs full geometry validation.

    Args:
        boxes: Floating [count, 5] values (cx, cy, width, height, theta); theta is
            in radians. Coordinates and side units are retained.

    Returns:
        Ok([count, 6] tensor) ending in cos(theta), sin(theta), or Err(InputError)
        for invalid shape/dtype.
    """
    if boxes.ndim != 2 or boxes.shape[1] != 5 or not boxes.is_floating_point():
        return Err(InputError(ErrorKind.SHAPE, "boxes", "expected floating [n, 5]"))
    angle = boxes[:, 4:5]
    return Ok(torch.cat((boxes[:, :4], angle.cos(), angle.sin()), dim=-1))
