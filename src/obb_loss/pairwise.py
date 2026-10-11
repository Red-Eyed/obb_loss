"""Chunked per-sample rectangular cost blocks without cross-sample pairs."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch
from torch import Tensor

from obb_loss.losses.stable_probiou import stable_probiou_loss, validate_pairs
from obb_loss.packed import ErrorKind, InputError, PackedBoxes, make_packed_boxes
from obb_loss.result import Err, Ok, Result

if TYPE_CHECKING:
    from obb_loss.config import MatcherConfig
    from obb_loss.protocols import BoxFunction


class CostBlock(NamedTuple):
    """Metadata for one sample's row-major rectangular block in packed costs."""

    pred_start: int
    target_start: int
    rows: int
    columns: int
    start: int
    end: int


class PackedCosts(NamedTuple):
    """Float64 row-major costs and ordered per-sample rectangular metadata."""

    values: Tensor
    blocks: tuple[CostBlock, ...]


def validate_output(
    values: Tensor, count: int, device: torch.device
) -> Result[Tensor, InputError]:
    """Check observable callback output; mathematical validity remains user-owned."""
    if not isinstance(values, Tensor):
        return Err(InputError(ErrorKind.CALLBACK, "output", "expected a tensor"))
    if (
        values.shape != (count,)
        or values.device != device
        or not values.is_floating_point()
    ):
        return Err(
            InputError(
                ErrorKind.CALLBACK, "output", "expected floating [n] on input device"
            )
        )
    if not bool(torch.isfinite(values).all()):
        return Err(InputError(ErrorKind.CALLBACK, "output", "cost/loss must be finite"))
    return Ok(values)


def cost_blocks(
    preds: PackedBoxes, targets: PackedBoxes, limit: int
) -> Result[tuple[CostBlock, ...], InputError]:
    """Transfer offsets once to host and reject solver storage beyond the limit."""
    if (
        preds.cu_seqlens.shape != targets.cu_seqlens.shape
        or preds.boxes.device != targets.boxes.device
    ):
        return Err(
            InputError(ErrorKind.ALIGNMENT, "batch", "sample count/device must match")
        )
    pred_cu = preds.cu_seqlens.detach().cpu()
    target_cu = targets.cu_seqlens.detach().cpu()
    blocks: list[CostBlock] = []
    start = 0
    for sample in range(pred_cu.shape[0] - 1):
        p, t = int(pred_cu[sample]), int(target_cu[sample])
        rows, columns = int(pred_cu[sample + 1]) - p, int(target_cu[sample + 1]) - t
        end = start + rows * columns
        if end > limit:
            return Err(
                InputError(
                    ErrorKind.DOMAIN,
                    "pairs",
                    "total pair count exceeds configured storage limit",
                )
            )
        blocks.append(CostBlock(p, t, rows, columns, start, end))
        start = end
    return Ok(tuple(blocks))


def _fill_block(
    output: Tensor,
    block: CostBlock,
    preds: PackedBoxes,
    targets: PackedBoxes,
    function: BoxFunction,
    chunk_size: int,
) -> Result[Tensor, InputError]:
    """Evaluate bounded aligned chunks; zero-sized blocks never divide by zero."""
    for start in range(block.start, block.end, chunk_size):
        end = min(start + chunk_size, block.end)
        index = torch.arange(
            start - block.start, end - block.start, device=output.device
        )
        first = preds.boxes[block.pred_start + index // block.columns]
        second = targets.boxes[block.target_start + index % block.columns]
        match validate_pairs(first, second):
            case Err() as failure:
                return failure
            case Ok():
                pass
        values = function(first, second)
        match validate_output(values, end - start, output.device):
            case Err() as failure:
                return failure
            case Ok():
                output[start:end] = values
    return Ok(output)


def pairwise_costs(
    preds: PackedBoxes,
    targets: PackedBoxes,
    config: MatcherConfig,
    function: BoxFunction = stable_probiou_loss,
) -> Result[PackedCosts, InputError]:
    """Construct checked segmented costs, preserving gradients if invoked directly.

    Full output is O(sum(n_i*m_i)); temporary pair indices/geometry are bounded by
    chunk_size. max_pairs limits full storage. No all-pairs global matrix is built.
    Offsets transfer to host once; checked chunks synchronize. Assignment callers
    should use no_grad. Custom function bugs propagate; output failures are typed.
    """
    for packed in (preds, targets):
        match make_packed_boxes(*packed):
            case Err() as failure:
                return failure
            case Ok():
                pass
    match cost_blocks(preds, targets, config.max_pairs):
        case Err() as failure:
            return failure
        case Ok(value=blocks):
            pass
    total = blocks[-1].end if blocks else 0
    output = torch.empty(total, device=preds.boxes.device, dtype=torch.float64)
    for block in blocks:
        match _fill_block(output, block, preds, targets, function, config.chunk_size):
            case Err() as failure:
                return failure
            case Ok():
                pass
    return Ok(PackedCosts(output, blocks))
