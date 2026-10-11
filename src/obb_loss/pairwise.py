"""Chunked per-sample rectangular cost blocks without cross-sample pairs."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch
from torch import Tensor

from obb_loss.losses.stable_probiou import stable_probiou_loss, validate_pair_domain
from obb_loss.packed import ErrorKind, InputError, PackedBatch, make_packed_boxes
from obb_loss.result import Err, Ok, Result

if TYPE_CHECKING:
    from obb_loss.config import MatcherConfig
    from obb_loss.protocols import BoxFunction


class CostBlock(NamedTuple):
    """Identify one sample rectangle within the packed cost buffer.

    Attributes:
        pred_start: Global row offset into packed predictions.
        target_start: Global column offset into packed references.
        rows: Prediction count for this sample.
        columns: Reference count for this sample.
        start: Inclusive offset into packed costs.
        end: Exclusive offset, with end - start equal to rows * columns.
    """

    pred_start: int
    target_start: int
    rows: int
    columns: int
    start: int
    end: int


class PackedCosts(NamedTuple):
    """Store rectangular costs for each sample without cross-sample pairs.

    Attributes:
        values: Floating one-dimensional row-major costs concatenated in sample
            order; pairwise builders produce float64.
        blocks: Ordered sample metadata identifying row/column counts, source
            offsets, and ranges within values.
    """

    values: Tensor
    blocks: tuple[CostBlock, ...]


def validate_output(
    values: Tensor, count: int, device: torch.device
) -> Result[Tensor, InputError]:
    """Validate the observable output contract of a custom cost or loss.

    Args:
        values: Callback result expected to contain one floating value per aligned
            pair.
        count: Required number of output elements.
        device: Required device shared with the input geometry.

    Returns:
        Ok(values) or Err(InputError) for non-tensor, shape, dtype, device, or
        nonfinite output.
    """
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
    preds: PackedBatch, targets: PackedBatch, limit: int
) -> Result[tuple[CostBlock, ...], InputError]:
    """Describe bounded rectangular cost storage for each sample.

    Offsets transfer once to the host; no cost values are computed.

    Args:
        preds: Valid packed prediction geometry; segment lengths define rectangle
            rows.
        targets: Valid same-batch/device reference geometry; segment lengths define
            columns.
        limit: Maximum total number of cost entries across all rectangles.

    Returns:
        Ok(ordered CostBlock metadata) or Err(InputError) for sample/device mismatch
        or excessive pair counts.
    """
    if (
        preds.cu_seqlens.shape != targets.cu_seqlens.shape
        or preds.values.device != targets.values.device
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
    preds: PackedBatch,
    targets: PackedBatch,
    function: BoxFunction,
    chunk_size: int,
) -> Result[Tensor, InputError]:
    """Evaluate bounded aligned chunks; zero-sized blocks never divide by zero."""
    for start in range(block.start, block.end, chunk_size):
        end = min(start + chunk_size, block.end)
        index = torch.arange(
            start - block.start, end - block.start, device=output.device
        )
        first = preds.values[block.pred_start + index // block.columns]
        second = targets.values[block.target_start + index % block.columns]
        match validate_pair_domain(first, second):
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
    preds: PackedBatch,
    targets: PackedBatch,
    config: MatcherConfig,
    function: BoxFunction = stable_probiou_loss,
) -> Result[PackedCosts, InputError]:
    """Build rectangular geometry cost blocks within each sample.

    Full storage is sum(queries_i * targets_i), not a global cross-sample matrix. Direct calls retain gradients; custom callable bugs propagate.

    Args:
        preds: Packed candidate boxes whose sample segments define cost rows.
        targets: Packed reference boxes whose corresponding segments define cost
            columns; same sample count/device and coordinate units as preds.
        config: Chunk-size limit on temporary pairs and max_pairs limit on total
            stored costs.
        function: Aligned [pairs, 6] -> [pairs] cost callable; outputs must be
            finite floating costs on the input device, with smaller values
            preferred.

    Returns:
        Ok(PackedCosts) with float64 row-major values and per-sample block metadata,
        or Err(InputError) for invalid input/output contracts or storage limits.
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
    output = torch.empty(total, device=preds.values.device, dtype=torch.float64)
    for block in blocks:
        match _fill_block(output, block, preds, targets, function, config.chunk_size):
            case Err() as failure:
                return failure
            case Ok():
                pass
    return Ok(PackedCosts(output, blocks))
