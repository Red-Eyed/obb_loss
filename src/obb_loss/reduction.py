"""Explicit differentiable reductions with defined empty behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_never

import torch

from obb_loss.config import Reduction

if TYPE_CHECKING:
    from jaxtyping import Float, Int
    from torch import Tensor


def reduce_loss(values: Tensor, reduction: Reduction) -> Tensor:
    """Reduce element losses with defined empty behavior.

    Args:
        values: Unreduced floating losses whose elements are the reduction units.
        reduction: NONE retains values; SUM adds all elements; MEAN averages
            elements, including zeros.

    Returns:
        The original tensor for NONE or a scalar for SUM/MEAN; empty scalar
        reductions are gradient-connected zeros.
    """
    match reduction:
        case Reduction.NONE:
            return values
        case Reduction.SUM:
            return values.sum()
        case Reduction.MEAN:
            return values.sum() / max(values.numel(), 1)
        case _:
            assert_never(reduction)


def sum_samples(
    values: Float[Tensor, "n"],  # noqa: F821, UP037
    cu_seqlens: Int[Tensor, "bp1"],  # noqa: F821, UP037
) -> Float[Tensor, "b"]:  # noqa: F821, UP037
    """Sum packed values within their owning samples.

    This pure helper trusts offsets and does not synchronize to infer output sizes.

    Args:
        values: Floating [count] element losses in packed sample order.
        cu_seqlens: Valid same-device cumulative offsets, beginning at zero and
            ending at count; repeated offsets define empty samples.

    Returns:
        Same-device [batch] sums, retaining zeros for empty samples and preserving
        gradients.
    """
    count = cu_seqlens.numel() - 1
    indices = torch.repeat_interleave(
        torch.arange(count, device=values.device),
        cu_seqlens[1:] - cu_seqlens[:-1],
        output_size=values.shape[0],
    )
    return values.new_zeros(count).index_add(0, indices, values)
