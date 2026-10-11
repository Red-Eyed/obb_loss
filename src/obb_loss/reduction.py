"""Explicit differentiable reductions with defined empty behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_never

import torch

from obb_loss.config import Reduction

if TYPE_CHECKING:
    from jaxtyping import Float, Int
    from torch import Tensor


def reduce_loss(values: Tensor, reduction: Reduction) -> Tensor:
    """Reduce unreduced elements; empty sum/mean is a graph-connected zero."""
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
    """Sum packed `[n]` values into `[batch]`, retaining zero-length samples.

    Offsets must be valid, same-device cumulative lengths ending at n. Inputs
    are borrowed. Known output sizes avoid data-dependent shape synchronization.
    Empty outputs remain connected to autograd.
    """
    count = cu_seqlens.numel() - 1
    indices = torch.repeat_interleave(
        torch.arange(count, device=values.device),
        cu_seqlens[1:] - cu_seqlens[:-1],
        output_size=values.shape[0],
    )
    return values.new_zeros(count).index_add(0, indices, values)
