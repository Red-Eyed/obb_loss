"""Explicit differentiable reductions with defined empty behavior."""

from typing import assert_never

from torch import Tensor

from obb_loss.config import Reduction


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
