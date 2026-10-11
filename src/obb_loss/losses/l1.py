"""DETR box L1: Carion et al., ECCV 2020, https://arxiv.org/abs/2005.12872."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor


def box_l1_loss(
    preds: Float[Tensor, "n 6"],  # noqa: F722
    targets: Float[Tensor, "n 6"],  # noqa: F722
) -> Float[Tensor, "n"]:  # noqa: F821, UP037
    """Compute center and side L1 error for each matched box pair.

    Direction coordinates are excluded. Inputs are trusted; autocast is disabled locally and gradients flow to both tensors.

    Args:
        preds: Predicted aligned boxes in (cx, cy, width, height, cos(theta),
            sin(theta)) order; positive sides and nonzero directions, in shared
            coordinate units. Shape [pairs, 6].
        targets: Reference boxes in the same order and units as preds, paired
            elementwise with predictions. Same [pairs, 6] shape and CPU/CUDA device.

    Returns:
        Float64 [pairs] sums of absolute errors in cx, cy, width, and height,
        measured in the supplied coordinate units.
    """
    with torch.autocast(preds.device.type, enabled=False):
        return (preds[:, :4].double() - targets[:, :4].double()).abs().sum(dim=-1)
