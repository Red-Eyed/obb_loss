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
    """Sum absolute center/side errors per pair in float64, excluding direction.

    Inputs must be aligned six-coordinate boxes on one CPU/CUDA device, using
    the same coordinate units. Orientation is supervised by a separate overlap loss.
    Autocast is disabled locally; gradients remain connected to both inputs.
    """
    with torch.autocast(preds.device.type, enabled=False):
        return (preds[:, :4].double() - targets[:, :4].double()).abs().sum(dim=-1)
