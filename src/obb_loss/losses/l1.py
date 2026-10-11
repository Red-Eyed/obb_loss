"""DETR box L1: Carion et al., ECCV 2020, https://arxiv.org/abs/2005.12872."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from obb_loss.reduction import sum_samples
from obb_loss.result import Ok, Result

if TYPE_CHECKING:
    from shape_extensions import IntVar
    from torch import Tensor

    from obb_loss.packed import InputError
    from obb_loss.set_context import SetCostContext, SetLossContext

    N = IntVar("N")


def box_l1_loss(
    preds: Tensor[[N, 6]],
    targets: Tensor[[N, 6]],
) -> Tensor[[N]]:
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


def set_l1_loss(context: SetLossContext) -> Result[Tensor[[int]], InputError]:
    """Sum matched center/side L1 errors within each sample.

    Args:
        context: Validated sets and gathered pairs; direction is excluded and shared
            coordinate units determine the error scale.

    Returns:
        Ok(float64 [samples] raw sums), including connected zeros for empty samples.
        Weighting, normalization, and reduction are not applied.
    """
    values = box_l1_loss(context.matched_preds.boxes, context.matched_targets.boxes)
    return Ok(sum_samples(values, context.assignment.cu_seqlens))


def set_l1_cost(context: SetCostContext) -> Result[Tensor[[int]], InputError]:
    """Compute center/side L1 for each bounded candidate pair.

    Args:
        context: Validated sets and aligned global query/reference indices in shared units.

    Returns:
        Ok(float64 [pairs] unweighted L1 values), excluding direction coordinates.
    """
    return Ok(
        box_l1_loss(
            context.preds.geometry.boxes[context.pred_indices],
            context.targets.geometry.boxes[context.target_indices],
        )
    )
