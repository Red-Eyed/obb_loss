"""Sigmoid target-class assignment cost; DETR: https://arxiv.org/abs/2005.12872.

This variant uses sigmoid instead of DETR's softmax class probability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from obb_loss.result import Ok, Result

if TYPE_CHECKING:
    from torch import Tensor

    from obb_loss.packed import InputError
    from obb_loss.set_context import SetCostContext


def set_classification_cost(
    context: SetCostContext,
) -> Result[Tensor[[int]], InputError]:
    """Compute negative target-class probability for each candidate pair.

    Args:
        context: Validated sets and same-sample global pair indices; target quality
            does not affect this cost.

    Returns:
        Ok(float64 [pairs] values in [-1, 0]); confident correct classes cost less.
    """
    classes = context.targets.labels[context.target_indices]
    scores = context.preds.logits[context.pred_indices, classes].double()
    return Ok(-scores.sigmoid())
