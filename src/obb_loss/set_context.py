"""Borrowed callback inputs for per-sample losses and bounded matching pairs."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch
from torch import Tensor

from obb_loss.reduction import sum_samples

if TYPE_CHECKING:
    from obb_loss.matching import Assignment
    from obb_loss.packed import PackedBoxes
    from obb_loss.set_data import PackedPredictions, PackedTargets


class SetLossContext(NamedTuple):
    """Borrow validated sets and their assignment for one criterion evaluation.

    Attributes:
        preds: All packed queries, including unmatched queries available to regularizers.
        targets: Packed references and caller-supplied class/quality supervision.
        assignment: Detached global pair indices and per-sample pair offsets.
        matched_preds: Matched query geometry, gathered once with gradients preserved.
        matched_targets: Matched reference geometry in assignment order.
    """

    preds: PackedPredictions
    targets: PackedTargets
    assignment: Assignment
    matched_preds: PackedBoxes
    matched_targets: PackedBoxes

    def zero(self) -> Tensor[[int]]:
        """Return float64 [samples] zeros connected to query geometry and logits."""
        values = self.preds.geometry.boxes[:, 0].double() * 0
        values = values + (self.preds.logits.double() * 0).sum(-1)
        return sum_samples(values, self.preds.geometry.cu_seqlens)


class SetCostContext(NamedTuple):
    """Borrow validated sets and bounded candidate pairs within one sample.

    Attributes:
        preds: Packed queries whose geometry/logits remain on their original device.
        targets: Packed reference geometry, class labels, and caller quality.
        pred_indices: Global query indices for this aligned cost chunk.
        target_indices: Global reference indices paired with pred_indices.
    """

    preds: PackedPredictions
    targets: PackedTargets
    pred_indices: Tensor[[int]]
    target_indices: Tensor[[int]]

    def zero(self) -> Tensor[[int]]:
        """Return float64 [pairs] zeros connected to candidate query inputs."""
        classes = self.targets.labels[self.target_indices]
        return (
            self.preds.geometry.boxes[self.pred_indices, 0].double() * 0
            + self.preds.logits[self.pred_indices, classes].double() * 0
        )


def matched_quality(context: SetLossContext) -> Tensor[[int, int]]:
    """Scatter caller quality to matched query/classes, retaining target gradients.

    Args:
        context: Validated sets and assignment; every unmatched query/class is negative.

    Returns:
        Float64 [queries, classes] quality, with zero outside matched labeled classes.
    """
    quality = torch.zeros_like(context.preds.logits, dtype=torch.float64)
    classes = context.targets.labels[context.assignment.target_indices]
    quality[context.assignment.pred_indices, classes] = context.targets.quality[
        context.assignment.target_indices
    ].double()
    return quality
