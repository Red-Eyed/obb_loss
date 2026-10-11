"""Per-sample assignment costs for class logits, box L1, and ProbIoU.

DETR cost weighting: Carion et al., ECCV 2020, https://arxiv.org/abs/2005.12872.
This oriented sigmoid-class variant uses negative target-class probability,
center/side L1, and ProbIoU loss instead of softmax probability and GIoU.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from typing_extensions import override

from obb_loss.config import SetMatcherConfig
from obb_loss.losses.l1 import box_l1_loss
from obb_loss.losses.stable_probiou import stable_probiou_loss, validate_pairs
from obb_loss.pairwise import PackedCosts, cost_blocks, validate_output
from obb_loss.result import Err, Ok, Result
from obb_loss.set_data import PackedPredictions, PackedTargets, validate_sets
from obb_loss.solver import solve_costs

if TYPE_CHECKING:
    from obb_loss.matching import Assignment
    from obb_loss.packed import InputError
    from obb_loss.pairwise import CostBlock

_DEFAULT_CONFIG = SetMatcherConfig()


def _chunk_cost(
    preds: PackedPredictions,
    targets: PackedTargets,
    pred_indices: Tensor,
    target_indices: Tensor,
    config: SetMatcherConfig,
) -> Result[Tensor, InputError]:
    """Evaluate enabled costs for bounded pairs, without using target quality."""
    values = torch.zeros(
        pred_indices.shape, dtype=torch.float64, device=pred_indices.device
    )
    weights = config.weights
    if weights.classification > 0:
        classes = targets.labels[target_indices]
        scores = preds.logits[pred_indices, classes].double().sigmoid()
        values = values - weights.classification * scores
    if weights.l1 > 0 or weights.probiou > 0:
        first = preds.geometry.boxes[pred_indices]
        second = targets.geometry.boxes[target_indices]
        if weights.l1 > 0:
            values = values + weights.l1 * box_l1_loss(first, second)
        if weights.probiou > 0:
            match validate_pairs(first, second):
                case Err() as failure:
                    return failure
                case Ok():
                    values = values + weights.probiou * stable_probiou_loss(
                        first, second
                    )
    return validate_output(values, pred_indices.shape[0], pred_indices.device)


def _fill_cost_block(
    values: Tensor,
    block: CostBlock,
    preds: PackedPredictions,
    targets: PackedTargets,
    config: SetMatcherConfig,
) -> Result[Tensor, InputError]:
    """Write one rectangular cost block in bounded row-major chunks."""
    for start in range(block.start, block.end, config.storage.chunk_size):
        end = min(start + config.storage.chunk_size, block.end)
        index = torch.arange(
            start - block.start, end - block.start, device=values.device
        )
        pred_indices = block.pred_start + index // block.columns
        target_indices = block.target_start + index % block.columns
        match _chunk_cost(preds, targets, pred_indices, target_indices, config):
            case Err() as failure:
                return failure
            case Ok(value=cost):
                values[start:end] = cost
    return Ok(values)


def set_pairwise_costs(
    preds: PackedPredictions,
    targets: PackedTargets,
    config: SetMatcherConfig = _DEFAULT_CONFIG,
) -> Result[PackedCosts, InputError]:
    """Build classification, L1, and ProbIoU cost blocks within each sample.

    Costs use negative sigmoid target-class probability, center/side L1, and ProbIoU loss. Disabled costs skip evaluation. Direct calls retain gradients.

    Args:
        preds: Packed geometry and [queries, classes] logits for one decoder layer,
            sharing box order and device.
        targets: Packed reference geometry, int64 class labels, and caller-supplied
            [targets] quality in [0, 1]; labels and quality follow box order.
            Quality is validated but does not affect matching costs.
        config: Independent matching-cost weights and temporary/full storage limits;
            unrelated to training-loss weights.

    Returns:
        Ok(PackedCosts) with float64 row-major values or Err(InputError) for invalid
        records, label bounds, enabled numerical domains, costs, or storage limits.
    """
    match validate_sets(preds, targets):
        case Err() as failure:
            return failure
        case Ok():
            pass
    match cost_blocks(preds.geometry, targets.geometry, config.storage.max_pairs):
        case Err() as failure:
            return failure
        case Ok(value=blocks):
            pass
    total = blocks[-1].end if blocks else 0
    with torch.autocast(preds.logits.device.type, enabled=False):
        values = torch.empty(total, dtype=torch.float64, device=preds.logits.device)
        for block in blocks:
            match _fill_cost_block(values, block, preds, targets, config):
                case Err() as failure:
                    return failure
                case Ok():
                    pass
    return Ok(PackedCosts(values, blocks))


class SetHungarianMatcher(nn.Module):
    """Assign queries to labeled targets with independent weighted set costs."""

    def __init__(self, config: SetMatcherConfig = _DEFAULT_CONFIG) -> None:
        """Configure classification-aware set assignment.

        Args:
            config: Independent class/L1/ProbIoU matching weights and chunk/full-cost
                storage limits.
        """
        super().__init__()
        self.config = config

    @override
    def forward(
        self, preds: PackedPredictions, targets: PackedTargets
    ) -> Result[Assignment, InputError]:
        """Match queries and labeled targets independently within each sample.

        The smaller side is fully paired. Matching disables autograd and transfers costs to the shared CPU solver; indices return to the input device.

        Args:
            preds: Packed geometry and [queries, classes] logits for one decoder layer,
                sharing box order and device.
            targets: Packed reference geometry, int64 class labels, and caller-supplied
                [targets] quality in [0, 1]; labels and quality follow box order.
                Quality never changes assignment.

        Returns:
            Ok(Assignment) with detached paired global indices and repeated offsets for
            empty samples, or Err(InputError) for invalid records, costs, enabled
            domains, or storage limits.
        """
        with torch.no_grad():
            match set_pairwise_costs(preds, targets, self.config):
                case Err() as failure:
                    return failure
                case Ok(value=costs):
                    return solve_costs(costs)
