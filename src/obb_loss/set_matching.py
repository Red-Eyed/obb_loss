"""Bounded packed Hungarian matching; DETR: https://arxiv.org/abs/2005.12872."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from typing_extensions import override

from obb_loss.config import MatcherConfig
from obb_loss.losses.l1 import box_l1_loss
from obb_loss.losses.stable_probiou import stable_probiou_loss, validate_pair_domain
from obb_loss.pairwise import PackedCosts, cost_blocks, validate_output
from obb_loss.result import Err, Ok, Result
from obb_loss.set_data import validate_sets
from obb_loss.solver import solve_costs

if TYPE_CHECKING:
    from obb_loss.matching import Assignment
    from obb_loss.packed import InputError
    from obb_loss.set_data import Predictions, Targets

_DEFAULT_CONFIG = MatcherConfig()


class SetHungarianMatcher(nn.Module):
    """Match each sample independently; override cost to change the matching formula."""

    def __init__(self, config: MatcherConfig = _DEFAULT_CONFIG) -> None:
        """Select matching multipliers and bounded cost storage.

        Args:
            config: Immutable matching weights, chunk size and total-pair limit.
        """
        super().__init__()
        self.config = config

    def cost(
        self,
        preds: Predictions,
        targets: Targets,
        pred_indices: Tensor,
        target_indices: Tensor,
    ) -> Result[Tensor, InputError]:
        """Calculate negative sigmoid-class/L1/ProbIoU costs for bounded candidate pairs.

        Args:
            preds: Admitted packed prediction geometry and logits.
            targets: Admitted references and class labels in the same units/device.
            pred_indices: Global query indices for the current same-sample chunk.
            target_indices: Global reference indices paired with pred_indices.

        Returns:
            Ok(float64 [pairs] costs), lower meaning preferable, or a typed numerical
            error. Zero multipliers skip their terms. Override for custom matching.
        """
        first, second = (
            preds.boxes.values[pred_indices],
            targets.boxes.values[target_indices],
        )
        values = first[:, 0].double() * 0
        if self.config.classification_weight > 0:
            labels = targets.labels[target_indices]
            values = (
                values
                - self.config.classification_weight
                * preds.logits[pred_indices, labels].double().sigmoid()
            )
        if self.config.l1_weight > 0:
            values = values + self.config.l1_weight * box_l1_loss(first, second)
        if self.config.probiou_weight > 0:
            match validate_pair_domain(first, second):
                case Err() as failure:
                    return failure
                case Ok():
                    values = values + self.config.probiou_weight * stable_probiou_loss(
                        first, second
                    )
        return Ok(values)

    def pairwise_costs(
        self, preds: Predictions, targets: Targets
    ) -> Result[PackedCosts, InputError]:
        """Build per-sample rectangular costs without solving assignment.

        Args:
            preds: Packed queries with logits; direct calls retain gradients.
            targets: Packed reference boxes and labels sharing sample count/device.

        Returns:
            Ok(float64 packed costs) or a typed input/domain/output/storage failure.
            Complete storage is sum(query_count * target_count) across samples;
            temporary pairs are bounded by config.chunk_size.
        """
        match validate_sets(preds, targets):
            case Err() as failure:
                return failure
            case Ok():
                pass
        match cost_blocks(preds.boxes, targets.boxes, self.config.max_pairs):
            case Err() as failure:
                return failure
            case Ok(value=blocks):
                pass
        total = blocks[-1].end if blocks else 0
        with torch.autocast(preds.boxes.values.device.type, enabled=False):
            values = torch.empty(
                total, dtype=torch.float64, device=preds.boxes.values.device
            )
            for block in blocks:
                for start in range(block.start, block.end, self.config.chunk_size):
                    end = min(start + self.config.chunk_size, block.end)
                    index = torch.arange(
                        start - block.start, end - block.start, device=values.device
                    )
                    match self.cost(
                        preds,
                        targets,
                        block.pred_start + index // block.columns,
                        block.target_start + index % block.columns,
                    ):
                        case Err() as failure:
                            return failure
                        case Ok(value=cost):
                            pass
                    match validate_output(cost, end - start, values.device):
                        case Err() as failure:
                            return failure
                        case Ok(value=cost):
                            values[start:end] = cost
        return Ok(PackedCosts(values, blocks))

    @override
    # Native Torch stubs declare variadic forward; packed tensors narrow it.
    def forward(  # pyrefly: ignore[bad-override]
        self, preds: Predictions, targets: Targets
    ) -> Result[Assignment, InputError]:
        """Find detached assignments with explicit cost transfer to the CPU solver.

        Args:
            preds: Packed prediction geometry and logits.
            targets: Packed reference geometry and labels, sharing units/device.

        Returns:
            Ok(global pair indices and sample offsets) or a typed cost/input failure.
            The smaller side is fully paired; indices return to the input device.
        """
        with torch.no_grad():
            match self.pairwise_costs(preds, targets):
                case Err() as failure:
                    return failure
                case Ok(value=costs):
                    return solve_costs(costs)
