"""Standalone rectangular minimum-cost assignment for packed geometry.

Assignment formulation: Kuhn (1955), The Hungarian Method for the Assignment Problem,
https://doi.org/10.1002/nav.3800020109. SciPy uses modified Jonker-Volgenant, described
by Crouse (2016), https://doi.org/10.1109/TAES.2016.140952; see docs/matching.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch
from torch import Tensor, nn
from typing_extensions import override

from obb_loss.config import MatcherConfig
from obb_loss.losses.stable_probiou import stable_probiou_loss
from obb_loss.packed import InputError, PackedBoxes
from obb_loss.pairwise import pairwise_costs
from obb_loss.result import Err, Ok, Result
from obb_loss.solver import solve_costs

if TYPE_CHECKING:
    from obb_loss.protocols import BoxFunction

_DEFAULT_CONFIG = MatcherConfig()


class Assignment(NamedTuple):
    """Global paired indices and image offsets, including empty assignments."""

    pred_indices: Tensor
    target_indices: Tensor
    cu_seqlens: Tensor


class HungarianMatcher(nn.Module):
    """Minimize per-image cost without detector classification/layer orchestration."""

    def __init__(
        self,
        config: MatcherConfig = _DEFAULT_CONFIG,
        cost: BoxFunction = stable_probiou_loss,
    ) -> None:
        """Store explicit storage limits and an unreduced aligned cost function."""
        super().__init__()
        self.config = config
        self.cost = cost

    @override
    def forward(
        self, preds: PackedBoxes, targets: PackedBoxes
    ) -> Result[Assignment, InputError]:
        """Return checked detached indices; smaller side is fully assigned per image.

        CPU solver receives one explicit detached cost transfer. Output indices
        live on the input device. Matching is nondifferentiable, calls no loss
        criterion, and does not generate quality targets or orchestrate layers.
        """
        with torch.no_grad():
            match pairwise_costs(preds, targets, self.config, self.cost):
                case Err() as failure:
                    return failure
                case Ok(value=costs):
                    return solve_costs(costs)


def gather_matched(
    preds: PackedBoxes, targets: PackedBoxes, assignment: Assignment
) -> tuple[PackedBoxes, PackedBoxes]:
    """Gather matched geometry with gradients using trusted same-batch assignment.

    Assignment must come from these input records; indices are borrowed. Gather
    allocates output geometry and preserves per-image boundaries, including empties.
    """
    return PackedBoxes(
        preds.boxes[assignment.pred_indices], assignment.cu_seqlens
    ), PackedBoxes(targets.boxes[assignment.target_indices], assignment.cu_seqlens)
