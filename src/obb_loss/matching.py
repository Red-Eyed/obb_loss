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
    """Describe paired global indices and sample boundaries.

    Matcher results are detached. Indices are global within their respective packed records.

    Attributes:
        pred_indices: Same-device int64 [pairs] indices into the original prediction
            geometry.
        target_indices: Same-device int64 [pairs] reference indices paired with
            pred_indices.
        cu_seqlens: Int64 [batch + 1] cumulative pair counts; repeated offsets
            preserve samples with no pairs.
    """

    pred_indices: Tensor
    target_indices: Tensor
    cu_seqlens: Tensor


class HungarianMatcher(nn.Module):
    """Minimize per-sample cost without detector classification/layer orchestration."""

    def __init__(
        self,
        config: MatcherConfig = _DEFAULT_CONFIG,
        cost: BoxFunction = stable_probiou_loss,
    ) -> None:
        """Configure geometry-only assignment and its cost function.

        Args:
            config: Validated chunk-size and total-pair storage limits for cost
                construction.
            cost: Pure callable accepting two aligned [pairs, 6] tensors and returning
                finite same-device floating [pairs] costs; smaller values are preferred.
        """
        super().__init__()
        self.config = config
        self.cost = cost

    @override
    # Torch stubs declare variadic forward; this module intentionally narrows it.
    def forward(  # pyrefly: ignore[bad-override]
        self, preds: PackedBoxes, targets: PackedBoxes
    ) -> Result[Assignment, InputError]:
        """Find independent minimum-cost assignments within each sample.

        The smaller side is fully paired. Costs transfer once to the CPU solver; returned indices use the input device. Custom callable bugs propagate.

        Args:
            preds: Packed candidate geometry; query counts may differ by sample.
            targets: Packed reference geometry in the same coordinate units, with the
                same sample count and device. Counts may differ from preds.

        Returns:
            Ok(Assignment) with detached global paired indices and sample offsets, or
            Err(InputError) for invalid records, costs, domain, or storage limits.
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
    """Gather paired packed geometry while preserving gradients.

    This trusted helper does not validate arbitrary indices or assignments.

    Args:
        preds: The exact packed predictions used to obtain assignment.
        targets: The exact packed references used to obtain assignment, in the same
            units.
        assignment: Valid global paired indices and sample offsets produced for
            these records.

    Returns:
        Prediction and target PackedBoxes in paired order, sharing assignment
        offsets; geometry is newly gathered and differentiable.
    """
    return PackedBoxes(
        preds.boxes[assignment.pred_indices], assignment.cu_seqlens
    ), PackedBoxes(targets.boxes[assignment.target_indices], assignment.cu_seqlens)
