"""Per-sample Hungarian assignment from composed bounded-pair callbacks.

Set assignment: Carion et al., ECCV 2020, https://arxiv.org/abs/2005.12872.
The default callbacks use sigmoid target-class probability, L1, and ProbIoU.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from typing_extensions import override

from obb_loss.composition import DefaultComposition, LossComposition, LossFunction
from obb_loss.config import MatcherConfig
from obb_loss.packed import ErrorKind, InputError
from obb_loss.pairwise import PackedCosts, cost_blocks, validate_output
from obb_loss.result import Err, Ok, Result
from obb_loss.set_context import SetCostContext
from obb_loss.set_data import PackedPredictions, PackedTargets, validate_sets
from obb_loss.set_defaults import default_set_costs
from obb_loss.solver import solve_costs

if TYPE_CHECKING:
    from collections.abc import Mapping

    from obb_loss.matching import Assignment
    from obb_loss.pairwise import CostBlock

_DEFAULT_CONFIG = MatcherConfig()


def _fill_cost_block(
    values: Tensor,
    block: CostBlock,
    preds: PackedPredictions,
    targets: PackedTargets,
    config: MatcherConfig,
    callbacks: LossComposition[SetCostContext],
) -> Result[Tensor, InputError]:
    """Evaluate same-sample aligned candidate chunks without cross-sample pairs."""
    for start in range(block.start, block.end, config.chunk_size):
        end = min(start + config.chunk_size, block.end)
        index = torch.arange(
            start - block.start, end - block.start, device=values.device
        )
        context = SetCostContext(
            preds,
            targets,
            block.pred_start + index // block.columns,
            block.target_start + index % block.columns,
        )
        match callbacks(context):
            case Err() as failure:
                return failure
            case Ok(value=outputs):
                pass
        total = context.zero()
        for output in outputs:
            total = total + output.values
        match validate_output(total, end - start, values.device):
            case Err() as failure:
                return failure
            case Ok():
                values[start:end] = total
    return Ok(values)


def _pairwise(
    preds: PackedPredictions,
    targets: PackedTargets,
    config: MatcherConfig,
    callbacks: LossComposition[SetCostContext],
) -> Result[PackedCosts, InputError]:
    """Validate sets and construct packed costs using an already registered composition."""
    if not callbacks.terms:
        return Err(
            InputError(ErrorKind.CONFIG, "costs", "at least one enabled cost required")
        )
    match validate_sets(preds, targets):
        case Err() as failure:
            return failure
        case Ok():
            pass
    match cost_blocks(preds.geometry, targets.geometry, config.max_pairs):
        case Err() as failure:
            return failure
        case Ok(value=blocks):
            pass
    total = blocks[-1].end if blocks else 0
    with torch.autocast(preds.logits.device.type, enabled=False):
        values = torch.empty(total, dtype=torch.float64, device=preds.logits.device)
        for block in blocks:
            match _fill_cost_block(values, block, preds, targets, config, callbacks):
                case Err() as failure:
                    return failure
                case Ok():
                    pass
    return Ok(PackedCosts(values, blocks))


def set_pairwise_costs(
    preds: PackedPredictions,
    targets: PackedTargets,
    config: MatcherConfig = _DEFAULT_CONFIG,
    *,
    costs: Mapping[str, LossFunction[SetCostContext]]
    | DefaultComposition = DefaultComposition.BUILTINS,
) -> Result[PackedCosts, InputError]:
    """Build composed candidate costs independently within each sample.

    Args:
        preds: Packed geometry and [queries, classes] logits in query order.
        targets: Same-sample-count/device references, labels, and caller quality.
            Default costs ignore quality; custom callbacks may explicitly use it.
        config: Chunk size and complete packed cost storage limits.
        costs: Named callbacks returning weighted [pairs] values through Result.
            Omission selects default sigmoid-class/L1/ProbIoU costs. Lower total
            cost is preferable; no normalization or reduction is applied.

    Returns:
        Ok(float64 row-major PackedCosts), retaining direct-call gradients, or a
        typed error for empty composition, invalid inputs, callback outputs,
        numerical domains, or storage limits.
    """
    functions = default_set_costs() if isinstance(costs, DefaultComposition) else costs
    return _pairwise(preds, targets, config, LossComposition(functions))


class SetHungarianMatcher(nn.Module):
    """Own cost callbacks and assign packed queries to references within each sample."""

    def __init__(
        self,
        config: MatcherConfig = _DEFAULT_CONFIG,
        *,
        costs: Mapping[str, LossFunction[SetCostContext]]
        | DefaultComposition = DefaultComposition.BUILTINS,
    ) -> None:
        """Register a cost composition with independent memory limits.

        Args:
            config: Positive chunk size and maximum complete candidate-pair count.
            costs: Named weighted callbacks over bounded pairs; functions and typed
                modules are supported. Omission selects the default costs. Explicit
                zero-weight WeightedLoss callbacks are omitted, including their state.
        """
        super().__init__()
        self.config = config
        functions = (
            default_set_costs() if isinstance(costs, DefaultComposition) else costs
        )
        self.costs = LossComposition(functions)

    @override
    # Native Torch stubs declare variadic forward; this packed matcher narrows it.
    def forward(  # pyrefly: ignore[bad-override]
        self, preds: PackedPredictions, targets: PackedTargets
    ) -> Result[Assignment, InputError]:
        """Match queries and references independently, with autograd disabled.

        Args:
            preds: Packed query geometry/logits on a shared CPU/CUDA device.
            targets: Packed references and labels sharing sample count and coordinate
                units with preds; quality is available only to explicit custom costs.

        Returns:
            Ok(detached global pair indices and sample offsets) or a typed cost/input
            error. The smaller side is fully paired. Costs transfer to the shared CPU
            solver, and assignment indices return to the input device.
        """
        with torch.no_grad():
            match _pairwise(preds, targets, self.config, self.costs):
                case Err() as failure:
                    return failure
                case Ok(value=costs):
                    return solve_costs(costs)
