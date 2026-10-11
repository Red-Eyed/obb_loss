"""Callback composition for one layer of packed set supervision.

Set supervision: Carion et al., ECCV 2020, https://arxiv.org/abs/2005.12872.
The caller chooses loss/cost callbacks and aggregates decoder layers.
"""

from __future__ import annotations

import math
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, NamedTuple

import torch
from torch import Tensor, nn
from typing_extensions import override

from obb_loss.composition import DefaultComposition, LossComposition, LossFunction
from obb_loss.config import Reduction
from obb_loss.matching import gather_matched
from obb_loss.packed import ErrorKind, InputError
from obb_loss.reduction import reduce_loss
from obb_loss.result import Err, Ok, Result
from obb_loss.set_context import SetLossContext
from obb_loss.set_defaults import default_set_losses
from obb_loss.set_matching import SetHungarianMatcher

if TYPE_CHECKING:
    from collections.abc import Mapping

    from obb_loss.matching import Assignment
    from obb_loss.set_data import PackedPredictions, PackedTargets


class TargetCount(Enum):
    """Normalize by each sample's target count or the local batch target count."""

    PER_SAMPLE = "per_sample"
    PER_BATCH = "per_batch"


class SetLosses(NamedTuple):
    """Return normalized weighted callbacks and their total.

    Attributes:
        total: Sum of normalized weighted callback values; [samples] for NONE,
            otherwise a scalar. Empty compositions return connected zeros.
        terms: Read-only mapping of caller names to normalized weighted values,
            using the same reduction as total. Disabled callbacks are omitted.
        assignment: Detached global pairs and cumulative per-sample pair counts.
    """

    total: Tensor
    terms: Mapping[str, Tensor]
    assignment: Assignment


def _sample_divisor(
    targets: PackedTargets, normalizer: float | TargetCount
) -> Result[Tensor, InputError]:
    """Resolve explicit normalization to a positive divisor per sample."""
    offsets = targets.geometry.cu_seqlens
    counts = (offsets[1:] - offsets[:-1]).double()
    match normalizer:
        case TargetCount.PER_SAMPLE:
            return Ok(counts.clamp_min(1))
        case TargetCount.PER_BATCH:
            return Ok(torch.full_like(counts, max(targets.geometry.boxes.shape[0], 1)))
        case float() | int():
            if math.isfinite(normalizer) and normalizer > 0:
                return Ok(torch.full_like(counts, normalizer))
    return Err(
        InputError(
            ErrorKind.DOMAIN,
            "normalizer",
            "target-count policy or finite positive scalar required",
        )
    )


class SetCriterion(nn.Module):
    """Match once, compose named per-sample callbacks, then normalize and reduce."""

    def __init__(
        self,
        *,
        matcher: SetHungarianMatcher | DefaultComposition = DefaultComposition.BUILTINS,
        losses: Mapping[str, LossFunction[SetLossContext]]
        | DefaultComposition = DefaultComposition.BUILTINS,
    ) -> None:
        """Register the matcher and independent loss composition.

        Args:
            matcher: Configured Hungarian matcher owning its cost callbacks and
                storage limits; omission constructs the default cost composition.
            losses: Named callbacks returning unnormalized per-sample sums through
                Result. Functions and typed modules are supported; weighting may be
                owned by a callback or WeightedLoss. Omission selects default VFL/L1/
                ProbIoU callbacks. An empty mapping disables all training terms.
        """
        super().__init__()
        self.matcher = (
            SetHungarianMatcher()
            if isinstance(matcher, DefaultComposition)
            else matcher
        )
        functions = (
            default_set_losses() if isinstance(losses, DefaultComposition) else losses
        )
        self.losses = LossComposition(functions)

    @override
    # Native Torch stubs declare variadic forward; the packed criterion narrows it.
    def forward(  # pyrefly: ignore[bad-override]
        self,
        preds: PackedPredictions,
        targets: PackedTargets,
        *,
        reduction: Reduction = Reduction.SUM,
        normalizer: float | TargetCount = TargetCount.PER_BATCH,
    ) -> Result[SetLosses, InputError]:
        """Match one layer and evaluate normalized named callback losses.

        Inputs are borrowed, never mutated or detached. Matching runs without
        autograd; callback evaluation disables autocast and preserves gradients.
        Normalization and reduction happen once, after each callback's weighting.

        Args:
            preds: Packed query geometry and [queries, classes] logits in shared
                reference coordinate units; all queries are visible to callbacks.
            targets: Same-sample-count/device packed references, class labels, and
                caller-supplied quality in [0, 1]. Quality is never synthesized.
            reduction: NONE retains [samples]; SUM adds that vector; MEAN averages
                all samples, including empty ones. Empty scalar reductions return zero.
            normalizer: PER_SAMPLE uses each sample's target count; PER_BATCH uses
                the local batch target count. Counts are clamped to one. A finite
                positive scalar supplies an external shared divisor; 1 keeps raw sums.

        Returns:
            Ok(SetLosses) with normalized weighted terms, their total, and assignment;
            Err(InputError) for invalid records, normalization, reduction, matching,
            callback outputs, numerical domains, or nonfinite totals. Callback bugs
            propagate; custom callbacks must return Result without mutating inputs.
        """
        if not isinstance(reduction, Reduction):
            return Err(
                InputError(ErrorKind.CONFIG, "reduction", "expected Reduction enum")
            )
        match self.matcher(preds, targets):
            case Err() as failure:
                return failure
            case Ok(value=assignment):
                pass
        match _sample_divisor(targets, normalizer):
            case Err() as failure:
                return failure
            case Ok(value=divisor):
                pass
        first, second = gather_matched(preds.geometry, targets.geometry, assignment)
        context = SetLossContext(preds, targets, assignment, first, second)
        with torch.autocast(preds.logits.device.type, enabled=False):
            match self.losses(context):
                case Err() as failure:
                    return failure
                case Ok(value=outputs):
                    pass
            total = context.zero()
            terms: dict[str, Tensor] = {}
            for output in outputs:
                total = total + output.values
                terms[output.name] = reduce_loss(output.values / divisor, reduction)
            total = reduce_loss(total / divisor, reduction)
            if not bool(torch.isfinite(total).all()) or any(
                not bool(torch.isfinite(value).all()) for value in terms.values()
            ):
                return Err(
                    InputError(
                        ErrorKind.DOMAIN, "loss", "nonfinite normalized term or total"
                    )
                )
            return Ok(SetLosses(total, MappingProxyType(terms), assignment))
