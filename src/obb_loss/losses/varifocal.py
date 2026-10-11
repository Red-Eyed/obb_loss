"""Varifocal loss: Zhang et al., CVPR 2021, https://arxiv.org/abs/2008.13367."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing_extensions import override

from obb_loss.classification import validate_quality
from obb_loss.config import VFLConfig, VFLParameters
from obb_loss.reduction import reduce_loss, sum_samples
from obb_loss.result import Err, Ok, Result
from obb_loss.set_context import matched_quality

if TYPE_CHECKING:
    from shape_extensions import IntVar

    N = IntVar("N")
    C = IntVar("C")

    from obb_loss.packed import InputError
    from obb_loss.set_context import SetLossContext

_DEFAULT_CONFIG = VFLConfig()
_DEFAULT_PARAMETERS = VFLParameters()


def set_varifocal_loss(
    context: SetLossContext, *, config: VFLParameters = _DEFAULT_PARAMETERS
) -> Result[Tensor[[int]], InputError]:
    """Sum Varifocal supervision over all query/classes within each sample.

    Args:
        context: Validated sets and assignment; caller quality supervises matched
            labeled classes, and every other query/class is negative.
        config: Immutable modulation parameters, independent of weighting/reduction.

    Returns:
        Ok(float64 [samples] raw sums); caller quality gradients are preserved.
        Weighting, target-count normalization, and reduction belong to composition.
    """
    values = varifocal_loss(
        context.preds.logits,
        matched_quality(context),
        alpha=config.alpha,
        gamma=config.gamma,
        quality_weighted=config.quality_weighted,
    )
    return Ok(sum_samples(values.sum(-1), context.preds.geometry.cu_seqlens))


def varifocal_loss(
    logits: Tensor[[N, C]],
    quality: Tensor[[N, C]],
    *,
    alpha: float = 0.75,
    gamma: float = 2.0,
    quality_weighted: bool = True,
) -> Tensor[[N, C]]:
    """Compute unreduced Varifocal loss with supplied quality targets.

    Inputs must share a CPU/CUDA device and be finite. Autocast is disabled locally. The logsigmoid form keeps fractional negative modulation finite at probability underflow.

    Args:
        logits: Floating [queries, classes] scores before sigmoid; no background
            column.
        quality: Floating [queries, classes] quality in [0, 1], aligned with logits;
            zero marks a negative class. Target gradients are preserved.
        alpha: Finite nonnegative multiplier for negative-class BCE weights.
        gamma: Finite nonnegative exponent on predicted probability for negative
            classes; larger values suppress easy negatives.
        quality_weighted: Whether positive-class BCE is weighted by its quality
            target; false uses unit positive weight.

    Returns:
        Float64 [queries, classes] losses, without reduction or a global loss
        weight.
    """
    with torch.autocast(logits.device.type, enabled=False):
        scores, targets = logits.double(), quality.double()
        negative_weight = alpha * (F.logsigmoid(scores) * gamma).exp()
        positive_weight = targets if quality_weighted else torch.ones_like(targets)
        weight = torch.where(targets > 0, positive_weight, negative_weight)
        return (
            F.binary_cross_entropy_with_logits(scores, targets, reduction="none")
            * weight
        )


class VarifocalLoss(nn.Module):
    """Configured VFL without detector assignment or layer orchestration."""

    def __init__(self, config: VFLConfig = _DEFAULT_CONFIG) -> None:
        """Configure the classification loss.

        Args:
            config: Validated immutable settings selecting VFL alpha, gamma, positive-
                quality weighting, output reduction, and global weight.
        """
        super().__init__()
        self.config = config

    @override
    # Torch stubs declare variadic forward; this module intentionally narrows it.
    def forward(self, logits: Tensor, quality: Tensor) -> Tensor:  # pyrefly: ignore[bad-override]
        """Evaluate classification loss on trusted quality targets.

        Inputs must share a CPU/CUDA device and satisfy the pure function domain. A zero configured weight skips the loss computation. Use checked for recoverable validation failures.

        Args:
            logits: Floating [queries, classes] scores before sigmoid; no background
                column.
            quality: Floating [queries, classes] quality in [0, 1], aligned with logits;
                zero marks a negative class. Target gradients are preserved.

        Returns:
            Weighted float64 [queries, classes] values for NONE, or a scalar for
            SUM/MEAN; empty scalar reductions are connected zeros.
        """
        if self.config.weight == 0:
            return reduce_loss(
                logits.double() * 0 + quality.double() * 0, self.config.reduction
            )
        values = varifocal_loss(
            logits,
            quality,
            alpha=self.config.alpha,
            gamma=self.config.gamma,
            quality_weighted=self.config.quality_weighted,
        )
        return reduce_loss(values, self.config.reduction) * self.config.weight

    def checked(self, logits: Tensor, quality: Tensor) -> Result[Tensor, InputError]:
        """Validate logits and supplied quality before evaluating classification loss.

        Validation checks current tensor contents and synchronizes accelerators.

        Args:
            logits: Floating [queries, classes] scores before sigmoid; no background
                column.
            quality: Floating [queries, classes] quality in [0, 1], aligned with logits;
                zero marks a negative class. Target gradients are preserved.

        Returns:
            Ok(loss) using configured weighting/reduction, or Err(InputError) for shape,
            dtype, device, finiteness, or quality-range violations.
        """
        match validate_quality(logits, quality):
            case Err() as failure:
                return failure
            case Ok():
                return Ok(self.forward(logits, quality))
