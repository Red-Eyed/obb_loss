"""Varifocal loss: Zhang et al., CVPR 2021, https://arxiv.org/abs/2008.13367."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing_extensions import override

from obb_loss.classification import validate_quality
from obb_loss.config import VFLConfig
from obb_loss.reduction import reduce_loss
from obb_loss.result import Err, Ok, Result

if TYPE_CHECKING:
    from jaxtyping import Float

    from obb_loss.packed import InputError

_DEFAULT_CONFIG = VFLConfig()


def varifocal_loss(
    logits: Float[Tensor, "n c"],  # noqa: F722
    quality: Float[Tensor, "n c"],  # noqa: F722
    *,
    alpha: float = 0.75,
    gamma: float = 2.0,
    quality_weighted: bool = True,
) -> Float[Tensor, "n c"]:  # noqa: F722
    """Return trusted unreduced float64 VFL with explicit caller-supplied quality.

    Equal shapes, finite logits, quality in [0,1], same CPU/CUDA device, and finite
    nonnegative alpha/gamma are required. No mutation/detach; target gradients flow.
    Positive BCE weight is quality (or one); negatives use alpha*p**gamma. The
    logsigmoid/exponential form avoids fractional-power NaNs at underflowed p=0.
    Autocast is disabled locally. See docs/losses.md for attribution and reduction.
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
        """Store immutable modulation/reduction settings outside forward."""
        super().__init__()
        self.config = config

    @override
    def forward(self, logits: Tensor, quality: Tensor) -> Tensor:
        """Evaluate trusted caller-provided quality; empty reductions are zero."""
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
        """Admit current logits and quality before trusted forward execution."""
        match validate_quality(logits, quality):
            case Err() as failure:
                return failure
            case Ok():
                return Ok(self.forward(logits, quality))
