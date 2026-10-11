"""Quality focal loss: Li et al., NeurIPS 2020, https://arxiv.org/abs/2006.04388."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing_extensions import override

from obb_loss.classification import validate_quality
from obb_loss.config import QFLConfig
from obb_loss.reduction import reduce_loss
from obb_loss.result import Err, Ok, Result

if TYPE_CHECKING:
    from jaxtyping import Float

    from obb_loss.packed import InputError

_DEFAULT_CONFIG = QFLConfig()


def quality_focal_loss(
    logits: Float[Tensor, "n c"],  # noqa: F722
    quality: Float[Tensor, "n c"],  # noqa: F722
    *,
    beta: float = 2.0,
) -> Float[Tensor, "n c"]:  # noqa: F722
    """Return trusted unreduced float64 BCE*abs(quality-sigmoid(logits))**beta.

    Same quality domain as VFL; beta must be finite and >=1 to avoid singular
    modulation gradients at equality. beta=1 chooses abs's zero subgradient.
    Inputs/targets remain connected to autograd. Autocast is disabled locally;
    low-precision leaf gradient casts can still overflow. See docs/losses.md.
    """
    with torch.autocast(logits.device.type, enabled=False):
        scores, targets = logits.double(), quality.double()
        modulation = (targets - scores.sigmoid()).abs().pow(beta)
        return (
            F.binary_cross_entropy_with_logits(scores, targets, reduction="none")
            * modulation
        )


class QualityFocalLoss(nn.Module):
    """Configured QFL for explicit quality targets."""

    def __init__(self, config: QFLConfig = _DEFAULT_CONFIG) -> None:
        """Store immutable beta, weighting, and reduction settings."""
        super().__init__()
        self.config = config

    @override
    def forward(self, logits: Tensor, quality: Tensor) -> Tensor:
        """Evaluate trusted QFL with caller-owned target and normalization policy."""
        if self.config.weight == 0:
            return reduce_loss(
                logits.double() * 0 + quality.double() * 0, self.config.reduction
            )
        values = quality_focal_loss(logits, quality, beta=self.config.beta)
        return reduce_loss(values, self.config.reduction) * self.config.weight

    def checked(self, logits: Tensor, quality: Tensor) -> Result[Tensor, InputError]:
        """Validate current quality inputs and return a recoverable rejection."""
        match validate_quality(logits, quality):
            case Err() as failure:
                return failure
            case Ok():
                return Ok(self.forward(logits, quality))
