"""Quality focal loss: Li et al., NeurIPS 2020, https://arxiv.org/abs/2006.04388."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing_extensions import override

from obb_loss.classification import validate_quality
from obb_loss.config import QFLConfig, QFLParameters
from obb_loss.reduction import reduce_loss
from obb_loss.result import Err, Ok, Result

if TYPE_CHECKING:
    from shape_extensions import IntVar

    N = IntVar("N")
    C = IntVar("C")

    from obb_loss.packed import InputError

_DEFAULT_CONFIG = QFLConfig()
_DEFAULT_PARAMETERS = QFLParameters()


def quality_focal_loss(
    logits: Tensor[[N, C]],
    quality: Tensor[[N, C]],
    *,
    beta: float = 2.0,
) -> Tensor[[N, C]]:
    """Compute unreduced Quality Focal loss with supplied quality targets.

    Inputs must share a CPU/CUDA device and be finite. Autocast is disabled locally; target gradients remain connected. beta >=1 avoids singular gradients at equal quality.

    Args:
        logits: Floating [queries, classes] scores before sigmoid; no background
            column.
        quality: Floating [queries, classes] quality in [0, 1], aligned with logits;
            zero marks a negative class. Target gradients are preserved.
        beta: Finite exponent >=1 on abs(quality - sigmoid(logits)); larger values
            suppress predictions close to target quality.

    Returns:
        Float64 [queries, classes] BCE losses multiplied by the quality-error
        modulation.
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
        """Configure the classification loss.

        Args:
            config: Validated immutable settings selecting QFL beta, output reduction,
                and global weight.
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
        values = quality_focal_loss(logits, quality, beta=self.config.beta)
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
