"""Aligned packed box criterion; detector layer orchestration belongs to callers."""

import torch
from torch import Tensor, nn
from typing_extensions import override

from obb_loss.config import ProbIoUConfig
from obb_loss.losses.stable_probiou import stable_probiou_loss, validate_pairs
from obb_loss.packed import ErrorKind, InputError, PackedBoxes, make_packed_boxes
from obb_loss.reduction import reduce_loss
from obb_loss.result import Err, Ok, Result

_DEFAULT_CONFIG = ProbIoUConfig()


def validate_alignment(
    preds: PackedBoxes, targets: PackedBoxes
) -> Result[tuple[PackedBoxes, PackedBoxes], InputError]:
    """Validate fresh records, then require one aligned target per prediction."""
    for packed in (preds, targets):
        match make_packed_boxes(*packed):
            case Err() as failure:
                return failure
            case Ok():
                pass
    if preds.cu_seqlens.shape != targets.cu_seqlens.shape or not torch.equal(
        preds.cu_seqlens, targets.cu_seqlens
    ):
        return Err(
            InputError(
                ErrorKind.ALIGNMENT, "cu_seqlens", "aligned image counts must match"
            )
        )
    match validate_pairs(preds.boxes, targets.boxes):
        case Err() as failure:
            return failure
        case Ok():
            return Ok((preds, targets))


class ProbIoULoss(nn.Module):
    """Stable loss on already matched packed boxes; no assignment or layer loop."""

    def __init__(self, config: ProbIoUConfig = _DEFAULT_CONFIG) -> None:
        """Store immutable settings; use checked for recoverable input validation."""
        super().__init__()
        self.config = config

    @override
    def forward(self, preds: PackedBoxes, targets: PackedBoxes) -> Tensor:
        """Compute trusted aligned float64 loss with caller-owned valid geometry.

        Records must have identical offsets and satisfy the pure function's domain.
        Inputs are borrowed; mean/sum of empty pairs is a differentiable zero.
        Use checked for typed failures; forward avoids synchronizing value checks.
        """
        values = stable_probiou_loss(preds.boxes, targets.boxes)
        return reduce_loss(values, self.config.reduction) * self.config.weight

    def checked(
        self, preds: PackedBoxes, targets: PackedBoxes
    ) -> Result[Tensor, InputError]:
        """Validate current tensors/layout/domain before invoking trusted forward."""
        match validate_alignment(preds, targets):
            case Err() as failure:
                return failure
            case Ok():
                return Ok(self.forward(preds, targets))
