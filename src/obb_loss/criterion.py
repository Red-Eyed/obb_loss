"""Aligned packed box criterion; detector layer orchestration belongs to callers."""

import torch
from torch import Tensor, nn
from typing_extensions import override

from obb_loss.config import LossConfig
from obb_loss.losses.stable_probiou import stable_probiou_loss, validate_pair_domain
from obb_loss.packed import ErrorKind, InputError, PackedBatch, make_packed_boxes
from obb_loss.reduction import reduce_loss
from obb_loss.result import Err, Ok, Result

_DEFAULT_CONFIG = LossConfig()


def validate_alignment(
    preds: PackedBatch, targets: PackedBatch
) -> Result[tuple[PackedBatch, PackedBatch], InputError]:
    """Validate already matched packed geometry.

    Args:
        preds: Predicted packed geometry; sample boundaries and box order must match
            targets.
        targets: Already matched reference geometry in the same units, with the same
            cumulative offsets as preds.

    Returns:
        Ok(the borrowed records) or Err(InputError) for invalid geometry, mismatched
        sample offsets or devices.
    """
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
                ErrorKind.ALIGNMENT, "cu_seqlens", "aligned sample counts must match"
            )
        )
    if preds.values.device != targets.values.device:
        return Err(InputError(ErrorKind.DEVICE, "pairs", "devices must match"))
    return Ok((preds, targets))


class ProbIoULoss(nn.Module):
    """Stable loss on already matched packed boxes; no assignment or layer loop."""

    def __init__(self, config: LossConfig = _DEFAULT_CONFIG) -> None:
        """Configure the aligned ProbIoU box loss.

        Args:
            config: Validated immutable output reduction and global weight. NONE retains
                matched pairs; MEAN averages pairs across the whole pack.
        """
        super().__init__()
        self.config = config

    @override
    # Torch stubs declare variadic forward; this module intentionally narrows it.
    def forward(self, preds: PackedBatch, targets: PackedBatch) -> Tensor:  # pyrefly: ignore[bad-override]
        """Evaluate ProbIoU on trusted, already matched packed boxes.

        Inputs must satisfy the ProbIoU numerical domain. A zero weight skips the loss computation. Use checked for typed input failures; this path omits synchronizing value validation.

        Args:
            preds: Predicted packed geometry; sample boundaries and box order must match
                targets.
            targets: Already matched reference geometry in the same units, with the same
                cumulative offsets as preds.

        Returns:
            Float64 [pairs] values for NONE, or a weighted scalar for SUM/MEAN; empty
            scalar reductions are connected zeros.
        """
        if self.config.weight == 0:
            values = preds.values[:, 0].double() * 0 + targets.values[:, 0].double() * 0
            return reduce_loss(values, self.config.reduction)
        values = stable_probiou_loss(preds.values, targets.values)
        return reduce_loss(values, self.config.reduction) * self.config.weight

    def checked(
        self, preds: PackedBatch, targets: PackedBatch
    ) -> Result[Tensor, InputError]:
        """Validate aligned packed boxes before evaluating ProbIoU.

        Args:
            preds: Predicted packed geometry; sample boundaries and box order must match
                targets.
            targets: Already matched reference geometry in the same units, with the same
                cumulative offsets as preds.

        Returns:
            Ok(loss) using the configured reduction/weight, or Err(InputError) for
            invalid geometry, offsets, alignment, device, or ProbIoU domain.
        """
        match validate_alignment(preds, targets):
            case Err() as failure:
                return failure
            case Ok():
                pass
        match validate_pair_domain(preds.values, targets.values):
            case Err() as failure:
                return failure
            case Ok():
                return Ok(self(preds, targets))
