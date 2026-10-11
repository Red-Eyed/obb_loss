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
    """Validate already matched packed geometry.

    Args:
        preds: Predicted packed geometry; sample boundaries and box order must match
            targets.
        targets: Already matched reference geometry in the same units, with the same
            cumulative offsets as preds.

    Returns:
        Ok(the borrowed records) or Err(InputError) for invalid geometry, mismatched
        sample offsets, devices, or pair-dependent ProbIoU domain.
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
    match validate_pairs(preds.boxes, targets.boxes):
        case Err() as failure:
            return failure
        case Ok():
            return Ok((preds, targets))


class ProbIoULoss(nn.Module):
    """Stable loss on already matched packed boxes; no assignment or layer loop."""

    def __init__(self, config: ProbIoUConfig = _DEFAULT_CONFIG) -> None:
        """Configure the aligned ProbIoU box loss.

        Args:
            config: Validated immutable output reduction and global weight. NONE retains
                matched pairs; MEAN averages pairs across the whole pack.
        """
        super().__init__()
        self.config = config

    @override
    # Torch stubs declare variadic forward; this module intentionally narrows it.
    def forward(self, preds: PackedBoxes, targets: PackedBoxes) -> Tensor:  # pyrefly: ignore[bad-override]
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
            values = preds.boxes[:, 0].double() * 0 + targets.boxes[:, 0].double() * 0
            return reduce_loss(values, self.config.reduction)
        values = stable_probiou_loss(preds.boxes, targets.boxes)
        return reduce_loss(values, self.config.reduction) * self.config.weight

    def checked(
        self, preds: PackedBoxes, targets: PackedBoxes
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
                return Ok(self.forward(preds, targets))
