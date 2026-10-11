"""Named typed callbacks for aligned packed geometry."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, cast

import torch
from torch import Tensor, nn
from typing_extensions import override

from obb_loss.composition import LossComposition, LossFunction
from obb_loss.config import Reduction
from obb_loss.criterion import validate_alignment
from obb_loss.losses.stable_probiou import stable_probiou_loss, validate_pair_domain
from obb_loss.packed import ErrorKind, InputError, PackedBoxes
from obb_loss.reduction import reduce_loss
from obb_loss.result import Err, Ok, Result

if TYPE_CHECKING:
    from collections.abc import Mapping


class BoxLossContext(NamedTuple):
    """Borrow aligned geometry for unreduced box callbacks.

    Attributes:
        preds: Valid packed prediction boxes, with one row per aligned pair.
        targets: Valid references sharing prediction offsets, device and units.
    """

    preds: PackedBoxes
    targets: PackedBoxes

    @property
    def output_count(self) -> int:
        """Return the aligned pair count without tensor computation."""
        return self.preds.boxes.shape[0]

    @property
    def device(self) -> torch.device:
        """Return the shared geometry device without tensor computation."""
        return cast("torch.device", self.preds.boxes.device)

    def zero(self) -> Tensor[[int]]:
        """Return float64 zeros connected to both aligned geometry tensors."""
        return (
            self.preds.boxes[:, 0].double() * 0 + self.targets.boxes[:, 0].double() * 0
        )


def box_probiou_loss(context: BoxLossContext) -> Result[Tensor[[int]], InputError]:
    """Evaluate ProbIoU for admitted aligned packed pairs.

    Args:
        context: Borrowed packed geometry admitted by CompositeBoxLoss.

    Returns:
        Ok(float64 [pairs] losses), without weighting or reduction, or a typed
        error for pair-dependent numerical limits.
    """
    match validate_pair_domain(context.preds.boxes, context.targets.boxes):
        case Err() as failure:
            return failure
        case Ok():
            return Ok(stable_probiou_loss(context.preds.boxes, context.targets.boxes))


class CompositeBoxLoss(nn.Module):
    """Compose named aligned-box callbacks with one final reduction."""

    def __init__(self, *, losses: Mapping[str, LossFunction[BoxLossContext]]) -> None:
        """Register independently weighted callbacks.

        Args:
            losses: Named Result-returning functions or modules receiving packed
                BoxLossContext; an empty mapping produces connected zeros. Module
                state is registered, and WeightedLoss can own each multiplier.
        """
        super().__init__()
        self.losses = LossComposition(losses)

    @override
    # Native Torch stubs declare variadic forward; packed geometry narrows it.
    def forward(  # pyrefly: ignore[bad-override]
        self,
        preds: PackedBoxes,
        targets: PackedBoxes,
        *,
        reduction: Reduction = Reduction.MEAN,
    ) -> Result[Tensor, InputError]:
        """Validate aligned boxes, compose weighted values, and reduce once.

        Args:
            preds: Borrowed packed predictions in six-coordinate geometry.
            targets: Borrowed aligned references sharing offsets, device and units.
            reduction: NONE retains [pairs]; SUM adds pairs; MEAN averages pairs.
                Empty scalar reductions return connected zero.

        Returns:
            Ok(float64 loss) or Err for input, callback or combined-output failures.
            Autocast is disabled locally; callback bugs propagate.
        """
        if not isinstance(reduction, Reduction):
            return Err(
                InputError(ErrorKind.CONFIG, "reduction", "expected Reduction enum")
            )
        match validate_alignment(preds, targets):
            case Err() as failure:
                return failure
            case Ok():
                pass
        context = BoxLossContext(preds, targets)
        with torch.autocast(context.device.type, enabled=False):
            match self.losses(context):
                case Err() as failure:
                    return failure
                case Ok(value=outputs):
                    pass
            values = context.zero()
            for output in outputs:
                values = values + output.values
            loss = reduce_loss(values, reduction)
            if not bool(torch.isfinite(loss).all()):
                return Err(
                    InputError(ErrorKind.DOMAIN, "loss", "nonfinite combined loss")
                )
            return Ok(loss)
