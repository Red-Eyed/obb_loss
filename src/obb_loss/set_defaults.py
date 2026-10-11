"""Default callback selections; every weight belongs to its callback wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING

from obb_loss.composition import WeightedLoss
from obb_loss.config import WeightConfig
from obb_loss.losses.l1 import set_l1_cost, set_l1_loss
from obb_loss.losses.stable_probiou import set_probiou_cost, set_probiou_loss
from obb_loss.losses.varifocal import set_varifocal_loss
from obb_loss.set_class_cost import set_classification_cost

if TYPE_CHECKING:
    from obb_loss.composition import LossFunction
    from obb_loss.set_context import SetCostContext, SetLossContext


def default_set_losses() -> dict[str, LossFunction[SetLossContext]]:
    """Return fresh VFL/L1/ProbIoU callbacks with weights 1/5/2.

    Returns:
        A mutable selection that callers can extend or replace before construction;
        the criterion copies its names and registers callback module state.
    """
    return {
        "classification": set_varifocal_loss,
        "l1": WeightedLoss(set_l1_loss, WeightConfig(weight=5)),
        "probiou": WeightedLoss(set_probiou_loss, WeightConfig(weight=2)),
    }


def default_set_costs() -> dict[str, LossFunction[SetCostContext]]:
    """Return fresh sigmoid-class/L1/ProbIoU cost callbacks with weights 1/5/2.

    Returns:
        Independent matching callbacks; training callback weights are never consulted.
        Lower combined cost means a preferable assignment.
    """
    return {
        "classification": set_classification_cost,
        "l1": WeightedLoss(set_l1_cost, WeightConfig(weight=5)),
        "probiou": WeightedLoss(set_probiou_cost, WeightConfig(weight=2)),
    }
