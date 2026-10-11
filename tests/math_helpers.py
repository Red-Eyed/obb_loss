"""Extract successful tensors for independent numeric oracles and gradcheck."""

from torch import Tensor

import obb_loss as obb


def value(result: obb.Result[Tensor, obb.InputError]) -> Tensor:
    """Fail the test when valid oracle inputs violate a loss metadata contract."""
    assert isinstance(result, obb.Ok), result
    return result.value


def stable_probiou_loss(preds: Tensor, targets: Tensor) -> Tensor:
    """Expose the ProbIoU tensor to gradient and scalar-equation tests."""
    return value(obb.stable_probiou_loss(preds, targets))


def stable_probiou(preds: Tensor, targets: Tensor) -> Tensor:
    """Expose Gaussian similarity for invariant tests."""
    return value(obb.stable_probiou(preds, targets))


def varifocal_loss(
    logits: Tensor,
    quality: Tensor,
    *,
    alpha: float = 0.75,
    gamma: float = 2.0,
    quality_weighted: bool = True,
) -> Tensor:
    """Expose VFL tensors to paper-equation and gradcheck oracles."""
    return value(
        obb.varifocal_loss(
            logits, quality, alpha=alpha, gamma=gamma, quality_weighted=quality_weighted
        )
    )


def quality_focal_loss(logits: Tensor, quality: Tensor, *, beta: float = 2.0) -> Tensor:
    """Expose QFL tensors to paper-equation and gradcheck oracles."""
    return value(obb.quality_focal_loss(logits, quality, beta=beta))
