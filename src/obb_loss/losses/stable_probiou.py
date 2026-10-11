"""Residual-form ProbIoU derived from Gaussian Bhattacharyya distance.

Source: Murrugarra-Llerena et al., Gaussian Bounding Boxes and Probabilistic
Intersection-over-Union for Object Detection (2021), https://arxiv.org/abs/2106.06072.
Positive shape and translation residuals avoid cancellation near identical boxes.
Inputs and numerical departures are documented below.
"""

from __future__ import annotations

import math
from itertools import zip_longest
from typing import TYPE_CHECKING

import torch
import torch.linalg
from einops import rearrange
from torch import Tensor

from obb_loss.result import (
    Err,
    ErrorKind,
    InputError,
    Ok,
    Result,
)

if TYPE_CHECKING:
    from typing import TypeVar

    from shape_extensions import Elements, IntTuple

    P = TypeVar("P", bound=IntTuple)
    T = TypeVar("T", bound=IntTuple)


def _validate_inputs(preds: Tensor, targets: Tensor) -> Result[None, InputError]:
    """Check broadcastable six-coordinate boxes, floating dtypes and shared device."""
    if (
        preds.ndim < 1
        or targets.ndim < 1
        or preds.shape[-1] != 6
        or targets.shape[-1] != 6
    ):
        return Err(
            InputError(ErrorKind.SHAPE, "boxes", "last axis must have six coordinates")
        )
    for first, second in zip_longest(
        reversed(preds.shape[:-1]), reversed(targets.shape[:-1]), fillvalue=1
    ):
        if first != second and first != 1 and second != 1:
            return Err(
                InputError(ErrorKind.SHAPE, "boxes", "leading axes must broadcast")
            )
    if not preds.is_floating_point() or not targets.is_floating_point():
        return Err(InputError(ErrorKind.DTYPE, "inputs", "floating tensors required"))
    if preds.device != targets.device or preds.device.type not in ("cpu", "cuda"):
        return Err(
            InputError(ErrorKind.DEVICE, "inputs", "same CPU/CUDA device required")
        )
    return Ok(None)


def _directions(boxes: Tensor) -> tuple[Tensor, Tensor]:
    """Normalize nonzero directions without overflowing their squared norms."""
    vectors = boxes[..., 4:6]
    scale = vectors.abs().amax(dim=-1, keepdim=True).detach()
    # Scaling before the norm avoids squaring extreme direction magnitudes.
    # This changes neither the angle nor its derivative for nonzero directions.
    normalized = vectors / scale
    normalized = normalized / torch.linalg.vector_norm(normalized, dim=-1, keepdim=True)
    return normalized[..., 0], normalized[..., 1]


def _degenerate(boxes: Tensor) -> Tensor:
    """Identify zero sides and direction pairs with no defined angle."""
    zero_side = (boxes[..., 2:4] == 0).any(-1)
    zero_direction = (boxes[..., 4:6] == 0).all(-1)
    return zero_side | zero_direction


def _safe_pair(first: Tensor, second: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Substitute valid geometry for pairs with zero sides or zero directions."""
    first64, second64 = first.double(), second.double()
    degenerate = _degenerate(first64) | _degenerate(second64)
    mask = rearrange(degenerate, "... -> ... 1")
    dummy = first64.new_tensor([0, 0, 1, 1, 1, 0])
    # Mask before divisions: selecting a finite loss afterward cannot undo
    # NaN derivatives from an unused branch. Both inputs need safe geometry.
    return (
        torch.where(mask, dummy, first64),
        torch.where(mask, dummy, second64),
        degenerate,
    )


def _scaled_pair(first: Tensor, second: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Express geometry relative to its detached largest side in float64."""
    first64, second64 = first.double(), second.double()
    scale = torch.maximum(
        first64[..., 2:4].amax(-1), second64[..., 2:4].amax(-1)
    ).detach()
    # Bhattacharyya distance is invariant to a common coordinate scale.
    # Detaching this numerical scale avoids gradients through the max selector;
    # the exact distance has zero derivative with respect to the common scale.
    scale_column = rearrange(scale, "... -> ... 1")
    return (
        first64[..., 2:4] / scale_column,
        second64[..., 2:4] / scale_column,
        (second64[..., :2] - first64[..., :2]) / scale_column,
    )


def _root_distance(first: Tensor, second: Tensor) -> Tensor:
    """Construct sqrt(BD) as a norm of positive shape and translation residuals."""
    sides1, sides2, delta = _scaled_pair(first, second)
    # A uniform rectangle has variance side**2 / 12 along each local axis.
    p, r = sides1[..., 0] / math.sqrt(12), sides1[..., 1] / math.sqrt(12)
    q, s = sides2[..., 0] / math.sqrt(12), sides2[..., 1] / math.sqrt(12)
    c1, s1 = _directions(first.double())
    c2, s2 = _directions(second.double())
    cosine, sine = c1 * c2 + s1 * s2, c1 * s2 - s1 * c2
    product = p * r * q * s
    # These residuals vanish at equal covariances. Their squared norm equals
    # det((Sigma1 + Sigma2)/2) / sqrt(det(Sigma1)*det(Sigma2)) - 1,
    # avoiding subtraction of nearly equal expanded determinants.
    shape = torch.stack(
        (p * r - q * s, cosine * (p * s - r * q), sine * (p * q - r * s)), dim=-1
    )
    shape = shape / rearrange(2 * product.sqrt(), "... -> ... 1")
    squared = shape.square().sum(-1)
    determinant = product * (1 + squared)
    # Project displacement into both box frames. The resulting sum of squares
    # gives the translation term without explicitly inverting a covariance.
    dx = delta[..., 0] * c1 + delta[..., 1] * s1
    dy = -delta[..., 0] * s1 + delta[..., 1] * c1
    qx, qy = dx * cosine + dy * sine, -dx * sine + dy * cosine
    translation = torch.stack((r * dx, p * dy, s * qx, q * qy), dim=-1)
    # Very distant boxes already have Hellinger loss indistinguishable from one.
    # Saturating residuals bounds their squares and intentionally flattens
    # gradients in that regime; it is a numerical departure from exact BD.
    translation = (
        translation / rearrange(4 * determinant.sqrt(), "... -> ... 1")
    ).clamp(-32, 32)
    # The shape contribution is log1p(squared)/2. Scaling the residuals
    # preserves this value while its ratio has the finite limit 1/2 at zero.
    # torch.where evaluates both branches, so even the unused ratio must
    # avoid division by zero to keep backward finite at exact equality.
    safe = torch.where(squared > 0, squared, torch.ones_like(squared))
    factor = torch.where(squared > 0, 0.5 * torch.log1p(squared) / safe, 0.5).sqrt()
    residual = torch.cat(
        (translation, shape * rearrange(factor, "... -> ... 1")), dim=-1
    )
    # vector_norm supplies a zero subgradient at coincidence, unlike an
    # unguarded sqrt(sum(residual**2)), whose backward divides by zero.
    return torch.linalg.vector_norm(residual, dim=-1)


def stable_probiou_loss(
    preds: Tensor[[*Elements[P], 6]],
    targets: Tensor[[*Elements[T], 6]],
) -> Result[Tensor, InputError]:
    """Compute unreduced Gaussian-box Hellinger loss.

    For nondegenerate pairs, relative to the largest side, sides must be
    >=1e-30 and center offsets <=1e30. Autocast is disabled locally; borrowed
    inputs are neither detached nor mutated. Equality uses a zero loss subgradient.
    Low-precision leaf gradient casts can overflow. Validation checks tensor
    metadata only; callers must satisfy the numerical domain. If either box
    has a zero side or a zero direction pair, loss is one with zero gradients
    for that pair. Nonzero direction pairs are normalized internally.

    Args:
        preds: Predicted aligned boxes in (cx, cy, width, height, cos(theta),
            sin(theta)) order; nonnegative sides and finite directions, in shared
            coordinate units. Leading dimensions must broadcast with targets.
        targets: Reference boxes in the same order and units as preds, paired
            elementwise with predictions. Leading dimensions must broadcast with
            preds.

    Returns:
        Ok(float64 broadcast loss values) or Err for shape/dtype/device mismatch.
        Valid-domain losses are in [0, 1]; zero means identical Gaussian boxes.
    """
    match _validate_inputs(preds, targets):
        case Err() as failure:
            return failure
        case Ok():
            pass
    with torch.autocast(device_type=preds.device.type, enabled=False):
        first, second, degenerate = _safe_pair(preds, targets)
        root = _root_distance(first, second)
        distance = root.square()
        # Hellinger = sqrt(1 - exp(-BD)). expm1 retains small distances;
        # multiplying sqrt(BD) by a ratio with limit one avoids sqrt(0)
        # in backward and preserves the norm's chosen zero subgradient.
        safe = torch.where(distance > 0, distance, torch.ones_like(distance))
        factor = torch.where(distance > 0, -torch.expm1(-distance) / safe, 1.0).sqrt()
        loss = (root * factor).clamp(0, 1)
        return Ok(torch.where(degenerate, torch.ones_like(loss), loss))


def stable_probiou(
    preds: Tensor[[*Elements[P], 6]],
    targets: Tensor[[*Elements[T], 6]],
) -> Result[Tensor, InputError]:
    """Compute Gaussian-box similarity as one minus Hellinger loss.

    For nondegenerate pairs, relative to the largest side, sides must be
    >=1e-30 and center offsets <=1e30. Autocast is disabled locally; borrowed
    inputs are neither detached nor mutated. Equality uses a zero loss subgradient.
    Low-precision leaf gradient casts can overflow. Validation checks tensor
    metadata only; callers must satisfy the numerical domain. If either box
    has a zero side or a zero direction pair, similarity is zero with zero
    gradients for that pair. Nonzero direction pairs are normalized internally.

    Args:
        preds: Predicted aligned boxes in (cx, cy, width, height, cos(theta),
            sin(theta)) order; nonnegative sides and finite directions, in shared
            coordinate units. Leading dimensions must broadcast with targets.
        targets: Reference boxes in the same order and units as preds, paired
            elementwise with predictions. Leading dimensions must broadcast with
            preds.

    Returns:
        Ok(float64 broadcast similarity) or Err for shape/dtype/device mismatch.
        Valid-domain values are in [0, 1]; one means identical Gaussian boxes.
    """
    match stable_probiou_loss(preds, targets):
        case Err() as failure:
            return failure
        case Ok(value=loss):
            return Ok(1 - loss)
