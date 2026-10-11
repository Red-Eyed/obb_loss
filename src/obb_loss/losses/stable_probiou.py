"""Residual-form ProbIoU derived from Gaussian Bhattacharyya distance.

Source: Murrugarra-Llerena et al., Gaussian Bounding Boxes and Probabilistic
Intersection-over-Union for Object Detection (2021), https://arxiv.org/abs/2106.06072.
Positive shape and translation residuals avoid cancellation near identical boxes.
Numerical domains and saturation behavior are documented in docs/losses.md.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from einops import rearrange
from torch import Tensor

from obb_loss.packed import ErrorKind, InputError, validate_geometry
from obb_loss.result import Err, Ok, Result

if TYPE_CHECKING:
    from jaxtyping import Float


def _directions(boxes: Tensor) -> tuple[Tensor, Tensor]:
    """Normalize nonzero directions without overflowing their squared norms."""
    vectors = boxes[..., 4:6]
    scale = vectors.abs().amax(dim=-1, keepdim=True).detach()
    normalized = vectors / scale
    normalized = normalized / torch.linalg.vector_norm(normalized, dim=-1, keepdim=True)
    return normalized[..., 0], normalized[..., 1]


def _scaled_pair(first: Tensor, second: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Express geometry relative to its detached largest side in float64."""
    first64, second64 = first.double(), second.double()
    scale = torch.maximum(
        first64[..., 2:4].amax(-1), second64[..., 2:4].amax(-1)
    ).detach()
    scale_column = rearrange(scale, "... -> ... 1")
    return (
        first64[..., 2:4] / scale_column,
        second64[..., 2:4] / scale_column,
        (second64[..., :2] - first64[..., :2]) / scale_column,
    )


def _root_distance(first: Tensor, second: Tensor) -> Tensor:
    """Construct sqrt(BD) as a norm of positive shape and translation residuals."""
    sides1, sides2, delta = _scaled_pair(first, second)
    p, r = sides1[..., 0] / math.sqrt(12), sides1[..., 1] / math.sqrt(12)
    q, s = sides2[..., 0] / math.sqrt(12), sides2[..., 1] / math.sqrt(12)
    c1, s1 = _directions(first.double())
    c2, s2 = _directions(second.double())
    cosine, sine = c1 * c2 + s1 * s2, c1 * s2 - s1 * c2
    product = p * r * q * s
    shape = torch.stack(
        (p * r - q * s, cosine * (p * s - r * q), sine * (p * q - r * s)), dim=-1
    )
    shape = shape / rearrange(2 * product.sqrt(), "... -> ... 1")
    squared = shape.square().sum(-1)
    determinant = product * (1 + squared)
    dx = delta[..., 0] * c1 + delta[..., 1] * s1
    dy = -delta[..., 0] * s1 + delta[..., 1] * c1
    qx, qy = dx * cosine + dy * sine, -dx * sine + dy * cosine
    translation = torch.stack((r * dx, p * dy, s * qx, q * qy), dim=-1)
    translation = (
        translation / rearrange(4 * determinant.sqrt(), "... -> ... 1")
    ).clamp(-32, 32)
    # Inactive branches must remain finite for backward at exact equality.
    safe = torch.where(squared > 0, squared, torch.ones_like(squared))
    factor = torch.where(squared > 0, 0.5 * torch.log1p(squared) / safe, 0.5).sqrt()
    residual = torch.cat(
        (translation, shape * rearrange(factor, "... -> ... 1")), dim=-1
    )
    return torch.linalg.vector_norm(residual, dim=-1)


def stable_probiou_loss(
    preds: Float[Tensor, "... 6"],  # noqa: F722
    targets: Float[Tensor, "... 6"],  # noqa: F722
) -> Float[Tensor, "..."]:  # noqa: UP037
    """Compute unreduced Gaussian-box Hellinger loss.

    Inputs must share a CPU/CUDA device. Relative to the largest side in each pair, sides must be >=1e-30 and center offsets <=1e30. Autocast is disabled locally; inputs are not detached or mutated. Equality has a zero loss subgradient. Low-precision leaf gradient casts can overflow. This pure function trusts inputs; checked_probiou_loss validates aligned pairs.

    Args:
        preds: Predicted aligned boxes in (cx, cy, width, height, cos(theta),
            sin(theta)) order; positive sides and nonzero directions, in shared
            coordinate units. Leading dimensions must broadcast with targets.
        targets: Reference boxes in the same order and units as preds, paired
            elementwise with predictions. Leading dimensions must broadcast with
            preds.

    Returns:
        Float64 values in [0, 1] over the broadcast leading dimensions; zero means
        identical Gaussian boxes.
    """
    with torch.autocast(device_type=preds.device.type, enabled=False):
        root = _root_distance(preds, targets)
        distance = root.square()
        safe = torch.where(distance > 0, distance, torch.ones_like(distance))
        factor = torch.where(distance > 0, -torch.expm1(-distance) / safe, 1.0).sqrt()
        return (root * factor).clamp(0, 1)


def stable_probiou(
    preds: Float[Tensor, "... 6"],  # noqa: F722
    targets: Float[Tensor, "... 6"],  # noqa: F722
) -> Float[Tensor, "..."]:  # noqa: UP037
    """Compute Gaussian-box similarity as one minus Hellinger loss.

    Inputs must share a CPU/CUDA device. Relative to the largest side in each pair, sides must be >=1e-30 and center offsets <=1e30. Autocast is disabled locally; inputs are not detached or mutated. Equality has a zero loss subgradient. Low-precision leaf gradient casts can overflow. This pure function trusts inputs; checked_probiou_loss validates aligned pairs.

    Args:
        preds: Predicted aligned boxes in (cx, cy, width, height, cos(theta),
            sin(theta)) order; positive sides and nonzero directions, in shared
            coordinate units. Leading dimensions must broadcast with targets.
        targets: Reference boxes in the same order and units as preds, paired
            elementwise with predictions. Leading dimensions must broadcast with
            preds.

    Returns:
        Float64 values in [0, 1] over the broadcast leading dimensions; one means
        identical Gaussian boxes.
    """
    return 1 - stable_probiou_loss(preds, targets)


def validate_pairs(
    preds: Tensor, targets: Tensor
) -> Result[tuple[Tensor, Tensor], InputError]:
    """Validate aligned pairs and their ProbIoU numerical domain.

    Args:
        preds: Predicted aligned boxes in (cx, cy, width, height, cos(theta),
            sin(theta)) order; positive sides and nonzero directions, in shared
            coordinate units. Shape [pairs, 6].
        targets: Reference boxes in the same order and units as preds, paired
            elementwise with predictions. Same shape and CPU/CUDA device.

    Returns:
        Ok(the borrowed tensor pair) or Err(InputError) for invalid geometry,
        alignment, device, normalized side sizes, or normalized offsets.
    """
    for name, value in (("preds", preds), ("targets", targets)):
        match validate_geometry(value, name):
            case Err() as failure:
                return failure
            case Ok():
                pass
    if preds.shape != targets.shape:
        return Err(
            InputError(ErrorKind.ALIGNMENT, "pairs", "expected equal [n, 6] shapes")
        )
    if preds.device != targets.device:
        return Err(InputError(ErrorKind.DEVICE, "pairs", "devices must match"))
    sides1, sides2, delta = _scaled_pair(preds, targets)
    if not bool((torch.minimum(sides1, sides2) >= 1e-30).all()):
        return Err(
            InputError(ErrorKind.DOMAIN, "pairs", "normalized sides must be >=1e-30")
        )
    if not bool((delta.abs() <= 1e30).all()):
        return Err(
            InputError(ErrorKind.DOMAIN, "pairs", "normalized offsets must be <=1e30")
        )
    return Ok((preds, targets))


def checked_probiou_loss(preds: Tensor, targets: Tensor) -> Result[Tensor, InputError]:
    """Validate aligned box pairs and compute their ProbIoU loss.

    Accelerator admission checks synchronize. Autograd is preserved for both inputs.

    Args:
        preds: Predicted aligned boxes in (cx, cy, width, height, cos(theta),
            sin(theta)) order; positive sides and nonzero directions, in shared
            coordinate units. Shape [pairs, 6].
        targets: Reference boxes in the same order and units as preds, paired
            elementwise with predictions. Same [pairs, 6] shape and CPU/CUDA device.

    Returns:
        Ok(float64 [pairs] Hellinger losses) or Err(InputError) for invalid
        geometry, alignment, device, or numerical domain.
    """
    match validate_pairs(preds, targets):
        case Err() as failure:
            return failure
        case Ok():
            return Ok(stable_probiou_loss(preds, targets))
