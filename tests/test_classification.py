"""Quality classification equations, domain checks, and extreme backward cases."""

import math

import pytest
import torch
import torch.autograd
import torch.testing
from torch import Tensor

from tests.math_helpers import quality_focal_loss, varifocal_loss


@pytest.fixture
def quality_pair(device: torch.device) -> tuple[Tensor, Tensor]:
    """Create logits and independently differentiable quality targets."""
    return (
        torch.tensor(
            [[0.3, -0.8, 1.2]], dtype=torch.float64, device=device, requires_grad=True
        ),
        torch.tensor(
            [[0.7, 0.0, 0.2]], dtype=torch.float64, device=device, requires_grad=True
        ),
    )


def test_scalar_paper_equations(quality_pair: tuple[Tensor, Tensor]) -> None:
    """Compare each paper's Eq. (2) to scalar BCE/modulation independent of torch."""
    logits, targets = quality_pair
    vfl, qfl = varifocal_loss(logits, targets), quality_focal_loss(logits, targets)
    for index in range(3):
        x, y = logits[0, index].item(), targets[0, index].item()
        p = 1 / (1 + math.exp(-x))
        bce = -y * math.log(p) - (1 - y) * math.log1p(-p)
        vweight = y if y > 0 else 0.75 * p**2
        assert vfl[0, index].item() == pytest.approx(bce * vweight, rel=1e-12)
        assert qfl[0, index].item() == pytest.approx(bce * abs(y - p) ** 2, rel=1e-12)
    # VFL switches positive/negative weighting at zero; target finite differences
    # must stay inside the positive branch. Negative-logit gradients are separate.
    assert torch.autograd.gradcheck(varifocal_loss, (logits, targets + 0.1))
    assert torch.autograd.gradcheck(varifocal_loss, (logits, targets.detach()))
    assert torch.autograd.gradcheck(quality_focal_loss, quality_pair)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_extreme_logits(device: torch.device, dtype: torch.dtype) -> None:
    """Large signed logits and small fractional VFL gamma keep backward finite."""
    logits = torch.tensor(
        [[-10000, 0, 10000]], device=device, dtype=dtype, requires_grad=True
    )
    quality = torch.tensor([[0, 0.5, 1]], device=device, dtype=dtype)
    for values in (
        varifocal_loss(logits, quality, gamma=0.5),
        quality_focal_loss(logits, quality),
    ):
        assert values.dtype == torch.float64
        assert torch.isfinite(values).all()
        gradients = torch.autograd.grad(values.sum(), logits)[0]
        assert torch.isfinite(gradients).all()


def test_empty_quality(device: torch.device) -> None:
    """Empty classification vectors remain differentiable under caller reduction."""
    logits = torch.empty((0, 3), device=device).requires_grad_()
    values = quality_focal_loss(logits, logits.detach())
    assert values.shape == (0, 3)
    values.sum().backward()
    assert logits.grad is not None
