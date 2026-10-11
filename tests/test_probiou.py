"""Independent Gaussian equations, analytic gradients and numerical stability."""

import math

import pytest
import torch
import torch.autograd
import torch.linalg
import torch.testing
from einops import rearrange
from torch import Tensor

from tests.math_helpers import stable_probiou, stable_probiou_loss


@pytest.fixture
def pair(device: torch.device) -> tuple[Tensor, Tensor]:
    """Provide well-conditioned independent prediction and target leaves."""
    return (
        torch.tensor(
            [[0.2, 0, 2, 1, math.cos(0.3), math.sin(0.3)]],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        ),
        torch.tensor(
            [[0, 0, 1.4, 0.8, math.cos(-0.4), math.sin(-0.4)]],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        ),
    )


def matrix_loss(first: Tensor, second: Tensor) -> Tensor:
    """Evaluate preprint Eqs. (7)/(10) using independent matrix solves/logdets."""
    matrices: list[Tensor] = []
    for boxes in (first, second):
        direction = boxes[:, 4:6] / torch.linalg.vector_norm(
            boxes[:, 4:6], dim=-1, keepdim=True
        )
        c, s = direction[:, 0], direction[:, 1]
        rotation = rearrange(
            torch.stack((c, -s, s, c), dim=-1), "n (r c) -> n r c", r=2, c=2
        )
        covariance = (
            rotation @ torch.diag_embed(boxes[:, 2:4].square() / 12) @ rotation.mT
        )
        matrices.append(covariance)
    average = (matrices[0] + matrices[1]) / 2
    delta = first[:, :2] - second[:, :2]
    solution = torch.linalg.solve(average, rearrange(delta, "n c -> n c 1"))
    translation = (delta * rearrange(solution, "n c 1 -> n c")).sum(-1) / 8
    shape = (
        torch.linalg.slogdet(average)[1]
        - (torch.linalg.slogdet(matrices[0])[1] + torch.linalg.slogdet(matrices[1])[1])
        / 2
    ) / 2
    return (-torch.expm1(-(translation + shape))).sqrt()


def test_independent_forward_backward(pair: tuple[Tensor, Tensor]) -> None:
    """Require agreement with independent paper equations for values and gradients."""
    first, second = pair
    actual = stable_probiou_loss(first, second)
    expected = matrix_loss(first, second)
    torch.testing.assert_close(actual, expected, atol=2e-14, rtol=2e-12)
    gradients = torch.autograd.grad(actual.sum(), pair)
    references = torch.autograd.grad(expected.sum(), pair)
    for gradient, reference in zip(gradients, references, strict=True):
        torch.testing.assert_close(gradient, reference, atol=2e-12, rtol=2e-10)
    assert torch.autograd.gradcheck(stable_probiou_loss, pair)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_identity_finite_zero_subgradient(
    device: torch.device, dtype: torch.dtype
) -> None:
    """Every supported input dtype has exact equality and finite zero gradients."""
    first = torch.tensor(
        [[0, 0, 2, 1, 1, 0]], device=device, dtype=dtype, requires_grad=True
    )
    second = first.detach().clone().requires_grad_()
    loss = stable_probiou_loss(first, second)
    assert loss.dtype == torch.float64
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=0, rtol=0)
    for gradient in torch.autograd.grad(loss.sum(), (first, second)):
        torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=0, rtol=0)


@pytest.mark.parametrize("offset", [1e-12, 1e-6, 0.1, 100.0])
def test_analytic_center_shift(device: torch.device, offset: float) -> None:
    """Resolve tiny losses and limiting one-sided gradients without subtraction bias."""
    first = torch.tensor([[0, 0, 2, 1, 1, 0]], device=device, dtype=torch.float64)
    second = first.clone()
    second[0, 0] = offset
    second.requires_grad_()
    loss = stable_probiou_loss(first, second)
    expected = math.sqrt(-math.expm1(-1.5 * (offset / 2) ** 2))
    assert loss.item() == pytest.approx(expected, abs=1e-15, rel=1e-12)
    gradient = torch.autograd.grad(loss.sum(), second)[0]
    assert torch.isfinite(gradient).all()
    if offset < 1e-3:
        assert gradient[0, 0].item() == pytest.approx(math.sqrt(1.5) / 2, rel=1e-6)


@pytest.mark.parametrize("scale", [1e-30, 1.0, 1e30])
@pytest.mark.parametrize("aspect", [1.0, 1e12, 1e24])
def test_thin_box_analytic_gradient(
    device: torch.device, scale: float, aspect: float
) -> None:
    """Check extreme domains using an analytic oracle that avoids matrix conditioning."""
    first = torch.tensor(
        [[0, 0, scale, scale / aspect, 1, 0]], device=device, dtype=torch.float64
    )
    second = first.clone()
    second[0, 1] = 0.1 * scale / aspect
    second.requires_grad_()
    loss = stable_probiou_loss(first, second)
    expected = math.sqrt(-math.expm1(-0.015))
    assert loss.item() == pytest.approx(expected, abs=2e-14)
    derivative = torch.autograd.grad(loss.sum(), second)[0][0, 1]
    assert (derivative * scale / aspect).item() == pytest.approx(
        math.exp(-0.015) * 0.3 / (2 * expected), rel=2e-12
    )


def test_autocast_scope(pair: tuple[Tensor, Tensor]) -> None:
    """Pure loss conversion preserves the surrounding autocast state."""
    first, second = pair
    with torch.autocast(first.device.type, dtype=torch.bfloat16):
        assert torch.is_autocast_enabled(first.device.type)
        actual = stable_probiou_loss(first, second)
        assert torch.is_autocast_enabled(first.device.type)
    torch.testing.assert_close(actual, stable_probiou_loss(first, second))


def test_empty_loss(device: torch.device) -> None:
    """An empty loss vector stays differentiable when callers sum it."""
    boxes = torch.empty((0, 6), device=device).requires_grad_()
    values = stable_probiou_loss(boxes, boxes)
    assert values.shape == (0,)
    values.sum().backward()
    assert boxes.grad is not None


def test_direction_rescaling_and_symmetry(pair: tuple[Tensor, Tensor]) -> None:
    """Direction magnitudes and argument order preserve Gaussian geometry."""
    first, second = pair
    rescaled = first.detach().clone()
    rescaled[:, 4:6] *= -1e30
    torch.testing.assert_close(
        stable_probiou(first, second),
        stable_probiou(rescaled, second),
        atol=1e-14,
        rtol=1e-12,
    )
    torch.testing.assert_close(
        stable_probiou(first, second),
        stable_probiou(second, first),
        atol=1e-14,
        rtol=1e-12,
    )
