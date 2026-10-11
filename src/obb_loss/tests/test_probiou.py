"""Independent Gaussian formula, analytic gradients, and packed criterion checks."""

import math

import pytest
import torch
from einops import rearrange
from torch import Tensor

from obb_loss.config import ProbIoUConfig, Reduction, parse_config
from obb_loss.criterion import ProbIoULoss
from obb_loss.losses.stable_probiou import (
    checked_probiou_loss,
    stable_probiou,
    stable_probiou_loss,
)
from obb_loss.packed import PackedBoxes
from obb_loss.result import Err, Ok


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
        torch.linalg.slogdet(average).logabsdet
        - (
            torch.linalg.slogdet(matrices[0]).logabsdet
            + torch.linalg.slogdet(matrices[1]).logabsdet
        )
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


def test_autocast_scope_and_packed_criterion(pair: tuple[Tensor, Tensor]) -> None:
    """Criterion preserves surrounding autocast and matches pure float64 results."""
    first, second = pair
    cu = torch.tensor([0, 1], device=first.device)
    criterion = ProbIoULoss(ProbIoUConfig(reduction=Reduction.SUM, weight=2))
    with torch.autocast(first.device.type, dtype=torch.bfloat16):
        assert torch.is_autocast_enabled(first.device.type)
        result = criterion.checked(PackedBoxes(first, cu), PackedBoxes(second, cu))
        assert torch.is_autocast_enabled(first.device.type)
    assert isinstance(result, Ok)
    torch.testing.assert_close(
        result.value, 2 * stable_probiou_loss(first, second).sum()
    )
    assert isinstance(
        criterion.checked(
            PackedBoxes(first, cu),
            PackedBoxes(second, torch.tensor([0, 0, 1], device=first.device)),
        ),
        Err,
    )


def test_empty_criterion(device: torch.device) -> None:
    """Empty mean remains graph-connected and does not generate NaN."""
    boxes = torch.empty((0, 6), device=device, requires_grad=True)
    packed = PackedBoxes(boxes, torch.tensor([0, 0], device=device))
    loss = ProbIoULoss().forward(packed, packed)
    assert loss.item() == 0
    loss.backward()
    assert boxes.grad is not None


def test_config_and_domain_rejection(pair: tuple[Tensor, Tensor]) -> None:
    """Recoverable config and numerical-domain failures are explicit results."""
    assert isinstance(parse_config(ProbIoUConfig, {"weight": -1}), Err)
    assert isinstance(parse_config(ProbIoUConfig, {"unexpected": 1}), Err)
    first, second = pair
    outside = first.detach().clone()
    outside[0, 3] = 1e-40
    assert isinstance(checked_probiou_loss(outside, second), Err)


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
