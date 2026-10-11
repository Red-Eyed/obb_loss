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


@pytest.fixture
def degenerate_pair(
    device: torch.device, zero_coordinates: tuple[int, ...], invalid_input: str
) -> tuple[Tensor, Tensor]:
    """Provide collapsed geometry on either or both sides of a comparison."""
    first = torch.tensor([[0.2, 0.1, 2, 1, 1, 0]], device=device, dtype=torch.float64)
    second = torch.tensor([[0, 0, 1.4, 0.8, 1, 0]], device=device, dtype=torch.float64)
    if invalid_input in ("prediction", "both"):
        first[:, zero_coordinates] = 0
    if invalid_input in ("target", "both"):
        second[:, zero_coordinates] = 0
    return first.requires_grad_(), second.requires_grad_()


@pytest.mark.parametrize(
    "zero_coordinates",
    [(2,), (3,), (2, 3), (4, 5), (0, 1, 2, 3, 4, 5)],
    ids=["zero-width", "zero-height", "zero-sides", "zero-direction", "all-zero"],
)
@pytest.mark.parametrize("invalid_input", ["prediction", "target", "both"])
def test_zero_sizes_require_finite_forward_backward(
    degenerate_pair: tuple[Tensor, Tensor],
) -> None:
    """Require finite loss, similarity and gradients for degenerate boxes."""
    first, second = degenerate_pair
    loss = stable_probiou_loss(first, second)
    assert torch.isfinite(loss).all()
    similarity = stable_probiou(first, second)
    torch.testing.assert_close(similarity, torch.zeros_like(similarity), atol=0, rtol=0)
    torch.testing.assert_close(loss, torch.ones_like(loss), atol=0, rtol=0)
    gradients = torch.autograd.grad(loss.sum(), degenerate_pair)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    for gradient in gradients:
        torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=0, rtol=0)


def test_safe_fallback_preserves_valid_pair_gradients(
    pair: tuple[Tensor, Tensor],
) -> None:
    """A degenerate row leaves the valid row's value and both gradients intact."""
    first, second = pair
    mixed_first = torch.cat((torch.zeros_like(first), first.detach())).requires_grad_()
    mixed_second = torch.cat((second.detach(), second.detach())).requires_grad_()
    actual = stable_probiou_loss(mixed_first, mixed_second)
    expected = stable_probiou_loss(first, second)
    torch.testing.assert_close(actual[:1], torch.ones_like(expected), atol=0, rtol=0)
    torch.testing.assert_close(actual[1:], expected, atol=0, rtol=0)
    gradients = torch.autograd.grad(actual.sum(), (mixed_first, mixed_second))
    references = torch.autograd.grad(expected.sum(), pair)
    for gradient, reference in zip(gradients, references, strict=True):
        torch.testing.assert_close(
            gradient[:1], torch.zeros_like(reference), atol=0, rtol=0
        )
        torch.testing.assert_close(gradient[1:], reference, atol=0, rtol=0)


@pytest.mark.parametrize("zero_coordinates", [(2,), (3,), (2, 3)])
@pytest.mark.parametrize("invalid_input", ["prediction"])
def test_l1_supplies_size_gradients_alongside_safe_fallback(
    degenerate_pair: tuple[Tensor, Tensor],
    zero_coordinates: tuple[int, ...],
) -> None:
    """L1 supplies a recovery gradient at zero box sizes despite constant ProbIoU."""
    first, second = degenerate_pair
    l1 = (first[..., :4] - second[..., :4]).abs().sum()
    combined = stable_probiou_loss(first, second).sum() + l1
    assert torch.isfinite(combined)
    actual = torch.autograd.grad(combined, first, retain_graph=True)[0]
    expected = torch.autograd.grad(l1, first)[0]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for coordinate in zero_coordinates:
        assert actual[0, coordinate].item() == -1


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("logit", [0.0, -1000.0])
def test_l1_size_gradient_can_vanish_at_saturated_sigmoid(
    device: torch.device, dtype: torch.dtype, logit: float
) -> None:
    """A nonzero L1 size derivative cannot bypass a zero decoding derivative."""
    raw = torch.tensor(logit, device=device, dtype=dtype, requires_grad=True)
    size = raw.sigmoid()
    loss = (size - 0.2).abs()
    size_gradient, raw_gradient = torch.autograd.grad(loss, (size, raw))
    if logit == 0:
        assert size.item() == 0.5
        assert size_gradient.item() == 1
        assert raw_gradient.item() == 0.25
    else:
        assert size.item() == 0
        assert size_gradient.item() == -1
        assert raw_gradient.item() == 0


@pytest.mark.parametrize(
    ("coordinates", "value"),
    [
        ((4, 5), 0.0),
        ((2,), 1e-12),
        ((3,), 1e-12),
        ((4, 5), 1e-12),
    ],
    ids=[
        "zero-direction",
        "tiny-width",
        "tiny-height",
        "tiny-direction",
    ],
)
@pytest.mark.parametrize("invalid_input", ["prediction", "target"])
def test_small_sizes_and_directions_require_finite_forward_backward(
    pair: tuple[Tensor, Tensor],
    coordinates: tuple[int, ...],
    value: float,
    invalid_input: str,
) -> None:
    """Exercise collapsed directions and small positive geometry during training."""
    first, second = (boxes.detach().clone() for boxes in pair)
    invalid = first if invalid_input == "prediction" else second
    invalid[:, coordinates] = value
    first.requires_grad_()
    second.requires_grad_()
    loss = stable_probiou_loss(first, second)
    assert torch.isfinite(loss).all()
    assert torch.isfinite(stable_probiou(first, second)).all()
    for gradient in torch.autograd.grad(loss.sum(), (first, second)):
        assert torch.isfinite(gradient).all()


def test_zero_size_row_requires_finite_shared_target_gradient(
    pair: tuple[Tensor, Tensor],
) -> None:
    """A collapsed prediction must not poison gradients into a broadcast target."""
    first, second = pair
    mixed_first = torch.cat((first.detach(), first.detach()))
    mixed_first[0, 2] = 0
    mixed_first.requires_grad_()
    actual = stable_probiou_loss(mixed_first, second)
    expected = stable_probiou_loss(first, second)
    torch.testing.assert_close(actual[1:], expected, atol=0, rtol=0)
    assert torch.isfinite(actual).all()
    gradients = torch.autograd.grad(actual.sum(), (mixed_first, second))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("size", [1.0, 1e-3, 1e-5])
def test_small_box_leaf_gradients_require_finite_values(
    device: torch.device, dtype: torch.dtype, size: float
) -> None:
    """FP32 and BF16 box outputs retain finite gradients for small positive boxes."""
    first = torch.tensor(
        [[0, 0, size, size, 1, 0]], device=device, dtype=dtype, requires_grad=True
    )
    second = torch.tensor(
        [[size / 4, 0, size, size, 1, 0]],
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    loss = stable_probiou_loss(first, second)
    assert torch.isfinite(loss).all()
    assert ((loss >= 0) & (loss <= 1)).all()
    for gradient in torch.autograd.grad(loss.sum(), (first, second)):
        assert torch.isfinite(gradient).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("zero_coordinates", [(2,), (3,), (2, 3), (4, 5)])
def test_collapsed_model_outputs_require_finite_backward(
    device: torch.device, dtype: torch.dtype, zero_coordinates: tuple[int, ...]
) -> None:
    """Check backward independently of forward assertions for model output dtypes."""
    first = torch.tensor([[0.5, 0.5, 0.2, 0.1, 1, 0]], device=device, dtype=dtype)
    first[:, zero_coordinates] = 0
    first.requires_grad_()
    second = torch.tensor(
        [[0.4, 0.4, 0.1, 0.2, 1, 0]], device=device, dtype=dtype, requires_grad=True
    )
    loss = stable_probiou_loss(first, second)
    for gradient in torch.autograd.grad(loss.sum(), (first, second)):
        assert torch.isfinite(gradient).all()


def test_zero_size_query_preserves_other_query_gradient(
    pair: tuple[Tensor, Tensor],
) -> None:
    """A collapsed query must not change another query's loss or gradient."""
    first, second = pair
    predictions = torch.cat((first.detach(), first.detach()))
    predictions[0, 2] = 0
    predictions.requires_grad_()
    targets = torch.cat((second.detach(), second.detach()))
    loss = stable_probiou_loss(predictions, targets)
    expected = stable_probiou_loss(first, second)
    torch.testing.assert_close(loss[1:], expected, atol=0, rtol=0)
    actual_gradient = torch.autograd.grad(loss.sum(), predictions)[0]
    expected_gradient = torch.autograd.grad(expected.sum(), first)[0]
    assert torch.isfinite(actual_gradient[1:]).all()
    torch.testing.assert_close(actual_gradient[1:], expected_gradient, atol=0, rtol=0)


@pytest.mark.parametrize("zero_coordinates", [(2,), (3,), (2, 3)])
def test_sigmoid_size_collapse_requires_finite_logit_gradients(
    device: torch.device, zero_coordinates: tuple[int, ...]
) -> None:
    """Require safe backward through FP32 decoding when finite logits saturate."""
    logits = torch.zeros((1, 5), device=device, dtype=torch.float32)
    logits[:, zero_coordinates] = -1000
    logits.requires_grad_()
    angle = logits[..., 4:5]
    boxes = torch.cat((logits[..., :4].sigmoid(), angle.cos(), angle.sin()), dim=-1)
    target = boxes.new_tensor([[0.5, 0.5, 0.2, 0.1, 1, 0]])
    loss = stable_probiou_loss(boxes, target)
    gradient = torch.autograd.grad(loss.sum(), logits)[0]
    assert torch.isfinite(gradient).all()


def test_broadcast_invalid_pairs_require_finite_forward_backward(
    pair: tuple[Tensor, Tensor],
) -> None:
    """Check pairwise broadcast masking without contaminating valid comparisons."""
    first, second = pair
    predictions = rearrange(
        torch.cat((first.detach(), torch.zeros_like(first))), "b c -> b 1 c"
    ).requires_grad_()
    targets = rearrange(
        second.detach().expand(3, 6).clone(), "n c -> 1 n c"
    ).requires_grad_()
    loss = stable_probiou_loss(predictions, targets)
    assert loss.shape == (2, 3)
    expected = stable_probiou_loss(first, second)
    torch.testing.assert_close(loss[0], expected.expand(3), atol=0, rtol=0)
    assert torch.isfinite(loss).all()
    for gradient in torch.autograd.grad(loss.sum(), (predictions, targets)):
        assert torch.isfinite(gradient).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "direction",
    [
        (0.0, 0.0),
        (0.0, 1.0),
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, -1.0),
        (3.0, -4.0),
        (-3.0, 4.0),
        (1.0, 1.0),
        (0.5, 0.5),
        (1e-12, -1e-12),
        (1e4, -1e4),
    ],
    ids=[
        "zero-pair",
        "zero-cosine",
        "zero-sine",
        "negative-cosine",
        "negative-sine",
        "non-unit",
        "opposite-non-unit",
        "equal-unit-components",
        "equal-half-components",
        "tiny-pair",
        "large-pair",
    ],
)
def test_model_direction_outputs_require_finite_forward_backward(
    device: torch.device, dtype: torch.dtype, direction: tuple[float, float]
) -> None:
    """Allow finite unconstrained direction outputs, including a collapsed pair."""
    cosine, sine = direction
    first = torch.tensor(
        [[0.5, 0.5, 0.2, 0.1, cosine, sine]],
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    second = torch.tensor(
        [[0.4, 0.4, 0.1, 0.2, 1, 0]],
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    loss = stable_probiou_loss(first, second)
    similarity = stable_probiou(first, second)
    assert torch.isfinite(loss).all()
    assert torch.isfinite(similarity).all()
    assert ((loss >= 0) & (loss <= 1)).all()
    torch.testing.assert_close(similarity, 1 - loss, atol=0, rtol=0)
    for gradient in torch.autograd.grad(loss.sum(), (first, second)):
        assert torch.isfinite(gradient).all()


@pytest.mark.parametrize("scale", [1e-12, 1e-4, 1.0, 1e4])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_nonunit_directions_preserve_values_and_scaled_gradients(
    device: torch.device, scale: float, sign: float
) -> None:
    """Direction magnitude and a half-turn preserve values and chain-rule gradients."""
    first = torch.tensor(
        [[0.5, 0.5, 0.2, 0.1, 0.6, -0.8]],
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )
    second = first.new_tensor([[0.4, 0.4, 0.1, 0.2, 1, 0]])
    rescaled = first.detach().clone()
    rescaled[..., 4:6] *= sign * scale
    rescaled.requires_grad_()
    expected = stable_probiou_loss(first, second)
    actual = stable_probiou_loss(rescaled, second)
    torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-12)
    expected_gradient = torch.autograd.grad(expected.sum(), first)[0]
    actual_gradient = torch.autograd.grad(actual.sum(), rescaled)[0]
    actual_gradient[..., 4:6] *= sign * scale
    assert torch.isfinite(actual_gradient).all()
    torch.testing.assert_close(
        actual_gradient, expected_gradient, atol=1e-12, rtol=1e-10
    )
