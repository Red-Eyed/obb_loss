"""Typed custom composition, ownership, module registration, and disabled terms."""

import pytest
import torch
from torch import Tensor, nn
from typing_extensions import override

from obb_loss.builder import BoxLossBuilder
from obb_loss.config import LossConfig, Reduction
from obb_loss.packed import PackedBoxes
from obb_loss.result import Err, Ok


class TrainableBoxTerm(nn.Module):
    """A user-defined term with parameters and buffers that must stay registered."""

    def __init__(self) -> None:
        """Create scalar trainable and persistent state."""
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))
        self.register_buffer("bias", torch.tensor(0.0))

    @override
    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        """Return unreduced center residuals without geometry mutation."""
        return (preds[:, 0] - targets[:, 0]).square() * self.scale


@pytest.fixture
def packed_pair(device: torch.device) -> tuple[PackedBoxes, PackedBoxes]:
    """Create one aligned sample with two matched prediction/target pairs."""
    first = torch.tensor(
        [[0.3, 0, 2, 1, 1, 0], [0.5, 0, 2, 1, 1, 0]],
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    second = first.detach().clone()
    second[:, 0] = 0
    cu = torch.tensor([0, 2], device=device)
    return PackedBoxes(first, cu), PackedBoxes(second, cu)


def test_module_registration_and_weighting(
    packed_pair: tuple[PackedBoxes, PackedBoxes],
) -> None:
    """Custom state survives device moves/state loading and receives gradients."""
    preds, targets = packed_pair
    component = TrainableBoxTerm()
    builder = BoxLossBuilder().add("center", component, weight=3)
    result = builder.build(LossConfig(reduction=Reduction.SUM, weight=2))
    assert isinstance(result, Ok)
    criterion = result.value.to(device=preds.boxes.device, dtype=torch.float64)
    assert component.scale.device == preds.boxes.device
    assert component.scale.dtype == torch.float64
    assert "term_center.function.scale" in criterion.state_dict()
    assert "term_center.function.bias" in criterion.state_dict()
    assert list(criterion.parameters()) == [component.scale]
    loss = criterion.checked(preds, targets)
    assert isinstance(loss, Ok)
    expected = ((preds.boxes[:, 0] - targets.boxes[:, 0]).square() * 12).sum()
    torch.testing.assert_close(loss.value, expected)
    loss.value.backward()
    assert component.scale.grad is not None
    assert preds.boxes.grad is not None
    criterion.load_state_dict(criterion.state_dict())


def test_disabled_and_immutable_builder(
    packed_pair: tuple[PackedBoxes, PackedBoxes],
) -> None:
    """Zero-weight functions never run and adding a term leaves its parent intact."""

    def forbidden(preds: Tensor, targets: Tensor) -> Tensor:
        """Fail if a disabled component is accidentally invoked."""
        pytest.fail("disabled function invoked")

    base = BoxLossBuilder()
    builder = base.add("disabled", forbidden, weight=0)
    assert base.terms == ()
    result = builder.build(LossConfig())
    assert isinstance(result, Ok)
    assert result.value.terms == ()
    loss = result.value.forward(*packed_pair)
    assert loss.item() == 0
    loss.backward()


def test_bad_callback_and_names(packed_pair: tuple[PackedBoxes, PackedBoxes]) -> None:
    """Runtime callback shape and assembly name failures remain typed."""

    def scalar(preds: Tensor, targets: Tensor) -> Tensor:
        """Deliberately violate the unreduced vector contract."""
        return preds.sum() + targets.sum()

    built = BoxLossBuilder().add("bad", scalar).build(LossConfig())
    assert isinstance(built, Ok)
    assert isinstance(built.value.checked(*packed_pair), Err)
    assert isinstance(BoxLossBuilder().add("bad.name", scalar).build(LossConfig()), Err)
    assert isinstance(
        BoxLossBuilder().add("x", scalar).add("x", scalar).build(LossConfig()), Err
    )
    assert isinstance(
        BoxLossBuilder().add("x", scalar, weight=float("nan")).build(LossConfig()), Err
    )
