"""Typed custom composition, ownership, module registration, and disabled terms."""

import pytest
import torch
import torch.testing
from torch import Tensor, nn
from typing_extensions import override

from obb_loss.box_composition import BoxLossContext, CompositeBoxLoss, box_probiou_loss
from obb_loss.composition import WeightedLoss
from obb_loss.config import Reduction, WeightConfig
from obb_loss.packed import InputError, PackedBoxes
from obb_loss.result import Err, Ok, Result


class TrainableBoxTerm(nn.Module):
    """A user-defined term with parameters and buffers that must stay registered."""

    def __init__(self) -> None:
        """Create scalar trainable and persistent state."""
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))
        self.register_buffer("bias", torch.tensor(0.0))

    @override
    # Torch stubs declare variadic forward; this module intentionally narrows it.
    def forward(self, context: BoxLossContext) -> Result[Tensor, InputError]:  # pyrefly: ignore[bad-override]
        """Return unreduced center residuals without geometry mutation."""
        return Ok(
            (context.preds.boxes[:, 0] - context.targets.boxes[:, 0]).square()
            * self.scale
        )


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


@pytest.mark.parametrize("reduction", tuple(Reduction))
def test_module_registration_and_weighting(
    packed_pair: tuple[PackedBoxes, PackedBoxes],
    reduction: Reduction,
) -> None:
    """Custom state survives device moves/state loading and receives gradients."""
    preds, targets = packed_pair
    component = TrainableBoxTerm()
    criterion = CompositeBoxLoss(
        losses={
            "center": WeightedLoss(component, WeightConfig(weight=6)),
        }
    ).to(device=preds.boxes.device, dtype=torch.float64)
    assert component.scale.device == preds.boxes.device
    assert component.scale.dtype == torch.float64
    assert "losses.term_0.function.scale" in criterion.state_dict()
    assert "losses.term_0.function.bias" in criterion.state_dict()
    assert list(criterion.parameters()) == [component.scale]
    observed: list[Result[Tensor, InputError]] = []

    def observe(
        module: nn.Module,
        inputs: tuple[BoxLossContext, ...],
        output: Result[Tensor, InputError],
    ) -> None:
        """Record the custom module hook to verify evaluation uses Module.__call__."""
        assert module is component
        assert len(inputs) == 1
        observed.append(output)

    component.register_forward_hook(observe)
    loss = criterion(preds, targets, reduction=reduction)
    assert isinstance(loss, Ok)
    assert len(observed) == 1
    expected = (preds.boxes[:, 0] - targets.boxes[:, 0]).square() * 12
    if reduction is Reduction.SUM:
        expected = expected.sum()
    elif reduction is Reduction.MEAN:
        expected = expected.mean()
    torch.testing.assert_close(loss.value, expected)
    loss.value.sum().backward()
    assert component.scale.grad is not None
    assert preds.boxes.grad is not None
    criterion.load_state_dict(criterion.state_dict())


def test_disabled_callback(packed_pair: tuple[PackedBoxes, PackedBoxes]) -> None:
    """Zero-weight callbacks have no registered state and never execute."""

    def forbidden(context: BoxLossContext) -> Result[Tensor, InputError]:
        """Fail if a disabled component is accidentally invoked."""
        pytest.fail("disabled function invoked")

    criterion = CompositeBoxLoss(
        losses={
            "disabled": WeightedLoss(forbidden, WeightConfig(weight=0)),
        }
    )
    assert criterion.losses.terms == ()
    loss = criterion(*packed_pair)
    assert isinstance(loss, Ok)
    assert loss.value.item() == 0
    loss.value.sum().backward()


def test_bad_callback(packed_pair: tuple[PackedBoxes, PackedBoxes]) -> None:
    """Runtime callback shape failures remain typed and include the term name."""

    def scalar(context: BoxLossContext) -> Result[Tensor, InputError]:
        """Deliberately violate the unreduced vector contract."""
        return Ok(context.preds.boxes.sum() + context.targets.boxes.sum())

    criterion = CompositeBoxLoss(losses={"bad.name": scalar})
    loss = criterion(*packed_pair)
    assert isinstance(loss, Err)
    assert loss.error.field == "bad.name"


def test_custom_loss_has_its_own_numerical_domain(
    packed_pair: tuple[PackedBoxes, PackedBoxes],
) -> None:
    """Valid geometry outside ProbIoU's domain remains usable by custom callbacks."""
    preds, targets = packed_pair
    boxes = preds.boxes.detach().clone()
    boxes[:, 0] = 1e31
    preds = PackedBoxes(boxes.requires_grad_(), preds.cu_seqlens)

    def center(context: BoxLossContext) -> Result[Tensor, InputError]:
        """Return a finite center residual beyond ProbIoU's normalized offset bound."""
        return Ok((context.preds.boxes[:, 0] - context.targets.boxes[:, 0]).square())

    custom = CompositeBoxLoss(losses={"center": center})(preds, targets)
    assert isinstance(custom, Ok)
    torch.testing.assert_close(custom.value, boxes.new_tensor(1e62))
    custom.value.backward()
    assert boxes.grad is not None
    assert bool(torch.isfinite(boxes.grad).all())
    assert isinstance(
        CompositeBoxLoss(losses={"probiou": box_probiou_loss})(preds, targets), Err
    )
