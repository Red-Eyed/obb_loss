"""Aligned box L1 equations and independent gradient verification."""

import torch
import torch.autograd
import torch.testing
from torch import Tensor

import obb_loss as obb
from tests.math_helpers import value


def test_l1_equations_and_gradients(device: torch.device) -> None:
    """L1 sums center/side errors, excludes direction, and agrees with gradcheck."""
    first = torch.tensor(
        [[0.2, -0.4, 2, 1, 0, 1]],
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )
    second = torch.tensor(
        [[0.0, 0, 1, 2, 1, 0]], device=device, dtype=torch.float64, requires_grad=True
    )

    def loss(preds: Tensor, targets: Tensor) -> Tensor:
        """Extract the successful loss for numerical differentiation."""
        return value(obb.box_l1_loss(preds, targets))

    torch.testing.assert_close(loss(first, second), first.new_tensor([2.6]))
    assert torch.autograd.gradcheck(loss, (first, second))
    loss(first, second).sum().backward()
    assert first.grad is not None
    assert bool((first.grad[:, 4:] == 0).all())
