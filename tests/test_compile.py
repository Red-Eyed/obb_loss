"""Verify supported tensor-only execution across changing pair dimensions."""

import torch
import torch.testing

import obb_loss as obb
from tests.math_helpers import stable_probiou_loss


def test_fullgraph_compiled_loss_dynamic_sizes() -> None:
    """AOT eager compilation preserves outputs and backward for different pair counts."""
    criterion = obb.stable_probiou_loss
    compiled = torch.compile(
        criterion, backend="aot_eager", fullgraph=True, dynamic=True
    )
    for count in (2, 3):
        first = (
            torch.tensor([[0.2, 0, 2, 1, 1, 0]], dtype=torch.float64)
            .repeat(count, 1)
            .requires_grad_()
        )
        second = first.detach().clone()
        second[:, 0] = 0
        result = compiled(first, second)
        assert isinstance(result, obb.Ok)
        actual = result.value
        expected = stable_probiou_loss(first, second)
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        assert first.grad is not None
        assert torch.isfinite(first.grad).all()
