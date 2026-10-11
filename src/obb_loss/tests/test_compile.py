"""Verify supported tensor-only execution across changing packed dimensions."""

import torch

from obb_loss.criterion import ProbIoULoss
from obb_loss.packed import PackedBoxes


def test_fullgraph_compiled_criterion_dynamic_sizes() -> None:
    """AOT eager compilation preserves outputs and backward for different pair counts."""
    criterion = ProbIoULoss()
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
        cu = torch.tensor([0, count])
        packed1, packed2 = PackedBoxes(first, cu), PackedBoxes(second, cu)
        actual = compiled(packed1, packed2)
        expected = criterion.forward(packed1, packed2)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        assert first.grad is not None
        assert torch.isfinite(first.grad).all()
