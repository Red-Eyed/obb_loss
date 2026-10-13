"""Assignment backend boundary tests with independent exhaustive optima."""

from __future__ import annotations

from itertools import permutations
from typing import TYPE_CHECKING

import pytest
import torch
import torch.testing

import obb_loss.matching as matching
from obb_loss import CriterionError, Err, Ok, hungarian_assignment
from obb_loss.criterion_types import CriterionErrorKind
from obb_loss.result import ErrorKind, InputError

if TYPE_CHECKING:
    from types import ModuleType


@pytest.mark.parametrize("shape", [(2, 3), (3, 2), (3, 3)])
def test_seeded_assignment_oracle(device: torch.device, shape: tuple[int, int]) -> None:
    """Compare rectangular solutions with every possible small assignment."""
    generator = torch.Generator().manual_seed(712)
    for _ in range(5):
        costs = torch.randn(shape, generator=generator, dtype=torch.float64).to(device)
        result = hungarian_assignment(costs.requires_grad_())
        assert isinstance(result, Ok)
        assignment = result.value
        actual = costs[assignment.predictions, assignment.targets].sum()
        rows, columns = shape
        if rows <= columns:
            expected = min(
                sum(float(costs[row, col].detach()) for row, col in enumerate(order))
                for order in permutations(range(columns), rows)
            )
        else:
            expected = min(
                sum(float(costs[row, col].detach()) for col, row in enumerate(order))
                for order in permutations(range(rows), columns)
            )
        torch.testing.assert_close(actual, actual.new_tensor(expected))
        assert assignment.predictions.device == device
        assert assignment.targets.device == device
        assert assignment.predictions.dtype == torch.int64
        assert assignment.predictions.unique().numel() == min(shape)
        assert assignment.targets.unique().numel() == min(shape)


def test_tied_costs_are_optimal() -> None:
    """Any valid tie order is allowed, with no duplicate assigned indices."""
    costs = torch.zeros(4, 3)
    outcome = hungarian_assignment(costs)
    assert isinstance(outcome, Ok)
    assert outcome.value.predictions.unique().numel() == 3
    assert outcome.value.targets.unique().numel() == 3


def missing_scipy(name: str) -> ModuleType:
    """Simulate an absent optional backend without altering the environment."""
    raise ModuleNotFoundError("missing optional scipy", name=name.split(".")[0])


def test_missing_backend_and_empty_bypass(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nonempty assignment needs SciPy; either empty dimension bypasses imports."""
    monkeypatch.setattr(matching, "import_module", missing_scipy)
    for shape in ((0, 3), (3, 0), (0, 0)):
        result = hungarian_assignment(torch.empty(shape[0], shape[1]))
        assert isinstance(result, Ok)
        assert result.value.predictions.numel() == 0
    result = hungarian_assignment(torch.zeros(1, 1))
    assert isinstance(result, Err)
    assert isinstance(result.error, CriterionError)
    assert result.error.kind is CriterionErrorKind.BACKEND
    assert "matching" in result.error.message


@pytest.mark.parametrize(
    ("costs", "kind"),
    [
        (torch.zeros(2), ErrorKind.SHAPE),
        (torch.zeros(2, 2, dtype=torch.int64), ErrorKind.DTYPE),
        (torch.empty(2, 2, device="meta"), ErrorKind.DEVICE),
    ],
)
def test_backend_metadata_errors(costs: torch.Tensor, kind: ErrorKind) -> None:
    """Metadata violations return typed errors before imports or transfers."""
    result = hungarian_assignment(costs)
    assert isinstance(result, Err)
    assert isinstance(result.error, InputError)
    assert result.error.kind is kind


def test_infeasible_costs_return_backend_error() -> None:
    """SciPy rejection is translated at the boundary, without a preflight value scan."""
    result = hungarian_assignment(torch.full((2, 2), float("inf")))
    assert isinstance(result, Err)
    assert isinstance(result.error, CriterionError)
    assert result.error.kind is CriterionErrorKind.BACKEND
