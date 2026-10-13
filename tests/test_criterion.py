"""End-to-end declarative criterion behavior and extension boundaries."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
import torch
import torch.testing

from obb_loss import (
    Cost,
    Criterion,
    CriterionError,
    CriterionErrorKind,
    Err,
    ImagePredictions,
    Loss,
    LossContext,
    Match,
    Matcher,
    Ok,
    Predictions,
    Targets,
    box_l1_loss,
    criterion_loss,
    hungarian_assignment,
    quality_focal_loss,
)
from obb_loss.losses.set_terms import (
    class_probability_cost,
    identity,
    matched_box_inputs,
    one_hot_classification_inputs,
    pairwise_box_l1_cost,
    target_count,
)
from obb_loss.result import ErrorKind, InputError, Result

if TYPE_CHECKING:
    from torch import Tensor

    from obb_loss.criterion_types import AssignmentSolver, CriterionFailure


@pytest.fixture
def data(device: torch.device) -> tuple[Predictions, tuple[Targets, ...]]:
    """Separate centers make the expected crossed assignments unambiguous."""
    boxes = torch.tensor(
        [
            [
                [1.0, 0.0, 2.0, 1.0, 1.0, 0.0],
                [8.0, 0.0, 2.0, 1.0, 1.0, 0.0],
                [20.0, 0.0, 2.0, 1.0, 1.0, 0.0],
            ],
            [
                [30.0, 0.0, 2.0, 1.0, 1.0, 0.0],
                [40.0, 0.0, 2.0, 1.0, 1.0, 0.0],
                [50.0, 0.0, 2.0, 1.0, 1.0, 0.0],
            ],
        ],
        device=device,
        requires_grad=True,
    )
    logits = torch.zeros(2, 3, 2, device=device, requires_grad=True)
    targets = (
        Targets(
            torch.tensor(
                [[9.0, 0.0, 2.0, 1.0, 1.0, 0.0], [2.0, 0.0, 2.0, 1.0, 1.0, 0.0]],
                device=device,
                requires_grad=True,
            ),
            torch.tensor([0, 1], device=device),
        ),
        Targets(
            torch.tensor(
                [[33.0, 0.0, 2.0, 1.0, 1.0, 0.0]], device=device, requires_grad=True
            ),
            torch.tensor([1], device=device),
        ),
    )
    return Predictions(boxes, logits), targets


def declaration(solver: AssignmentSolver = hungarian_assignment) -> Criterion:
    """Declare test policies with matching weights distinct from loss weights."""
    return Criterion(
        Matcher(
            solver,
            (
                Cost("l1", pairwise_box_l1_cost, 2.0),
                Cost("class", class_probability_cost, 1.0),
            ),
        ),
        (
            Loss("l1", box_l1_loss, matched_box_inputs, 5.0, target_count),
            Loss(
                "class",
                quality_focal_loss,
                one_hot_classification_inputs,
                1.0,
                target_count,
            ),
        ),
    )


def test_matching_inside_criterion(
    data: tuple[Predictions, tuple[Targets, ...]],
) -> None:
    """One call assigns, projects, reduces and returns gradients without mutation."""
    predictions, targets = data
    saved = tuple(
        tensor.detach().clone()
        for tensor in (
            predictions.boxes,
            predictions.logits,
            *(target.boxes for target in targets),
        )
    )
    with torch.autocast(predictions.boxes.device.type):
        result = criterion_loss(declaration(), predictions, targets)
    assert isinstance(result, Ok)
    output = result.value
    assert output.total.dtype == torch.float64
    torch.testing.assert_close(
        output.matches[0].predictions,
        torch.tensor([0, 1], device=predictions.boxes.device),
    )
    torch.testing.assert_close(
        output.matches[0].targets, torch.tensor([1, 0], device=predictions.boxes.device)
    )
    assert output.matches[1].predictions.tolist() == [0]
    torch.testing.assert_close(output.raw["l1"], output.total.new_tensor(5 / 3))
    classification = torch.log(output.total.new_tensor(2.0))
    torch.testing.assert_close(output.total, 25 / 3 + classification)
    output.total.backward()
    assert predictions.boxes.grad is not None
    assert predictions.logits.grad is not None
    assert torch.count_nonzero(predictions.boxes.grad[0, 2]) == 0
    assert torch.count_nonzero(predictions.boxes.grad[1, 1:]) == 0
    assert torch.count_nonzero(predictions.logits.grad) == predictions.logits.numel()
    for original, copy in zip(
        (predictions.boxes, predictions.logits, *(target.boxes for target in targets)),
        saved,
        strict=True,
    ):
        torch.testing.assert_close(original, copy)


def test_solver_is_injected_and_costs_are_detached(
    data: tuple[Predictions, tuple[Targets, ...]],
) -> None:
    """Replace assignment with an ordinary function; loss gradients still survive."""
    observed: list[Tensor] = []

    def solver(costs: Tensor) -> Result[Match, CriterionFailure]:
        """Choose reversed query indices without editing the criterion shell."""
        assert not costs.requires_grad
        observed.append(costs)
        indices = torch.arange(costs.shape[1], device=costs.device)
        return Ok(Match(costs.shape[0] - 1 - indices, indices))

    predictions, targets = data
    result = criterion_loss(declaration(solver), predictions, targets)
    assert isinstance(result, Ok)
    assert len(observed) == 2
    assert result.value.matches[0].predictions.tolist() == [2, 1]
    result.value.total.backward()
    assert predictions.boxes.grad is not None
    assert torch.count_nonzero(predictions.boxes.grad[0, 2]) == 1


def forbidden_solver(costs: Tensor) -> Result[Match, CriterionFailure]:
    """Fail if the shell performs an assignment that should be bypassed."""
    raise AssertionError("disabled solver called")


def forbidden_projection(
    context: LossContext,
) -> Result[tuple[Tensor, Tensor], InputError]:
    """Fail if a disabled term prepares any inputs."""
    raise AssertionError("disabled projection called")


def forbidden_compute(first: Tensor, second: Tensor) -> Result[Tensor, InputError]:
    """Fail if disabled or failed-projection computation is performed."""
    raise AssertionError("disabled compute called")


def forbidden_cost(
    predictions: ImagePredictions, targets: Targets
) -> Result[Tensor, InputError]:
    """Fail if a disabled matching cost is evaluated."""
    raise AssertionError("disabled cost called")


def forbidden_normalizer(context: LossContext) -> Tensor:
    """Fail if a disabled term evaluates normalization."""
    raise AssertionError("disabled normalizer called")


def test_disabled_terms_do_no_work(
    data: tuple[Predictions, tuple[Targets, ...]],
) -> None:
    """Zero weights skip all functions, including matching when all losses are off."""
    predictions, targets = data
    disabled = Loss(
        "disabled", forbidden_compute, forbidden_projection, 0.0, forbidden_normalizer
    )
    specification = declaration()
    specification = replace(
        specification,
        matcher=replace(
            specification.matcher,
            costs=(*specification.matcher.costs, Cost("off", forbidden_cost, 0.0)),
        ),
        losses=(*specification.losses, disabled),
    )
    enabled = criterion_loss(specification, predictions, targets)
    assert isinstance(enabled, Ok)
    assert "disabled" not in enabled.value.raw
    inactive = Criterion(Matcher(forbidden_solver, ()), (disabled,))
    result = criterion_loss(inactive, predictions, targets)
    assert isinstance(result, Ok)
    assert result.value.total.item() == 0
    assert result.value.raw == {}
    assert result.value.weighted == {}
    assert all(
        assignment.predictions.numel() == 0 for assignment in result.value.matches
    )
    result.value.total.backward()
    assert predictions.boxes.grad is not None
    assert predictions.logits.grad is not None
    assert torch.count_nonzero(predictions.boxes.grad) == 0
    assert torch.count_nonzero(predictions.logits.grad) == 0


@pytest.mark.parametrize("empty", ["targets", "queries", "batch"])
def test_empty_problems_bypass_assignment(
    data: tuple[Predictions, tuple[Targets, ...]],
    empty: str,
) -> None:
    """Classification evaluates negatives for empty targets; regression remains zero."""
    predictions, targets = data
    match empty:
        case "targets":
            targets = tuple(
                Targets(target.boxes[:0], target.labels[:0]) for target in targets
            )
        case "queries":
            predictions = Predictions(
                predictions.boxes[:, :0], predictions.logits[:, :0]
            )
        case _:
            predictions = Predictions(
                predictions.boxes[:0, :0], predictions.logits[:0, :0]
            )
            targets = ()
    result = criterion_loss(declaration(forbidden_solver), predictions, targets)
    assert isinstance(result, Ok)
    assert result.value.raw["l1"].item() == 0
    assert all(
        assignment.predictions.numel() == 0 for assignment in result.value.matches
    )
    if empty == "targets":
        torch.testing.assert_close(
            result.value.total, torch.log(result.value.total.new_tensor(2.0)) * 3
        )
    else:
        assert result.value.total.item() == 0
    result.value.total.backward()


def test_more_targets_than_queries(
    data: tuple[Predictions, tuple[Targets, ...]],
) -> None:
    """Rectangular matching exposes unmatched targets through returned assignments."""
    predictions, targets = data
    predictions = Predictions(predictions.boxes[:, :1], predictions.logits[:, :1])
    result = criterion_loss(declaration(), predictions, targets)
    assert isinstance(result, Ok)
    assert result.value.matches[0].targets.tolist() == [1]
    assert result.value.matches[0].predictions.tolist() == [0]
    assert result.value.matches[1].targets.tolist() == [0]


@pytest.mark.parametrize(
    "invalid", ["cost_name", "loss_name", "weight", "no_costs", "blank"]
)
def test_invalid_declarations(
    data: tuple[Predictions, tuple[Targets, ...]], invalid: str
) -> None:
    """Declaration errors are typed and occur before invoking any backend."""
    specification = declaration(forbidden_solver)
    match invalid:
        case "cost_name":
            specification = replace(
                specification,
                matcher=replace(
                    specification.matcher, costs=(specification.matcher.costs[0],) * 2
                ),
            )
        case "loss_name":
            specification = replace(
                specification, losses=(specification.losses[0],) * 2
            )
        case "weight":
            specification = replace(
                specification,
                matcher=replace(
                    specification.matcher,
                    costs=(Cost("bad", forbidden_cost, float("nan")),),
                ),
            )
        case "no_costs":
            specification = replace(
                specification, matcher=Matcher(forbidden_solver, ())
            )
        case _:
            specification = replace(
                specification,
                matcher=replace(
                    specification.matcher, costs=(Cost("", forbidden_cost, 1.0),)
                ),
            )
    result = criterion_loss(specification, *data)
    assert isinstance(result, Err)
    assert isinstance(result.error, CriterionError)
    assert result.error.kind is CriterionErrorKind.SPECIFICATION


@pytest.mark.parametrize(
    "invalid", ["shape", "dtype", "device", "labels", "target_shape", "batch"]
)
def test_metadata_errors(
    data: tuple[Predictions, tuple[Targets, ...]], invalid: str
) -> None:
    """Invalid input metadata prevents assignment and returns a precise category."""
    predictions, targets = data
    kind = ErrorKind.SHAPE
    match invalid:
        case "shape":
            invalid_boxes = torch.empty(2, 3, 5, device=predictions.boxes.device)
            predictions = replace(predictions, boxes=invalid_boxes)  # pyrefly: ignore[bad-argument-type]
        case "dtype":
            predictions = replace(predictions, logits=predictions.logits.long())
            kind = ErrorKind.DTYPE
        case "device":
            predictions = replace(predictions, logits=predictions.logits.to("meta"))
            kind = ErrorKind.DEVICE
        case "labels":
            targets = (
                replace(targets[0], labels=targets[0].labels.float()),
                targets[1],
            )
            kind = ErrorKind.DTYPE
        case "target_shape":
            invalid_boxes = torch.empty(2, 5, device=predictions.boxes.device)
            bad_target = replace(targets[0], boxes=invalid_boxes)  # pyrefly: ignore[bad-argument-type]
            targets = (bad_target, targets[1])
        case _:
            targets = targets[:1]
    result = criterion_loss(declaration(forbidden_solver), predictions, targets)
    assert isinstance(result, Err)
    assert isinstance(result.error, InputError)
    assert result.error.kind is kind


def failed_projection(
    context: LossContext,
) -> Result[tuple[Tensor, Tensor], InputError]:
    """Return an expected projection failure rather than numerical values."""
    return Err(InputError(ErrorKind.SHAPE, "custom", "deliberate projection failure"))


def test_projection_and_solver_failures(
    data: tuple[Predictions, tuple[Targets, ...]],
) -> None:
    """The shell preserves typed extension failures without swallowing bugs."""
    specification = declaration()
    specification = replace(
        specification,
        losses=(Loss("failed", forbidden_compute, failed_projection, 1.0, identity),),
    )
    result = criterion_loss(specification, *data)
    assert isinstance(result, Err)
    assert isinstance(result.error, InputError)
    assert result.error.field == "custom"

    def solver(costs: Tensor) -> Result[Match, CriterionFailure]:
        """Simulate a recoverable backend failure."""
        return Err(CriterionError(CriterionErrorKind.BACKEND, "custom", "unavailable"))

    result = criterion_loss(declaration(solver), *data)
    assert isinstance(result, Err)
    assert isinstance(result.error, CriterionError)
    assert result.error.field == "custom"
    with pytest.raises(AssertionError, match="disabled solver called"):
        criterion_loss(declaration(forbidden_solver), *data)


def test_bad_cost_and_loss_outputs(
    data: tuple[Predictions, tuple[Targets, ...]],
) -> None:
    """Custom numerical functions cannot bypass the tensor metadata contract."""
    predictions, targets = data

    def bad_cost(
        preds: ImagePredictions, target: Targets
    ) -> Result[Tensor, InputError]:
        """Deliberately return an incorrect pairwise shape."""
        return Ok(preds.boxes.new_zeros(1))

    specification = declaration(forbidden_solver)
    specification = replace(
        specification, matcher=Matcher(forbidden_solver, (Cost("bad", bad_cost, 1.0),))
    )
    result = criterion_loss(specification, predictions, targets)
    assert isinstance(result, Err)
    assert isinstance(result.error, InputError)
    assert result.error.kind is ErrorKind.SHAPE

    def bad_loss(first: Tensor, second: Tensor) -> Result[Tensor, InputError]:
        """Deliberately return an integer loss tensor."""
        return Ok(first.new_zeros(first.shape[0], dtype=torch.int64))

    specification = replace(
        declaration(),
        losses=(Loss("bad", bad_loss, matched_box_inputs, 1.0, identity),),
    )
    result = criterion_loss(specification, predictions, targets)
    assert isinstance(result, Err)
    assert isinstance(result.error, InputError)
    assert result.error.kind is ErrorKind.DTYPE
