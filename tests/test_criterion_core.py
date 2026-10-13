"""Independent oracles for pure cost construction and declared loss composition."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING

import pytest
import torch
import torch.nn.functional as F
import torch.testing

from obb_loss import (
    Cost,
    Err,
    ImagePredictions,
    Loss,
    LossContext,
    LossOutput,
    Match,
    Ok,
    Predictions,
    Targets,
    box_l1_loss,
    quality_focal_loss,
    stable_probiou_loss,
)
from obb_loss.criterion_core import (
    build_cost_matrix,
    evaluate_losses,
    prepare_context,
)
from obb_loss.losses.set_terms import (
    class_probability_cost,
    identity,
    matched_box_inputs,
    one_hot_classification_inputs,
    pairwise_box_l1_cost,
    pairwise_probiou_cost,
    query_count,
    supplied_quality_inputs,
    target_count,
)
from obb_loss.result import ErrorKind, InputError, Result
from tests.math_helpers import stable_probiou_loss as oracle_probiou

if TYPE_CHECKING:
    from torch import Tensor


@pytest.fixture
def context(device: torch.device) -> LossContext:
    """Use different per-image counts and crossed assignments to expose pairing bugs."""
    predictions = Predictions(
        torch.tensor(
            [
                [[1.0, 0.0, 2.0, 1.0, 1.0, 0.0], [4.0, 0.0, 2.0, 1.0, 1.0, 0.0]],
                [[8.0, 0.0, 2.0, 1.0, 1.0, 0.0], [12.0, 0.0, 2.0, 1.0, 1.0, 0.0]],
            ],
            device=device,
            requires_grad=True,
        ),
        torch.zeros(2, 2, 2, device=device, requires_grad=True),
    )
    targets = (
        Targets(
            torch.tensor(
                [[3.0, 0.0, 2.0, 1.0, 1.0, 0.0]], device=device, requires_grad=True
            ),
            torch.tensor([1], device=device),
        ),
        Targets(
            torch.tensor(
                [[10.0, 0.0, 2.0, 1.0, 1.0, 0.0], [7.0, 0.0, 2.0, 1.0, 1.0, 0.0]],
                device=device,
                requires_grad=True,
            ),
            torch.tensor([0, 1], device=device),
        ),
    )
    assignments = (
        Match(torch.tensor([1], device=device), torch.tensor([0], device=device)),
        Match(torch.tensor([0, 1], device=device), torch.tensor([1, 0], device=device)),
    )
    outcome = prepare_context(predictions, targets, assignments)
    assert isinstance(outcome, Ok)
    return outcome.value


def squared_centers(predicted: Tensor, target: Tensor) -> Result[Tensor, InputError]:
    """Demonstrate a new numerical term without modifying the criterion."""
    return Ok((predicted[:, 0] - target[:, 0]).square())


def test_composition_oracle_and_gradients(context: LossContext) -> None:
    """Crossed pairs have errors 1,1,2; unmatched boxes get no regression gradient."""
    terms = (
        Loss("l1", box_l1_loss, matched_box_inputs, 5.0, target_count),
        Loss("centers", squared_centers, matched_box_inputs, 2.0, identity),
        Loss(
            "classes",
            quality_focal_loss,
            one_hot_classification_inputs,
            1.0,
            query_count,
        ),
    )
    result = evaluate_losses(terms, context)
    assert isinstance(result, Ok)
    output = result.value
    assert output.raw.keys() == {"l1", "centers", "classes"}
    assert output.total.dtype == torch.float64
    # At zero logits, every entry has BCE log(2) and focal modulation 1/4.
    classification = (
        torch.log(torch.tensor(2.0, dtype=torch.float64, device=output.total.device))
        / 2
    )
    torch.testing.assert_close(output.raw["l1"], output.total.new_tensor(4 / 3))
    torch.testing.assert_close(output.raw["centers"], output.total.new_tensor(6))
    torch.testing.assert_close(output.raw["classes"], classification)
    torch.testing.assert_close(output.total, 5 * 4 / 3 + 12 + classification)
    output.total.backward()
    boxes, logits = context.predictions.boxes, context.predictions.logits
    assert boxes.grad is not None
    assert logits.grad is not None
    assert torch.count_nonzero(boxes.grad[0, 0]) == 0
    assert torch.count_nonzero(logits.grad[0, 0]) == 2
    for target in context.targets:
        assert target.boxes.grad is not None
        assert torch.count_nonzero(target.boxes.grad[:, 0]) == target.boxes.shape[0]


def test_costs_against_independent_oracles(context: LossContext) -> None:
    """Cost weights differ from loss weights; compare each pair independently."""
    predictions = ImagePredictions(
        context.predictions.boxes[1], context.predictions.logits[1]
    )
    target = context.targets[1]
    costs = (
        Cost("classes", class_probability_cost, 3.0),
        Cost("l1", pairwise_box_l1_cost, 2.0),
        Cost("probiou", pairwise_probiou_cost, 4.0),
    )
    result = build_cost_matrix(costs, predictions, target)
    assert isinstance(result, Ok)
    expected = torch.empty_like(result.value)
    for query in range(2):
        for obj in range(2):
            predicted = predictions.boxes[query : query + 1]
            reference = target.boxes[obj : obj + 1]
            l1 = (predicted[:, :4].double() - reference[:, :4].double()).abs().sum()
            expected[query, obj] = (
                -1.5 + 2 * l1 + 4 * oracle_probiou(predicted, reference)[0]
            )
    torch.testing.assert_close(result.value, expected)


def test_supplied_quality_preserves_gradients(context: LossContext) -> None:
    """A declaration binds external quality without automatic detachment."""
    quality = torch.full_like(context.predictions.logits, 0.3).requires_grad_()
    projection = partial(supplied_quality_inputs, quality=quality)
    term = Loss("quality", quality_focal_loss, projection, 1.0, identity)
    result = evaluate_losses((term,), context)
    assert isinstance(result, Ok)
    logits = context.predictions.logits.double()
    expected = (
        F.binary_cross_entropy_with_logits(logits, quality.double(), reduction="none")
        * (quality.double() - logits.sigmoid()).square()
    ).sum()
    torch.testing.assert_close(result.value.total, expected)
    result.value.total.backward()
    assert quality.grad is not None
    assert torch.count_nonzero(quality.grad) == quality.numel()


def test_explicit_quality_detachment(context: LossContext) -> None:
    """Binding detached quality implements an explicit alternative target policy."""
    quality = torch.full_like(context.predictions.logits, 0.3).requires_grad_()
    term = Loss(
        "quality",
        quality_focal_loss,
        partial(supplied_quality_inputs, quality=quality.detach()),
        1.0,
        identity,
    )
    result = evaluate_losses((term,), context)
    assert isinstance(result, Ok)
    result.value.total.backward()
    assert quality.grad is None


def test_projection_and_normalizer_metadata(context: LossContext) -> None:
    """Invalid quality and divisor shapes return errors before reduction."""
    bad_quality = torch.zeros(2, 2, device=context.predictions.boxes.device)
    projected = supplied_quality_inputs(context, quality=bad_quality)
    assert isinstance(projected, Err)
    assert projected.error.kind is ErrorKind.SHAPE
    loss = Loss(
        "l1",
        box_l1_loss,
        matched_box_inputs,
        1.0,
        lambda ctx: ctx.predictions.boxes.new_ones(2),
    )
    result = evaluate_losses((loss,), context)
    assert isinstance(result, Err)
    assert result.error.kind is ErrorKind.SHAPE


@pytest.mark.parametrize("failure", ["shape", "dtype", "device"])
def test_bad_match_metadata(context: LossContext, failure: str) -> None:
    """Reject malformed backend outputs before gathering any indexed tensors."""
    match failure:
        case "shape":
            bad = Match(
                torch.zeros(1, 1, dtype=torch.int64),  # pyrefly: ignore[bad-argument-type]
                torch.zeros(1, dtype=torch.int64),
            )
        case "dtype":
            bad = Match(torch.zeros(1), torch.zeros(1))
        case _:
            bad = Match(
                torch.empty(1, dtype=torch.int64, device="meta"),
                torch.empty(1, dtype=torch.int64, device="meta"),
            )
    result = prepare_context(
        context.predictions, context.targets, (bad, context.matches[1])
    )
    assert isinstance(result, Err)
    assert result.error.kind.value == failure


def matched_objective(
    predictions: Predictions, targets: tuple[Targets, ...], matches: tuple[Match, ...]
) -> Result[LossOutput, InputError]:
    """Prepare gathered tensors inside the compiled graph from borrowed leaves."""
    match prepare_context(predictions, targets, matches):
        case Err() as failure:
            return failure
        case Ok(prepared):
            loss = Loss("l1", box_l1_loss, matched_box_inputs, 2.0, target_count)
            return evaluate_losses((loss,), prepared)


def test_core_compiles_with_explicit_matches(context: LossContext) -> None:
    """Tensor-only preparation/evaluation supports fullgraph AOT eager gradients."""
    compiled = torch.compile(matched_objective, backend="aot_eager", fullgraph=True)
    arguments = (context.predictions, context.targets, context.matches)
    actual = compiled(*arguments)
    expected = matched_objective(*arguments)
    assert isinstance(actual, Ok)
    assert isinstance(expected, Ok)
    torch.testing.assert_close(actual.value.total, expected.value.total)
    actual.value.total.backward()
    assert context.predictions.boxes.grad is not None


def test_autocast_and_borrowed_inputs(context: LossContext) -> None:
    """Autocast does not alter core precision or mutate borrowed floating inputs."""
    boxes = context.predictions.boxes.detach().clone()
    logits = context.predictions.logits.detach().clone()
    with torch.autocast(boxes.device.type):
        result = evaluate_losses(
            (
                Loss(
                    "probiou",
                    stable_probiou_loss,
                    matched_box_inputs,
                    1.0,
                    target_count,
                ),
            ),
            context,
        )
    assert isinstance(result, Ok)
    assert result.value.total.dtype == torch.float64
    torch.testing.assert_close(context.predictions.boxes, boxes)
    torch.testing.assert_close(context.predictions.logits, logits)


def test_normalization_and_empty_batch(context: LossContext) -> None:
    """Metadata normalizers handle zero queries and an entirely empty batch."""
    predictions = Predictions(
        context.predictions.boxes[:0, :0], context.predictions.logits[:0, :0]
    )
    prepared = prepare_context(predictions, (), ())
    assert isinstance(prepared, Ok)
    assert prepared.value.predicted_boxes.shape == (0, 6)
    assert target_count(prepared.value).item() == 1
    assert query_count(prepared.value).item() == 1
    assert isinstance(prepare_context(context.predictions, context.targets, ()), Err)
    mismatched = replace(context.predictions, logits=context.predictions.logits[:, :1])
    assert isinstance(
        prepare_context(mismatched, context.targets, context.matches), Err
    )
