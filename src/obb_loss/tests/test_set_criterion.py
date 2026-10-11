"""Configured packed set losses, matching mathematics and public input contracts."""

import math
from itertools import permutations

import pytest
import torch
import torch.autograd
import torch.testing
from torch import Tensor
from typing_extensions import override

import obb_loss as obb


class QualityCriterion(obb.SetCriterion):
    """Supply explicit constant quality in assignment order for independent oracles."""

    @override
    def classification_quality(
        self,
        preds: obb.Predictions,
        targets: obb.Targets,
        assignment: obb.Assignment,
    ) -> obb.Result[Tensor, obb.InputError]:
        """Return caller-selected 0.7 scores; annotations carry no quality field."""
        return obb.Ok(
            torch.full(
                assignment.pred_indices.shape,
                0.7,
                device=preds.boxes.values.device,
                dtype=torch.float64,
            )
        )


@pytest.fixture
def sets(device: torch.device) -> tuple[obb.Predictions, obb.Targets]:
    """Two queries prefer different assignments under geometry and classification."""
    boxes = torch.tensor(
        [[0.2, 0, 2, 1, 1, 0], [0.8, 0, 2, 1, 1, 0]],
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )
    logits = torch.tensor(
        [[4.0, -4.0], [-4.0, 4.0]],
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )
    return (
        obb.Predictions(
            obb.PackedBatch(boxes, torch.tensor([0, 2], device=device)), logits
        ),
        obb.Targets(
            obb.PackedBatch(
                torch.tensor(
                    [[0.0, 0, 2, 1, 1, 0]], device=device, dtype=torch.float64
                ),
                torch.tensor([0, 1], device=device),
            ),
            torch.tensor([1], device=device),
        ),
    )


def test_l1_math_and_gradients(device: torch.device) -> None:
    """L1 sums center/side errors, excludes direction and agrees with gradcheck."""
    first = torch.tensor(
        [[0.2, -0.4, 2, 1, 0, 1]],
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )
    second = torch.tensor(
        [[0.0, 0, 1, 2, 1, 0]], device=device, dtype=torch.float64, requires_grad=True
    )
    torch.testing.assert_close(obb.box_l1_loss(first, second), first.new_tensor([2.6]))
    assert torch.autograd.gradcheck(obb.box_l1_loss, (first, second))
    obb.box_l1_loss(first, second).sum().backward()
    assert first.grad is not None
    assert bool((first.grad[:, 4:] == 0).all())


def test_weighted_cost_and_assignment(
    sets: tuple[obb.Predictions, obb.Targets],
) -> None:
    """Configured independent cost multipliers match the scalar assignment oracle."""
    preds, targets = sets
    matcher = obb.SetHungarianMatcher(
        obb.MatcherConfig(
            classification_weight=10, l1_weight=2, probiou_weight=0, chunk_size=1
        )
    )
    costs = matcher.pairwise_costs(preds, targets)
    assert isinstance(costs, obb.Ok)
    expected = preds.boxes.values.new_tensor(
        [-10 / (1 + math.exp(4)) + 0.4, -10 / (1 + math.exp(-4)) + 1.6]
    )
    torch.testing.assert_close(costs.value.values, expected)
    matched = matcher(preds, targets)
    assert isinstance(matched, obb.Ok)
    assert matched.value.pred_indices.tolist() == [1]
    assert not matched.value.pred_indices.requires_grad


@pytest.mark.parametrize("qfl", [False, True])
def test_component_losses_and_backward(
    sets: tuple[obb.Predictions, obb.Targets], qfl: bool
) -> None:
    """Configured training weights match independent pure losses and preserve gradients."""
    preds, targets = sets
    config = obb.SetCriterionConfig(
        matcher=obb.MatcherConfig(
            classification_weight=0, l1_weight=1, probiou_weight=0
        ),
        classification="qfl" if qfl else "vfl",
        classification_weight=3,
        l1_weight=4,
        probiou_weight=2,
        normalizer=2,
    )
    result = QualityCriterion(config)(preds, targets)
    assert isinstance(result, obb.Ok)
    first, second = obb.gather_matched(
        preds.boxes, targets.boxes, result.value.assignment
    )
    torch.testing.assert_close(
        result.value.terms["l1"], preds.boxes.values.new_tensor(0.4)
    )
    torch.testing.assert_close(
        result.value.terms["probiou"],
        obb.stable_probiou_loss(first.values, second.values).sum(),
    )
    quality = torch.zeros_like(preds.logits)
    quality[0, 1] = 0.7
    expected = (
        (
            obb.quality_focal_loss(preds.logits, quality)
            if qfl
            else obb.varifocal_loss(preds.logits, quality)
        ).sum()
        * 3
        / 2
    )
    torch.testing.assert_close(result.value.terms["classification"], expected)
    torch.testing.assert_close(result.value.total, sum(result.value.terms.values()))
    result.value.total.backward()
    assert preds.boxes.values.grad is not None
    assert preds.logits.grad is not None
    assert bool((preds.boxes.values.grad[1] == 0).all())
    assert bool((preds.logits.grad[1] != 0).all())


def test_loss_weights_do_not_change_matching(
    sets: tuple[obb.Predictions, obb.Targets],
) -> None:
    """Training weights alter losses while independent cost weights alter assignment."""
    preds, targets = sets
    geometry = obb.MatcherConfig(classification_weight=0, l1_weight=1, probiou_weight=0)
    first = QualityCriterion(obb.SetCriterionConfig(matcher=geometry))(preds, targets)
    second = QualityCriterion(
        obb.SetCriterionConfig(
            matcher=geometry, classification_weight=9, l1_weight=0, probiou_weight=0
        )
    )(preds, targets)
    third = QualityCriterion(
        obb.SetCriterionConfig(
            matcher=obb.MatcherConfig(
                classification_weight=10, l1_weight=1, probiou_weight=0
            )
        )
    )(preds, targets)
    assert isinstance(first, obb.Ok)
    assert isinstance(second, obb.Ok)
    assert isinstance(third, obb.Ok)
    assert (
        first.value.assignment.pred_indices.tolist()
        == second.value.assignment.pred_indices.tolist()
        == [0]
    )
    assert third.value.assignment.pred_indices.tolist() == [1]
    assert not torch.equal(first.value.total, second.value.total)


def test_disabled_losses_skip_quality(
    sets: tuple[obb.Predictions, obb.Targets],
) -> None:
    """Zero loss weights avoid classification quality while retaining connected zero."""
    preds, targets = sets
    criterion = obb.SetCriterion(
        obb.SetCriterionConfig(classification_weight=0, l1_weight=0, probiou_weight=0)
    )
    result = criterion(preds, targets)
    assert isinstance(result, obb.Ok)
    assert result.value.total.item() == 0
    result.value.total.backward()
    assert preds.boxes.values.grad is not None
    assert preds.logits.grad is not None
    assert bool((preds.logits.grad == 0).all())
    assert isinstance(obb.SetCriterion()(preds, targets), obb.Err)


@pytest.mark.parametrize("normalizer", [0, -1, math.nan, math.inf])
def test_invalid_config_normalizer(normalizer: float) -> None:
    """External configuration rejects divisors that cannot produce finite normalization."""
    assert isinstance(
        obb.parse_config(obb.SetCriterionConfig, {"normalizer": normalizer}), obb.Err
    )


def test_set_config_validation() -> None:
    """Configuration rejects malformed multipliers and unknown fields and roundtrips."""
    for weight in (-1.0, math.nan, math.inf):
        assert isinstance(
            obb.parse_config(obb.SetCriterionConfig, {"l1_weight": weight}), obb.Err
        )
    assert isinstance(obb.parse_config(obb.SetCriterionConfig, {"unknown": 1}), obb.Err)
    config = obb.SetCriterionConfig(
        classification="qfl", qfl=obb.QFLParameters(beta=1.5)
    )
    assert (
        obb.SetCriterionConfig.model_validate_json(config.model_dump_json()) == config
    )


def test_revalidate_fields_and_storage(
    sets: tuple[obb.Predictions, obb.Targets],
) -> None:
    """Current labels, layouts, logits and pair storage are checked on every call."""
    preds, targets = sets
    criterion = QualityCriterion()
    assert isinstance(
        criterion(preds, targets._replace(labels=targets.labels + 2)), obb.Err
    )
    assert isinstance(
        criterion(preds._replace(logits=preds.logits[:1]), targets), obb.Err
    )
    assert isinstance(
        obb.SetHungarianMatcher(obb.MatcherConfig(max_pairs=1))(preds, targets), obb.Err
    )
    assert isinstance(
        criterion(
            preds,
            targets._replace(
                boxes=targets.boxes._replace(
                    values=torch.tensor(0.0, device=preds.boxes.values.device)
                )
            ),
        ),
        obb.Err,
    )


def test_seeded_assignment_reference(device: torch.device) -> None:
    """Sixteen seeded rectangular problems agree with exhaustive scalar optima."""
    generator = torch.Generator(device=device).manual_seed(901)
    matcher = obb.SetHungarianMatcher(
        obb.MatcherConfig(
            classification_weight=2, l1_weight=3, probiou_weight=0, chunk_size=2
        )
    )
    for _ in range(16):
        first = torch.rand(
            3, 6, generator=generator, device=device, dtype=torch.float64
        )
        second = torch.rand(
            2, 6, generator=generator, device=device, dtype=torch.float64
        )
        first[:, 2:4] += 0.1
        second[:, 2:4] += 0.1
        logits = torch.randn(
            3, 2, generator=generator, device=device, dtype=torch.float64
        )
        preds = obb.Predictions(
            obb.PackedBatch(first, torch.tensor([0, 3], device=device)), logits
        )
        targets = obb.Targets(
            obb.PackedBatch(second, torch.tensor([0, 2], device=device)),
            torch.tensor([0, 1], device=device),
        )
        reference = first.new_tensor(
            [
                -2 / (1 + math.exp(-logits[row, col].item()))
                + 3
                * sum(
                    abs(first[row, k].item() - second[col, k].item()) for k in range(4)
                )
                for row in range(3)
                for col in range(2)
            ]
        )
        costs = matcher.pairwise_costs(preds, targets)
        result = matcher(preds, targets)
        assert isinstance(costs, obb.Ok)
        assert isinstance(result, obb.Ok)
        torch.testing.assert_close(costs.value.values, reference)
        expected = min(
            sum(reference[rows[col] * 2 + col].item() for col in range(2))
            for rows in permutations(range(3), 2)
        )
        actual = sum(
            reference[row * 2 + col].item()
            for row, col in zip(
                result.value.pred_indices.tolist(),
                result.value.target_indices.tolist(),
                strict=True,
            )
        )
        assert actual == pytest.approx(expected)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_set_mixed_precision(device: torch.device, dtype: torch.dtype) -> None:
    """Mixed-precision leaves receive finite gradients from internally float64 losses."""
    boxes = torch.tensor(
        [[[0.1, 0, 2, 1, 1, 0]]], device=device, dtype=dtype, requires_grad=True
    )
    logits = torch.tensor(
        [[[0.3, -0.8]]], device=device, dtype=dtype, requires_grad=True
    )
    preds = obb.pack_predictions(boxes, logits)
    targets = obb.pack_targets(
        [
            obb.TargetSample(
                torch.tensor([[0.0, 0, 2, 1, 1, 0]], device=device, dtype=dtype),
                torch.tensor([0], device=device),
            )
        ],
        device=device,
        dtype=dtype,
    )
    assert isinstance(preds, obb.Ok)
    assert isinstance(targets, obb.Ok)
    with torch.autocast(
        device.type, dtype=torch.bfloat16 if device.type == "cpu" else torch.float16
    ):
        result = QualityCriterion()(preds.value, targets.value)
        assert isinstance(result, obb.Ok)
        assert result.value.total.dtype == torch.float64
        result.value.total.backward()
    assert boxes.grad is not None
    assert logits.grad is not None
    assert bool(torch.isfinite(boxes.grad).all())
    assert bool(torch.isfinite(logits.grad).all())


def test_empty_set_adapters_and_field_rejections() -> None:
    """Empty annotated packs remain valid and malformed labels stay typed errors."""
    preds = obb.pack_predictions(torch.empty(0, 2, 6), torch.empty(0, 2, 3))
    targets = obb.pack_targets([], device=torch.device("cpu"), dtype=torch.float32)
    assert isinstance(preds, obb.Ok)
    assert isinstance(targets, obb.Ok)
    result = QualityCriterion()(preds.value, targets.value)
    assert isinstance(result, obb.Ok)
    assert result.value.total.item() == 0
    boxes = torch.tensor([[0.0, 0, 2, 1, 1, 0]])
    for labels in (
        torch.tensor([0.0]),
        torch.tensor([-1]),
        torch.empty(0, dtype=torch.int64),
    ):
        assert isinstance(
            obb.pack_targets(
                [obb.TargetSample(boxes, labels)],
                device=boxes.device,
                dtype=boxes.dtype,
            ),
            obb.Err,
        )
