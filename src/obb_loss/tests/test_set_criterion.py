"""Per-layer set matching, normalization, component weights, and gradient contracts."""

import math
from itertools import permutations

import pytest
import torch
import torch.autograd
import torch.testing
from torch import Tensor

import obb_loss as obb
import obb_loss.set_criterion as criterion_module
import obb_loss.set_matching as matching_module


@pytest.fixture
def sets(device: torch.device) -> tuple[obb.PackedPredictions, obb.PackedTargets]:
    """Two queries prefer different targets under geometry and classification costs."""
    boxes = torch.tensor(
        [[[0.2, 0.0, 2.0, 1.0, 1.0, 0.0], [0.8, 0.0, 2.0, 1.0, 1.0, 0.0]]],
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )
    logits = torch.tensor(
        [[[4.0, -4.0], [-4.0, 4.0]]],
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )
    preds = obb.pack_predictions(boxes, logits)
    targets = obb.pack_targets(
        [
            obb.TargetSample(
                torch.tensor(
                    [[0.0, 0.0, 2.0, 1.0, 1.0, 0.0]], device=device, dtype=torch.float64
                ),
                torch.tensor([1], device=device),
                torch.tensor(
                    [0.7], device=device, dtype=torch.float64, requires_grad=True
                ),
            )
        ],
        device=device,
        dtype=torch.float64,
    )
    assert isinstance(preds, obb.Ok)
    assert isinstance(targets, obb.Ok)
    preds.value.geometry.boxes.retain_grad()
    preds.value.logits.retain_grad()
    targets.value.quality.retain_grad()
    return preds.value, targets.value


def test_l1_math_and_gradients(device: torch.device) -> None:
    """L1 sums center/side errors, ignores direction, and has correct double gradients."""
    first = torch.tensor(
        [[0.2, -0.4, 2.0, 1.0, 0.0, 1.0]],
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )
    second = torch.tensor(
        [[0.0, 0.0, 1.0, 2.0, 1.0, 0.0]],
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )
    torch.testing.assert_close(
        obb.box_l1_loss(first, second),
        torch.tensor([2.6], device=device, dtype=torch.float64),
    )
    assert torch.autograd.gradcheck(obb.box_l1_loss, (first, second))
    obb.box_l1_loss(first, second).sum().backward()
    assert first.grad is not None
    assert bool((first.grad[:, 4:] == 0).all())


def test_weighted_cost_and_assignment(
    sets: tuple[obb.PackedPredictions, obb.PackedTargets],
) -> None:
    """Match the independent analytic weighted cost and choose its minimum."""
    preds, targets = sets
    config = obb.SetMatcherConfig(
        weights=obb.SetWeights(classification=10.0, l1=2.0, probiou=0.0),
        storage=obb.MatcherConfig(chunk_size=1),
    )
    costs = obb.set_pairwise_costs(preds, targets, config)
    assert isinstance(costs, obb.Ok)
    expected = torch.tensor(
        [-10 / (1 + math.exp(4)) + 0.4, -10 / (1 + math.exp(-4)) + 1.6],
        device=preds.logits.device,
        dtype=torch.float64,
    )
    torch.testing.assert_close(costs.value.values, expected)
    assignment = obb.SetHungarianMatcher(config).forward(preds, targets)
    assert isinstance(assignment, obb.Ok)
    assert assignment.value.pred_indices.tolist() == [1]
    assert not assignment.value.pred_indices.requires_grad


@pytest.mark.parametrize("classification", [obb.VFLConfig(), obb.QFLConfig()])
def test_component_losses_and_backward(
    sets: tuple[obb.PackedPredictions, obb.PackedTargets],
    classification: obb.VFLConfig | obb.QFLConfig,
) -> None:
    """The weighted total matches plain losses and preserves caller quality gradients."""
    preds, targets = sets
    config = obb.SetCriterionConfig(
        weights=obb.SetWeights(classification=3.0, l1=4.0, probiou=2.0),
        matcher=obb.SetMatcherConfig(
            weights=obb.SetWeights(classification=0.0, l1=1.0, probiou=0.0)
        ),
        classification=classification,
    )
    result = obb.SetCriterion(config).forward(preds, targets, normalizer=2.0)
    assert isinstance(result, obb.Ok)
    losses = result.value
    torch.testing.assert_close(
        losses.l1, torch.tensor(0.1, device=preds.logits.device, dtype=torch.float64)
    )
    first, second = obb.gather_matched(
        preds.geometry, targets.geometry, losses.assignment
    )
    torch.testing.assert_close(
        losses.probiou, obb.stable_probiou_loss(first.boxes, second.boxes).sum() / 2
    )
    quality = torch.zeros_like(preds.logits)
    quality[0, 1] = 0.7
    match classification:
        case obb.VFLConfig():
            expected = obb.varifocal_loss(preds.logits, quality).sum() / 2
        case obb.QFLConfig():
            expected = obb.quality_focal_loss(preds.logits, quality).sum() / 2
    torch.testing.assert_close(losses.classification, expected)
    torch.testing.assert_close(
        losses.total, 3 * expected + 4 * losses.l1 + 2 * losses.probiou
    )
    losses.total.backward()
    assert preds.geometry.boxes.grad is not None
    assert preds.logits.grad is not None
    assert targets.quality.grad is not None
    assert bool(torch.isfinite(targets.quality.grad).all())
    assert bool((preds.geometry.boxes.grad[1] == 0).all())
    assert bool((preds.logits.grad[1] != 0).all())


def test_loss_weights_do_not_change_matching(
    sets: tuple[obb.PackedPredictions, obb.PackedTargets],
) -> None:
    """Loss weights change totals while matching weights independently change assignments."""
    preds, targets = sets
    geometrical = obb.SetMatcherConfig(
        weights=obb.SetWeights(classification=0.0, l1=1.0, probiou=0.0)
    )
    first = obb.SetCriterion(obb.SetCriterionConfig(matcher=geometrical)).forward(
        preds, targets
    )
    second = obb.SetCriterion(
        obb.SetCriterionConfig(
            matcher=geometrical,
            weights=obb.SetWeights(classification=9, l1=0, probiou=0),
        )
    ).forward(preds, targets)
    third = obb.SetCriterion(
        obb.SetCriterionConfig(
            matcher=obb.SetMatcherConfig(
                weights=obb.SetWeights(classification=10, l1=1, probiou=0)
            )
        )
    ).forward(preds, targets)
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


def test_disabled_terms_skip_functions(
    sets: tuple[obb.PackedPredictions, obb.PackedTargets],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disabled matching and loss terms never evaluate their functions."""

    def forbidden(first: Tensor, second: Tensor) -> Tensor:
        """Fail if an omitted numerical term runs."""
        pytest.fail("disabled term evaluated")

    for module in (criterion_module, matching_module):
        monkeypatch.setattr(module, "box_l1_loss", forbidden)
        monkeypatch.setattr(module, "stable_probiou_loss", forbidden)
    monkeypatch.setattr(criterion_module, "_classification_loss", forbidden)
    config = obb.SetCriterionConfig(
        weights=obb.SetWeights(classification=0, l1=0, probiou=0),
        matcher=obb.SetMatcherConfig(
            weights=obb.SetWeights(classification=1, l1=0, probiou=0)
        ),
    )
    preds, targets = sets
    result = obb.SetCriterion(config).forward(preds, targets)
    assert isinstance(result, obb.Ok)
    assert result.value.total.item() == 0
    result.value.total.backward()
    assert preds.logits.grad is not None
    assert bool((preds.logits.grad == 0).all())


@pytest.mark.parametrize("normalizer", [0.0, -1.0, math.nan, math.inf])
def test_reject_normalizer(
    sets: tuple[obb.PackedPredictions, obb.PackedTargets], normalizer: float
) -> None:
    """Reject invalid normalization rather than silently creating nonfinite losses."""
    assert isinstance(obb.SetCriterion().forward(*sets, normalizer=normalizer), obb.Err)


def test_set_config_validation() -> None:
    """Configs reject ambiguous double weighting and all-disabled matching costs."""
    assert isinstance(
        obb.parse_config(
            obb.SetMatcherConfig,
            {"weights": {"classification": 0, "l1": 0, "probiou": 0}},
        ),
        obb.Err,
    )
    assert isinstance(
        obb.parse_config(
            obb.SetCriterionConfig, {"classification": {"kind": "vfl", "weight": 2}}
        ),
        obb.Err,
    )
    config = obb.SetCriterionConfig(classification=obb.QFLConfig())
    assert (
        obb.SetCriterionConfig.model_validate_json(config.model_dump_json()) == config
    )


def test_empty_samples_and_isolation(device: torch.device) -> None:
    """Empty sample boundaries survive and classification stays active with no targets."""
    boxes = torch.tensor(
        [[[0.1, 0, 2, 1, 1, 0]], [[9.0, 0, 2, 1, 1, 0]]],
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    logits = torch.zeros(2, 1, 2, device=device, requires_grad=True)
    preds = obb.pack_predictions(boxes, logits)
    samples = [
        obb.TargetSample(
            torch.empty(0, 6, device=device),
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, device=device),
        )
        for _ in range(2)
    ]
    targets = obb.pack_targets(samples, device=device, dtype=torch.float32)
    assert isinstance(preds, obb.Ok)
    assert isinstance(targets, obb.Ok)
    result = obb.SetCriterion().forward(preds.value, targets.value)
    assert isinstance(result, obb.Ok)
    assert result.value.assignment.cu_seqlens.tolist() == [0, 0, 0]
    assert result.value.l1.item() == result.value.probiou.item() == 0
    assert result.value.classification.item() > 0
    result.value.total.backward()
    assert boxes.grad is not None
    assert bool((boxes.grad == 0).all())
    assert logits.grad is not None
    assert bool(torch.isfinite(logits.grad).all())
    labeled = obb.pack_targets(
        [
            samples[0],
            obb.TargetSample(
                torch.tensor([[0.0, 0, 2, 1, 1, 0]], device=device),
                torch.tensor([1], device=device),
                torch.tensor([0.8], device=device),
            ),
        ],
        device=device,
        dtype=torch.float32,
    )
    assert isinstance(labeled, obb.Ok)
    result = obb.SetCriterion().forward(preds.value, labeled.value)
    assert isinstance(result, obb.Ok)
    assert result.value.assignment.pred_indices.tolist() == [1]
    assert result.value.assignment.cu_seqlens.tolist() == [0, 0, 1]


def test_revalidate_fields_and_storage(
    sets: tuple[obb.PackedPredictions, obb.PackedTargets],
) -> None:
    """Fresh calls reject mutated labels, quality, layouts, and oversized cost storage."""
    preds, targets = sets
    invalid = obb.PackedTargets(targets.geometry, targets.labels + 2, targets.quality)
    assert isinstance(obb.SetCriterion().forward(preds, invalid), obb.Err)
    invalid = obb.PackedTargets(targets.geometry, targets.labels, targets.quality + 1)
    assert isinstance(obb.SetCriterion().forward(preds, invalid), obb.Err)
    config = obb.SetMatcherConfig(storage=obb.MatcherConfig(max_pairs=1))
    assert isinstance(obb.SetHungarianMatcher(config).forward(preds, targets), obb.Err)
    bad = obb.PackedPredictions(preds.geometry, preds.logits[:1])
    assert isinstance(obb.SetCriterion().forward(bad, targets), obb.Err)
    malformed = obb.PackedTargets(
        obb.PackedBoxes(
            torch.tensor(0.0, device=preds.logits.device), targets.geometry.cu_seqlens
        ),
        targets.labels,
        targets.quality,
    )
    assert isinstance(obb.SetCriterion().forward(preds, malformed), obb.Err)


def test_seeded_assignment_reference(device: torch.device) -> None:
    """Seeded rectangular costs agree with a scalar oracle and brute-force assignments."""
    generator = torch.Generator(device=device).manual_seed(901)
    config = obb.SetMatcherConfig(
        weights=obb.SetWeights(classification=2.0, l1=3.0, probiou=0.0),
        storage=obb.MatcherConfig(chunk_size=2),
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
        preds = obb.PackedPredictions(
            obb.PackedBoxes(first, torch.tensor([0, 3], device=device)), logits
        )
        targets = obb.PackedTargets(
            obb.PackedBoxes(second, torch.tensor([0, 2], device=device)),
            torch.tensor([0, 1], device=device),
            torch.tensor([0.6, 0.9], device=device),
        )
        reference = torch.tensor(
            [
                -2 / (1 + math.exp(-logits[row, column].item()))
                + 3
                * sum(
                    abs(
                        first[row, coordinate].item()
                        - second[column, coordinate].item()
                    )
                    for coordinate in range(4)
                )
                for row in range(3)
                for column in range(2)
            ],
            device=device,
            dtype=torch.float64,
        )
        costs = obb.set_pairwise_costs(preds, targets, config)
        assert isinstance(costs, obb.Ok)
        torch.testing.assert_close(costs.value.values, reference)
        result = obb.SetHungarianMatcher(config).forward(preds, targets)
        assert isinstance(result, obb.Ok)
        expected = min(
            sum(reference[rows[column] * 2 + column].item() for column in range(2))
            for rows in permutations(range(3), 2)
        )
        actual = sum(
            reference[row * 2 + column].item()
            for row, column in zip(
                result.value.pred_indices.tolist(),
                result.value.target_indices.tolist(),
                strict=True,
            )
        )
        assert actual == pytest.approx(expected)
        changed_quality = obb.PackedTargets(
            targets.geometry, targets.labels, 1 - targets.quality
        )
        unchanged = obb.set_pairwise_costs(preds, changed_quality, config)
        assert isinstance(unchanged, obb.Ok)
        torch.testing.assert_close(unchanged.value.values, costs.value.values)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_set_mixed_precision(device: torch.device, dtype: torch.dtype) -> None:
    """Float64 losses and finite leaf gradients survive outer autocast for each dtype."""
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
                torch.tensor([0.7], device=device, dtype=dtype),
            ),
        ],
        device=device,
        dtype=dtype,
    )
    assert isinstance(preds, obb.Ok)
    assert isinstance(targets, obb.Ok)
    autocast_dtype = torch.bfloat16 if device.type == "cpu" else torch.float16
    with torch.autocast(device.type, dtype=autocast_dtype):
        result = obb.SetCriterion().forward(preds.value, targets.value)
        assert isinstance(result, obb.Ok)
        assert result.value.total.dtype == torch.float64
        result.value.total.backward()
    assert boxes.grad is not None
    assert logits.grad is not None
    assert bool(torch.isfinite(boxes.grad).all())
    assert bool(torch.isfinite(logits.grad).all())


def test_empty_set_adapters_and_field_rejections() -> None:
    """Zero samples, zero queries, and malformed target fields have explicit outcomes."""
    preds = obb.pack_predictions(torch.empty(0, 2, 6), torch.empty(0, 2, 3))
    targets = obb.pack_targets([], device=torch.device("cpu"), dtype=torch.float32)
    assert isinstance(preds, obb.Ok)
    assert isinstance(targets, obb.Ok)
    result = obb.SetCriterion().forward(preds.value, targets.value)
    assert isinstance(result, obb.Ok)
    assert result.value.total.item() == 0
    assert result.value.assignment.cu_seqlens.tolist() == [0]
    boxes = torch.tensor([[0.0, 0, 2, 1, 1, 0]])
    fields = (
        obb.TargetSample(boxes, torch.tensor([0.0]), torch.tensor([0.7])),
        obb.TargetSample(boxes, torch.tensor([-1]), torch.tensor([0.7])),
        obb.TargetSample(boxes, torch.tensor([0]), torch.tensor([math.nan])),
        obb.TargetSample(boxes, torch.empty(0, dtype=torch.int64), torch.empty(0)),
    )
    for sample in fields:
        assert isinstance(
            obb.pack_targets([sample], device=boxes.device, dtype=boxes.dtype), obb.Err
        )
    assert isinstance(obb.pack_predictions(boxes, torch.empty(1, 1, 2)), obb.Err)
