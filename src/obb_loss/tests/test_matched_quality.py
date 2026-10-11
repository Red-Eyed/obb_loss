"""Matched classification quality ownership, validation and layer-specific policy."""

from enum import Enum

import pytest
import torch
import torch.testing
from torch import Tensor
from typing_extensions import override

import obb_loss as obb


class InvalidQuality(Enum):
    """Represent independently invalid matched quality inputs."""

    SHAPE = "shape"
    DTYPE = "dtype"
    DEVICE = "device"
    RANGE = "range"
    FINITE = "finite"


@pytest.fixture
def quality_sets(
    device: torch.device,
) -> tuple[obb.Predictions, obb.Targets]:
    """Two queries match reversed target order, leaving one reference unmatched."""
    boxes = torch.tensor(
        [[2, 0, 2, 1, 1, 0], [0, 0, 2, 1, 1, 0]],
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )
    logits = torch.zeros(2, 2, device=device, dtype=torch.float64, requires_grad=True)
    targets = torch.tensor(
        [[0, 0, 2, 1, 1, 0], [2, 0, 2, 1, 1, 0], [10, 0, 2, 1, 1, 0]],
        device=device,
        dtype=torch.float64,
    )
    return (
        obb.Predictions(
            obb.PackedBatch(boxes, torch.tensor([0, 2], device=device)), logits
        ),
        obb.Targets(
            obb.PackedBatch(targets, torch.tensor([0, 3], device=device)),
            torch.tensor([1, 0, 1], device=device),
        ),
    )


class ScoreCriterion(obb.SetCriterion):
    """Bind caller-owned scores to a matched classification policy."""

    def __init__(self, scores: Tensor, *, qfl: bool = False) -> None:
        """Select classification and a geometry-only assignment oracle."""
        super().__init__(
            obb.SetCriterionConfig(
                classification="qfl" if qfl else "vfl",
                l1_weight=0,
                probiou_weight=0,
                normalizer=1.0,
                matcher=obb.MatcherConfig(
                    classification_weight=0, l1_weight=1, probiou_weight=0
                ),
            )
        )
        self.scores = scores

    @override
    def classification_quality(
        self,
        preds: obb.Predictions,
        targets: obb.Targets,
        assignment: obb.Assignment,
    ) -> obb.Result[Tensor, obb.InputError]:
        """Return supplied scores in assignment order."""
        assert assignment.pred_indices.tolist() == [0, 1]
        assert assignment.target_indices.tolist() == [1, 0]
        return obb.Ok(self.scores)


@pytest.mark.parametrize("qfl", [False, True])
def test_quality_follows_pairs_not_targets(
    quality_sets: tuple[obb.Predictions, obb.Targets], qfl: bool
) -> None:
    """Scatter pair quality to matched classes and preserve score gradients."""
    preds, targets = quality_sets
    scores = preds.logits.new_tensor([0.2, 0.9]).requires_grad_()
    result = ScoreCriterion(scores, qfl=qfl)(preds, targets)
    assert isinstance(result, obb.Ok)
    dense = scores.new_tensor([[0.2, 0], [0, 0.9]])
    expected = (
        obb.quality_focal_loss(preds.logits, dense)
        if qfl
        else obb.varifocal_loss(preds.logits, dense)
    )
    torch.testing.assert_close(result.value.total, expected.sum())
    result.value.total.backward()
    assert scores.grad is not None
    assert bool(torch.isfinite(scores.grad).all())


@pytest.mark.parametrize("invalid", tuple(InvalidQuality))
@pytest.mark.parametrize("qfl", [False, True])
def test_invalid_pair_quality_is_typed(
    quality_sets: tuple[obb.Predictions, obb.Targets],
    invalid: InvalidQuality,
    qfl: bool,
) -> None:
    """Reject incorrect pair count, dtype, device, range and finiteness."""
    preds, targets = quality_sets
    match invalid:
        case InvalidQuality.SHAPE:
            scores = preds.logits.new_full((3,), 0.5)
        case InvalidQuality.DTYPE:
            scores = torch.ones(2, device=preds.logits.device, dtype=torch.int64)
        case InvalidQuality.DEVICE:
            scores = torch.ones(2, device="meta")
        case InvalidQuality.RANGE:
            scores = preds.logits.new_tensor([0.5, 1.1])
        case InvalidQuality.FINITE:
            scores = preds.logits.new_tensor([0.5, float("nan")])
    result = ScoreCriterion(scores, qfl=qfl)(preds, targets)
    assert isinstance(result, obb.Err)


@pytest.mark.parametrize("detach", [False, True])
def test_quality_detachment_is_caller_policy(
    quality_sets: tuple[obb.Predictions, obb.Targets], detach: bool
) -> None:
    """Preserve supplied gradients without undoing caller detachment."""
    preds, targets = quality_sets
    scores = preds.logits.new_full((2,), 0.7).requires_grad_()
    result = ScoreCriterion(scores.detach() if detach else scores)(preds, targets)
    assert isinstance(result, obb.Ok)
    result.value.total.backward()
    assert (scores.grad is None) == detach
    assert preds.logits.grad is not None
