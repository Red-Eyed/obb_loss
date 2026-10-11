"""Packed matching optimality, segment isolation, and chunk memory contracts."""

from itertools import permutations

import pytest
import torch
import torch.testing
from einops import rearrange
from torch import Tensor

from obb_loss.adapters import pack_boxes
from obb_loss.config import MatcherConfig
from obb_loss.criterion import ProbIoULoss
from obb_loss.losses.stable_probiou import stable_probiou_loss
from obb_loss.matching import HungarianMatcher, gather_matched
from obb_loss.packed import PackedBoxes
from obb_loss.pairwise import pairwise_costs
from obb_loss.result import Err, Ok


@pytest.fixture
def matching_pair(device: torch.device) -> tuple[PackedBoxes, PackedBoxes]:
    """Create irregular two-sample geometry with one empty target sample."""
    first = torch.tensor(
        [
            [0, 0, 2, 1, 1, 0],
            [2, 0, 2, 1, 1, 0],
            [5, 0, 2, 1, 1, 0],
            [9, 0, 2, 1, 1, 0],
        ],
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )
    second = torch.tensor(
        [[2.1, 0, 2, 1, 1, 0], [0.1, 0, 2, 1, 1, 0]], device=device, dtype=torch.float64
    )
    return PackedBoxes(first, torch.tensor([0, 3, 4], device=device)), PackedBoxes(
        second, torch.tensor([0, 2, 2], device=device)
    )


def test_assignment_brute_force_and_gradient_gather(
    matching_pair: tuple[PackedBoxes, PackedBoxes],
) -> None:
    """Optimal cost equals exhaustive rectangular assignment; gather retains autograd."""
    preds, targets = matching_pair
    outcome = HungarianMatcher(MatcherConfig(chunk_size=2)).forward(preds, targets)
    assert isinstance(outcome, Ok)
    assignment = outcome.value
    torch.testing.assert_close(
        assignment.cu_seqlens, torch.tensor([0, 2, 2], device=preds.boxes.device)
    )
    costs = stable_probiou_loss(
        rearrange(preds.boxes[:3], "n c -> n 1 c"),
        rearrange(targets.boxes, "m c -> 1 m c"),
    )
    optimal = min(
        sum(costs[row, column].item() for column, row in enumerate(rows))
        for rows in permutations(range(3), 2)
    )
    assigned_cost = (
        costs[assignment.pred_indices, assignment.target_indices].sum().item()
    )
    assert assigned_cost == pytest.approx(optimal, abs=1e-14)
    assert assignment.pred_indices.unique().numel() == 2
    assert assignment.target_indices.unique().numel() == 2
    assert not assignment.pred_indices.requires_grad
    aligned = gather_matched(preds, targets, assignment)
    ProbIoULoss().forward(*aligned).backward()
    assert preds.boxes.grad is not None
    assert torch.isfinite(preds.boxes.grad).all()
    torch.testing.assert_close(
        preds.boxes.grad[3],
        torch.zeros(6, device=preds.boxes.device, dtype=torch.float64),
    )


def test_chunk_limit_and_cost_reference(
    matching_pair: tuple[PackedBoxes, PackedBoxes],
) -> None:
    """No callback invocation exceeds chunk_size; packed costs match dense oracle."""
    preds, targets = matching_pair
    sizes: list[int] = []

    def cost(first: Tensor, second: Tensor) -> Tensor:
        """Observe allocated pair count while using the stable mathematical cost."""
        sizes.append(first.shape[0])
        return stable_probiou_loss(first, second)

    outcome = pairwise_costs(preds, targets, MatcherConfig(chunk_size=2), cost)
    assert isinstance(outcome, Ok)
    assert sizes == [2, 2, 2]
    assert outcome.value.blocks[1].start == outcome.value.blocks[1].end
    reference = stable_probiou_loss(
        rearrange(preds.boxes[:3], "n c -> n 1 c"),
        rearrange(targets.boxes, "m c -> 1 m c"),
    )
    torch.testing.assert_close(
        outcome.value.values, rearrange(reference, "n m -> (n m)")
    )
    assert isinstance(pairwise_costs(preds, targets, MatcherConfig(max_pairs=5)), Err)


def test_more_targets_than_queries(
    matching_pair: tuple[PackedBoxes, PackedBoxes],
) -> None:
    """Rectangular matching assigns the smaller side without dropping sample identity."""
    first, second = matching_pair
    outcome = HungarianMatcher().forward(second, first)
    assert isinstance(outcome, Ok)
    assert outcome.value.pred_indices.numel() == 2


def test_all_empty_and_zero_samples(device: torch.device) -> None:
    """Completely empty input supports both zero and multiple samples."""
    for samples in (0, 3):
        outcome = pack_boxes(
            [torch.empty((0, 6), device=device)] * samples,
            device=device,
            dtype=torch.float32,
        )
        assert isinstance(outcome, Ok)
        result = HungarianMatcher().forward(outcome.value, outcome.value)
        assert isinstance(result, Ok)
        assert result.value.pred_indices.numel() == 0
        assert result.value.cu_seqlens.shape == (samples + 1,)


def test_sample_isolation(device: torch.device) -> None:
    """Never select a geometrically closer box from another sample."""
    preds = torch.tensor(
        [[0, 0, 2, 1, 1, 0], [10, 0, 2, 1, 1, 0]], device=device, dtype=torch.float64
    )
    targets = preds.flip(0)
    cu = torch.tensor([0, 1, 2], device=device)
    result = HungarianMatcher().forward(
        PackedBoxes(preds, cu), PackedBoxes(targets, cu)
    )
    assert isinstance(result, Ok)
    torch.testing.assert_close(result.value.pred_indices, result.value.target_indices)


def test_callback_shape_rejection(
    matching_pair: tuple[PackedBoxes, PackedBoxes],
) -> None:
    """A typed callable's invalid runtime shape is a recoverable contract failure."""

    def wrong_shape(first: Tensor, second: Tensor) -> Tensor:
        """Return an invalid scalar to exercise checked callback admission."""
        return (first.sum() + second.sum()) * 0

    result = HungarianMatcher(cost=wrong_shape).forward(*matching_pair)
    assert isinstance(result, Err)
