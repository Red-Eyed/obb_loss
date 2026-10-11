"""Subclass extension contracts without callbacks or composition containers."""

from collections.abc import Mapping

import pytest
import torch
from torch import Tensor, nn
from typing_extensions import override

import obb_loss as obb


class CustomCriterion(obb.SetCriterion):
    """Register a trainable custom regularizer using ordinary Module ownership."""

    def __init__(self) -> None:
        """Disable built-ins and register the custom scale as a parameter."""
        super().__init__(obb.SetCriterionConfig(classification="none", normalizer=1.0))
        self.scale = nn.Parameter(torch.tensor(2.0))
        self.register_buffer("offset", torch.tensor(1.0))

    @override
    def compute_losses(
        self,
        preds: obb.Predictions,
        targets: obb.Targets,
        assignment: obb.Assignment,
    ) -> obb.Result[Mapping[str, Tensor], obb.InputError]:
        """Penalize all query directions, including unmatched predictions."""
        residual = preds.boxes.values[:, 4:6].double().square().sum(-1) - self.offset
        return obb.Ok(
            {
                "direction": self.scale
                * obb.sum_samples(residual.square(), preds.boxes.cu_seqlens)
            }
        )


def test_custom_term_registration_and_unmatched_gradient(device: torch.device) -> None:
    """A subclass owns parameters/buffers and supervises unmatched queries."""
    boxes = torch.tensor(
        [[0, 0, 2, 1, 1, 0], [4, 0, 2, 1, 2, 0]],
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )
    preds = obb.Predictions(
        obb.PackedBatch(boxes, torch.tensor([0, 2], device=device)),
        torch.zeros(2, 1, device=device, requires_grad=True),
    )
    targets = obb.Targets(
        obb.PackedBatch(boxes[:1].detach(), torch.tensor([0, 1], device=device)),
        torch.tensor([0], device=device),
    )
    criterion = CustomCriterion().to(device)
    assert "scale" in dict(criterion.named_parameters())
    assert "offset" in criterion.state_dict()
    result = criterion(preds, targets, reduction=obb.Reduction.NONE)
    assert isinstance(result, obb.Ok)
    torch.testing.assert_close(result.value.total, boxes.new_tensor([18.0]))
    result.value.total.sum().backward()
    assert boxes.grad is not None
    assert boxes.grad[1, 4].item() == 48
    assert criterion.scale.grad is not None
    assert criterion.scale.grad.item() == 9


@pytest.mark.parametrize("invalid", ["shape", "dtype", "finite", "device"])
def test_invalid_custom_outputs_are_typed(device: torch.device, invalid: str) -> None:
    """Reject malformed subclass loss vectors at the forward boundary."""
    boxes = torch.tensor([[0.0, 0, 2, 1, 1, 0]], device=device)
    preds = obb.Predictions(
        obb.PackedBatch(boxes, torch.tensor([0, 1], device=device)),
        torch.zeros(1, 1, device=device),
    )
    targets = obb.Targets(
        obb.PackedBatch(boxes, preds.boxes.cu_seqlens), torch.tensor([0], device=device)
    )
    outputs = {
        "shape": torch.zeros(2, device=device),
        "dtype": torch.ones(1, device=device, dtype=torch.int64),
        "finite": torch.full((1,), float("inf"), device=device),
        "device": torch.zeros(1, device="meta"),
    }

    class InvalidCriterion(obb.SetCriterion):
        """Return a deliberately malformed named vector."""

        @override
        def compute_losses(
            self,
            preds: obb.Predictions,
            targets: obb.Targets,
            assignment: obb.Assignment,
        ) -> obb.Result[Mapping[str, Tensor], obb.InputError]:
            """Exercise output validation independently of built-in formulas."""
            return obb.Ok({"custom": outputs[invalid]})

    result = InvalidCriterion()(preds, targets)
    assert isinstance(result, obb.Err)
    assert result.error.field == "custom"


def test_custom_matcher_cost_chunks_and_gradients(device: torch.device) -> None:
    """An overridden cost stays within samples and retains direct-call gradients."""
    boxes = torch.tensor(
        [[0.0, 0, 2, 1, 1, 0], [2, 0, 2, 1, 1, 0], [9, 0, 2, 1, 1, 0]],
        device=device,
        requires_grad=True,
    )
    preds = obb.Predictions(
        obb.PackedBatch(boxes, torch.tensor([0, 2, 3], device=device)),
        torch.zeros(3, 1, device=device),
    )
    targets = obb.Targets(
        obb.PackedBatch(boxes[[0, 2]].detach(), torch.tensor([0, 1, 2], device=device)),
        torch.zeros(2, dtype=torch.int64, device=device),
    )
    observed: list[tuple[int, int]] = []

    class CenterMatcher(obb.SetHungarianMatcher):
        """Prefer the largest query center within each sample."""

        @override
        def cost(
            self,
            preds: obb.Predictions,
            targets: obb.Targets,
            pred_indices: Tensor,
            target_indices: Tensor,
        ) -> obb.Result[Tensor, obb.InputError]:
            """Record bounded pair indices and return a differentiable query cost."""
            assert pred_indices.numel() == 1
            observed.append((int(pred_indices[0]), int(target_indices[0])))
            return obb.Ok(-preds.boxes.values[pred_indices, 0].double())

    matcher = CenterMatcher(obb.MatcherConfig(chunk_size=1))
    costs = matcher.pairwise_costs(preds, targets)
    assert isinstance(costs, obb.Ok)
    assert observed == [(0, 0), (1, 0), (2, 1)]
    costs.value.values.sum().backward()
    assert boxes.grad is not None
    torch.testing.assert_close(boxes.grad[:, 0], boxes.new_full((3,), -1))
    result = matcher(preds, targets)
    assert isinstance(result, obb.Ok)
    assert result.value.pred_indices.tolist() == [1, 2]
    assert result.value.target_indices.tolist() == [0, 1]
