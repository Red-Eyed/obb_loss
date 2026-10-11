"""Custom callback weighting, state ownership, sample gradients, and typed failures."""

from enum import Enum

import pytest
import torch
import torch.testing
from torch import Tensor, nn
from typing_extensions import override

import obb_loss as obb
from obb_loss.packed import ErrorKind


@pytest.fixture
def composed_sets(
    device: torch.device,
) -> tuple[obb.PackedPredictions, obb.PackedTargets]:
    """Two samples include an unmatched nonunit direction and an empty target segment."""
    boxes = torch.tensor(
        [[0, 0, 2, 1, 2, 0], [1, 0, 2, 1, 0, 0.5], [0, 0, 2, 1, 1, 0]],
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )
    logits = torch.zeros(3, 2, device=device, dtype=torch.float64, requires_grad=True)
    preds = obb.PackedPredictions(
        obb.PackedBoxes(boxes, torch.tensor([0, 2, 3], device=device)), logits
    )
    targets = obb.PackedTargets(
        obb.PackedBoxes(
            torch.tensor([[0, 0, 2, 1, 1, 0]], device=device, dtype=torch.float64),
            torch.tensor([0, 1, 1], device=device),
        ),
        torch.tensor([0], device=device),
        torch.tensor([0.8], device=device),
    )
    return preds, targets


def _direction(context: obb.SetLossContext) -> obb.Result[Tensor, obb.InputError]:
    """Compute unweighted raw direction-constraint sums over all packed queries."""
    values = (
        context.preds.geometry.boxes[:, 4:6].double().square().sum(-1) - 1
    ).square()
    return obb.Ok(obb.sum_samples(values, context.preds.geometry.cu_seqlens))


def test_custom_direction_math_and_gradients(
    composed_sets: tuple[obb.PackedPredictions, obb.PackedTargets],
) -> None:
    """Weighting preserves analytic values and reaches unmatched queries in one sample."""
    preds, targets = composed_sets
    criterion = obb.SetCriterion(
        losses={"direction": obb.WeightedLoss(_direction, obb.WeightConfig(weight=2))}
    )
    result = criterion(preds, targets, reduction=obb.Reduction.NONE, normalizer=1.0)
    assert isinstance(result, obb.Ok)
    torch.testing.assert_close(
        result.value.total,
        torch.tensor([19.125, 0], device=preds.logits.device, dtype=torch.float64),
    )
    assert tuple(result.value.terms) == ("direction",)
    assert result.value.assignment.pred_indices.tolist() == [0]
    result.value.total[0].backward()
    assert preds.geometry.boxes.grad is not None
    expected = torch.zeros_like(preds.geometry.boxes)
    expected[0, 4], expected[1, 5] = 48, -3
    torch.testing.assert_close(preds.geometry.boxes.grad, expected)
    assert preds.logits.grad is not None
    assert bool((preds.logits.grad == 0).all())


class TrainablePenalty(nn.Module):
    """A custom sample callback with trainable and persistent state."""

    def __init__(self) -> None:
        """Create a multiplier and a stored zero bias."""
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))
        self.register_buffer("bias", torch.tensor(0.0))

    @override
    # Native Torch stubs declare variadic forward; this callback narrows it.
    def forward(  # pyrefly: ignore[bad-override]
        self, context: obb.SetLossContext
    ) -> obb.Result[Tensor, obb.InputError]:
        """Return raw per-sample penalties using the registered multiplier."""
        match _direction(context):
            case obb.Err() as failure:
                return failure
            case obb.Ok(value=values):
                return obb.Ok(values * self.scale + self.bias)


def test_module_state_hooks_and_disabled_registration(
    composed_sets: tuple[obb.PackedPredictions, obb.PackedTargets],
) -> None:
    """Modules retain hooks/gradients/state; omitted zero-weight terms register nothing."""
    preds, targets = composed_sets
    component = TrainablePenalty()
    observed: list[Tensor] = []

    def observe(
        module: nn.Module,
        inputs: tuple[obb.SetLossContext, ...],
        output: obb.Result[Tensor, obb.InputError],
    ) -> None:
        """Record invocation through Module.__call__ without modifying its result."""
        assert module is component
        assert len(inputs) == 1
        assert isinstance(output, obb.Ok)
        observed.append(output.value)

    component.register_forward_hook(observe)
    criterion = obb.SetCriterion(
        losses={"learned": obb.WeightedLoss(component, obb.WeightConfig(weight=3))}
    ).to(device=preds.logits.device, dtype=torch.float64)
    assert list(criterion.parameters()) == [component.scale]
    assert any(name.endswith(".bias") for name in criterion.state_dict())
    assert component.scale.dtype == torch.float64
    result = criterion(preds, targets, normalizer=1.0)
    assert isinstance(result, obb.Ok)
    result.value.total.backward()
    assert len(observed) == 1
    assert component.scale.grad is not None
    assert component.scale.grad.item() == pytest.approx(3 * 9.5625)
    criterion.load_state_dict(criterion.state_dict())
    disabled = obb.SetCriterion(
        losses={"off": obb.WeightedLoss(component, obb.WeightConfig(weight=0))}
    )
    assert not list(disabled.parameters())
    assert not disabled.state_dict()
    result = disabled(preds, targets)
    assert isinstance(result, obb.Ok)
    assert result.value.total.item() == 0
    assert len(observed) == 1


def test_custom_cost_chunking_and_gradients(
    composed_sets: tuple[obb.PackedPredictions, obb.PackedTargets],
) -> None:
    """Custom same-sample costs choose a new assignment and direct costs retain gradients."""
    preds, targets = composed_sets
    chunks: list[int] = []

    def prefer_large_center(
        context: obb.SetCostContext,
    ) -> obb.Result[Tensor, obb.InputError]:
        """Prefer larger query x coordinates for each bounded candidate pair."""
        chunks.append(context.pred_indices.shape[0])
        assert bool((context.pred_indices < 2).all())
        assert bool((context.target_indices == 0).all())
        return obb.Ok(-context.preds.geometry.boxes[context.pred_indices, 0].double())

    callbacks = {
        "center": obb.WeightedLoss(prefer_large_center, obb.WeightConfig(weight=2))
    }
    config = obb.MatcherConfig(chunk_size=1)
    costs = obb.set_pairwise_costs(preds, targets, config, costs=callbacks)
    assert isinstance(costs, obb.Ok)
    torch.testing.assert_close(
        costs.value.values,
        torch.tensor([0, -2], device=preds.logits.device, dtype=torch.float64),
    )
    costs.value.values.sum().backward()
    assert preds.geometry.boxes.grad is not None
    assert preds.geometry.boxes.grad[:, 0].tolist() == [-2, -2, 0]
    result = obb.SetCriterion(
        matcher=obb.SetHungarianMatcher(config, costs=callbacks), losses={}
    )(preds, targets)
    assert isinstance(result, obb.Ok)
    assert result.value.assignment.pred_indices.tolist() == [1]
    assert chunks == [1, 1, 1, 1]


class InvalidOutput(Enum):
    """Observable callback contract violations supported by checked rejection."""

    RANK = "rank"
    COUNT = "count"
    DTYPE = "dtype"
    FINITE = "finite"
    DEVICE = "device"


@pytest.mark.parametrize("violation", list(InvalidOutput))
def test_callback_output_rejections(
    composed_sets: tuple[obb.PackedPredictions, obb.PackedTargets],
    violation: InvalidOutput,
) -> None:
    """Reject malformed custom vectors with the caller's term name in diagnostics."""

    def malformed(context: obb.SetLossContext) -> obb.Result[Tensor, obb.InputError]:
        """Return a deliberate tensor contract violation without raising."""
        zero = context.zero()
        match violation:
            case InvalidOutput.RANK:
                value = zero.sum()
            case InvalidOutput.COUNT:
                value = zero[:-1]
            case InvalidOutput.DTYPE:
                value = zero.long()
            case InvalidOutput.FINITE:
                value = zero + float("inf")
            case InvalidOutput.DEVICE:
                value = torch.zeros(zero.shape, device="meta")
        return obb.Ok(value)

    result = obb.SetCriterion(losses={"invalid": obb.WeightedLoss(malformed)})(
        *composed_sets
    )
    assert isinstance(result, obb.Err)
    assert result.error.field.startswith("invalid")


def test_typed_callback_errors_and_weight_overflow(
    composed_sets: tuple[obb.PackedPredictions, obb.PackedTargets],
) -> None:
    """Preserve callback failures and reject overflow after finite weighting inputs."""

    def rejected(context: obb.SetLossContext) -> obb.Result[Tensor, obb.InputError]:
        """Report an explicit caller-defined numerical-domain failure."""
        return obb.Err(
            obb.InputError(ErrorKind.DOMAIN, "direction", "caller rejection")
        )

    result = obb.SetCriterion(losses={"custom": rejected})(*composed_sets)
    assert isinstance(result, obb.Err)
    assert result.error.field == "custom.direction"
    assert result.error.detail == "caller rejection"

    def large(context: obb.SetLossContext) -> obb.Result[Tensor, obb.InputError]:
        """Return finite values that overflow under a finite multiplier."""
        return obb.Ok(context.zero() + 1e308)

    result = obb.SetCriterion(
        losses={"overflow": obb.WeightedLoss(large, obb.WeightConfig(weight=1e308))}
    )(*composed_sets)
    assert isinstance(result, obb.Err)
    assert result.error.field == "overflow"


def test_empty_matching_composition(
    composed_sets: tuple[obb.PackedPredictions, obb.PackedTargets],
) -> None:
    """A matcher requires a cost even when every supplied wrapper is disabled."""
    empty: dict[str, obb.LossFunction[obb.SetCostContext]] = {}
    for callbacks in (
        empty,
        {"off": obb.WeightedLoss(obb.set_l1_cost, obb.WeightConfig(weight=0))},
    ):
        result = obb.SetHungarianMatcher(costs=callbacks)(*composed_sets)
        assert isinstance(result, obb.Err)
        assert result.error.kind is ErrorKind.CONFIG
