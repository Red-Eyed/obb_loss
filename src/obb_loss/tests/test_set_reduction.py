"""Per-sample set losses, explicit normalization, batch reduction, and gradient isolation."""

import pytest
import torch
import torch.testing

import obb_loss as obb


@pytest.fixture
def batch(device: torch.device) -> tuple[obb.PackedPredictions, obb.PackedTargets]:
    """Three samples have unequal query/target counts, including no targets in the last."""
    boxes = torch.tensor(
        [[x, 0, 2, 1, 1, 0] for x in (0.1, 3.0, 1.2, 2.4, 9.0, 0.2)],
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )
    logits = torch.full(
        (6, 2), 0.3, device=device, dtype=torch.float64, requires_grad=True
    )
    preds = obb.PackedPredictions(
        obb.PackedBoxes(boxes, torch.tensor([0, 2, 5, 6], device=device)), logits
    )
    targets = obb.PackedTargets(
        obb.PackedBoxes(
            torch.tensor(
                [[x, 0, 2, 1, 1, 0] for x in (0.0, 1.0, 2.0)],
                device=device,
                dtype=torch.float64,
            ),
            torch.tensor([0, 1, 3, 3], device=device),
        ),
        torch.tensor([0, 1, 0], device=device),
        torch.tensor([0.7, 0.6, 0.9], device=device, dtype=torch.float64),
    )
    return preds, targets


@pytest.mark.parametrize(
    "normalizer", [obb.TargetCount.PER_SAMPLE, obb.TargetCount.PER_BATCH, 2.0]
)
def test_reductions_match_sample_vector(
    batch: tuple[obb.PackedPredictions, obb.PackedTargets],
    normalizer: obb.TargetCount | float,
) -> None:
    """SUM and MEAN reduce the same normalized sample vector for every component."""
    criterion = obb.SetCriterion()
    none = criterion.forward(
        *batch, reduction=obb.Reduction.NONE, normalizer=normalizer
    )
    summed = criterion.forward(
        *batch, reduction=obb.Reduction.SUM, normalizer=normalizer
    )
    mean = criterion.forward(
        *batch, reduction=obb.Reduction.MEAN, normalizer=normalizer
    )
    assert isinstance(none, obb.Ok)
    assert isinstance(summed, obb.Ok)
    assert isinstance(mean, obb.Ok)
    for vector, total, average in zip(
        (*none.value.terms.values(), none.value.total),
        (*summed.value.terms.values(), summed.value.total),
        (*mean.value.terms.values(), mean.value.total),
        strict=True,
    ):
        assert vector.shape == (3,)
        torch.testing.assert_close(total, vector.sum())
        torch.testing.assert_close(average, vector.mean())
    match normalizer:
        case obb.TargetCount.PER_SAMPLE:
            expected = [0.1, 0.3, 0.0]
        case obb.TargetCount.PER_BATCH:
            expected = [0.1 / 3, 0.6 / 3, 0.0]
            default = criterion.forward(*batch)
            assert isinstance(default, obb.Ok)
            torch.testing.assert_close(default.value.total, summed.value.total)
        case float():
            expected = [0.05, 0.3, 0.0]
    torch.testing.assert_close(
        none.value.terms["l1"],
        5 * torch.tensor(expected, device=batch[0].logits.device, dtype=torch.float64),
    )
    assert none.value.terms["classification"][-1].item() > 0
    assert none.value.terms["probiou"][-1].item() == 0


def test_sample_losses_match_independent_calls(
    batch: tuple[obb.PackedPredictions, obb.PackedTargets],
) -> None:
    """Per-sample normalized values equal running each sample by itself."""
    preds, targets = batch
    criterion = obb.SetCriterion()
    together = criterion.forward(
        preds,
        targets,
        reduction=obb.Reduction.NONE,
        normalizer=obb.TargetCount.PER_SAMPLE,
    )
    assert isinstance(together, obb.Ok)
    for sample, (p0, p1, t0, t1) in enumerate(
        ((0, 2, 0, 1), (2, 5, 1, 3), (5, 6, 3, 3))
    ):
        first = obb.PackedPredictions(
            obb.PackedBoxes(
                preds.geometry.boxes[p0:p1],
                torch.tensor([0, p1 - p0], device=preds.logits.device),
            ),
            preds.logits[p0:p1],
        )
        second = obb.PackedTargets(
            obb.PackedBoxes(
                targets.geometry.boxes[t0:t1],
                torch.tensor([0, t1 - t0], device=preds.logits.device),
            ),
            targets.labels[t0:t1],
            targets.quality[t0:t1],
        )
        alone = criterion.forward(
            first,
            second,
            reduction=obb.Reduction.SUM,
            normalizer=obb.TargetCount.PER_SAMPLE,
        )
        assert isinstance(alone, obb.Ok)
        for vector, scalar in zip(
            (*together.value.terms.values(), together.value.total),
            (*alone.value.terms.values(), alone.value.total),
            strict=True,
        ):
            torch.testing.assert_close(vector[sample], scalar)


def test_selected_sample_gradient_isolation(
    batch: tuple[obb.PackedPredictions, obb.PackedTargets],
) -> None:
    """Backpropagating a selected sample touches only that sample's queries."""
    preds, targets = batch
    result = obb.SetCriterion().forward(
        preds,
        targets,
        reduction=obb.Reduction.NONE,
        normalizer=obb.TargetCount.PER_SAMPLE,
    )
    assert isinstance(result, obb.Ok)
    result.value.total[1].backward()
    for tensor in (preds.geometry.boxes, preds.logits):
        assert tensor.grad is not None
        assert bool((tensor.grad[:2] == 0).all())
        assert bool((tensor.grad[5:] == 0).all())
        assert bool((tensor.grad[2:5] != 0).any())


@pytest.mark.parametrize("reduction", list(obb.Reduction))
def test_zero_samples_and_empty_gradients(
    device: torch.device, reduction: obb.Reduction
) -> None:
    """Zero samples yield an empty vector or a differentiable scalar zero."""
    boxes = torch.empty(0, 6, device=device, dtype=torch.float64).requires_grad_()
    logits = torch.empty(0, 2, device=device, dtype=torch.float64).requires_grad_()
    offsets = torch.tensor([0], device=device)
    preds = obb.PackedPredictions(obb.PackedBoxes(boxes, offsets), logits)
    targets = obb.PackedTargets(
        obb.PackedBoxes(boxes.detach(), offsets),
        torch.empty(0, device=device, dtype=torch.int64),
        torch.empty(0, device=device),
    )
    result = obb.SetCriterion().forward(
        preds, targets, reduction=reduction, normalizer=obb.TargetCount.PER_SAMPLE
    )
    assert isinstance(result, obb.Ok)
    expected_shape = (0,) if reduction is obb.Reduction.NONE else ()
    assert result.value.total.shape == expected_shape
    result.value.total.sum().backward()
    assert boxes.grad is not None
    assert logits.grad is not None


def test_zero_query_samples_preserve_positions(device: torch.device) -> None:
    """Repeated query offsets keep empty samples in their original batch positions."""
    boxes = torch.tensor(
        [[0.1, 0, 2, 1, 1, 0]], device=device, dtype=torch.float64, requires_grad=True
    )
    logits = torch.zeros(1, 2, device=device, dtype=torch.float64, requires_grad=True)
    offsets = torch.tensor([0, 0, 1, 1], device=device)
    preds = obb.PackedPredictions(obb.PackedBoxes(boxes, offsets), logits)
    targets = obb.PackedTargets(
        obb.PackedBoxes(
            torch.tensor([[0.0, 0, 2, 1, 1, 0]], device=device, dtype=torch.float64),
            offsets,
        ),
        torch.tensor([0], device=device),
        torch.tensor([0.7], device=device),
    )
    result = obb.SetCriterion().forward(
        preds,
        targets,
        reduction=obb.Reduction.NONE,
        normalizer=obb.TargetCount.PER_SAMPLE,
    )
    assert isinstance(result, obb.Ok)
    assert result.value.total.shape == (3,)
    assert result.value.total[0].item() == result.value.total[2].item() == 0
    assert result.value.total[1].item() > 0
    assert result.value.assignment.cu_seqlens.tolist() == [0, 0, 1, 1]
    result.value.total.sum().backward()
    assert boxes.grad is not None
    assert logits.grad is not None
