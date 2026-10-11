"""Packed admission and integration adapter contracts."""

import pytest
import torch
from einops import rearrange
from torch import Tensor

from obb_loss.adapters import boxes_from_angles, pack_boxes, pack_dense_boxes
from obb_loss.packed import make_packed_boxes, sample_indices
from obb_loss.result import Err, Ok


@pytest.fixture
def boxes() -> Tensor:
    """Provide finite six-coordinate boxes as independent prediction leaves."""
    return torch.tensor(
        [[0, 0, 2, 1, 1, 0], [1, 0, 2, 1, 1, 0]],
        dtype=torch.float64,
        requires_grad=True,
    )


def test_list_empty_sample_order_and_gradients(boxes: Tensor) -> None:
    """Retain an empty middle sample while gradients flow to original storage."""
    outcome = pack_boxes(
        [boxes[:1], boxes[:0], boxes[1:]], device=boxes.device, dtype=boxes.dtype
    )
    assert isinstance(outcome, Ok)
    packed = outcome.value
    torch.testing.assert_close(packed.cu_seqlens, torch.tensor([0, 1, 1, 2]))
    torch.testing.assert_close(sample_indices(packed), torch.tensor([0, 2]))
    packed.boxes.sum().backward()
    torch.testing.assert_close(boxes.grad, torch.ones_like(boxes))


@pytest.mark.parametrize("offsets", [[1, 2], [0, 3], [0, 2, 1, 2]])
def test_reject_bad_offsets(boxes: Tensor, offsets: list[int]) -> None:
    """Reject wrong endpoints and nonmonotonic offsets through typed outcomes."""
    assert isinstance(make_packed_boxes(boxes, torch.tensor(offsets)), Err)


@pytest.mark.parametrize(("index", "value"), [(2, 0.0), (3, -1.0), (0, float("nan"))])
def test_reject_invalid_geometry(boxes: Tensor, index: int, value: float) -> None:
    """Reject nonfinite centers and invalid sides without changing inputs."""
    invalid = boxes.detach().clone()
    invalid[0, index] = value
    assert isinstance(make_packed_boxes(invalid, torch.tensor([0, 2])), Err)


def test_dense_adapter_matches_direct_record(boxes: Tensor) -> None:
    """Dense flattening uses the same packed convention and retains gradients."""
    outcome = pack_dense_boxes(rearrange(boxes, "b c -> b 1 c"))
    assert isinstance(outcome, Ok)
    torch.testing.assert_close(outcome.value.boxes, boxes)
    torch.testing.assert_close(outcome.value.cu_seqlens, torch.tensor([0, 1, 2]))


@pytest.mark.parametrize("samples", [0, 3])
def test_completely_empty(samples: int) -> None:
    """Distinguish zero samples from multiple empty samples with explicit metadata."""
    outcome = pack_boxes(
        [torch.empty(0, 6)] * samples, device=torch.device("cpu"), dtype=torch.float32
    )
    assert isinstance(outcome, Ok)
    assert outcome.value.boxes.shape == (0, 6)
    assert outcome.value.cu_seqlens.shape == (samples + 1,)
    assert sample_indices(outcome.value).numel() == 0


def test_angles_are_explicit_and_differentiable() -> None:
    """An explicit radians conversion preserves the angle's autograd connection."""
    values = torch.tensor([[0.0, 0.0, 2.0, 1.0, 0.3]], requires_grad=True)
    outcome = boxes_from_angles(values)
    assert isinstance(outcome, Ok)
    outcome.value.sum().backward()
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()
