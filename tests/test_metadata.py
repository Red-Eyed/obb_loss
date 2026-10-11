"""Public loss metadata errors and absence of tensor-value validation."""

from collections.abc import Callable

import pytest
import torch
from einops import rearrange
from torch import Tensor

import obb_loss as obb


@pytest.fixture(
    params=[
        obb.stable_probiou_loss,
        obb.stable_probiou,
        obb.box_l1_loss,
        obb.varifocal_loss,
        obb.quality_focal_loss,
    ]
)
def loss_function(
    request: pytest.FixtureRequest,
) -> Callable[[Tensor, Tensor], obb.Result[Tensor, obb.InputError]]:
    """Exercise every public tensor loss with the same metadata contract failures."""
    function: Callable[[Tensor, Tensor], obb.Result[Tensor, obb.InputError]] = (
        request.param
    )
    return function


@pytest.mark.parametrize("invalid", ["shape", "dtype", "device"])
def test_metadata_errors(
    loss_function: Callable[[Tensor, Tensor], obb.Result[Tensor, obb.InputError]],
    invalid: str,
) -> None:
    """Return a typed error before evaluating incompatible tensor pairs."""
    first = torch.ones(2, 6)
    match invalid:
        case "shape":
            second = torch.ones(3, 5)
        case "dtype":
            second = torch.ones(2, 6, dtype=torch.int64)
        case _:
            second = torch.ones(2, 6, device="meta")
    result = loss_function(first, second)
    assert isinstance(result, obb.Err)
    assert result.error.kind.value == invalid


def test_probiou_broadcast_contract() -> None:
    """Broadcast boxes without materializing all pairs; reject incompatible axes."""
    first = torch.tensor([[[0.0, 0, 2, 1, 1, 0]], [[1.0, 0, 2, 1, 1, 0]]])
    second = rearrange(
        first, "samples singleton coordinates -> singleton samples coordinates"
    )
    result = obb.stable_probiou_loss(first, second)
    assert isinstance(result, obb.Ok)
    assert result.value.shape == (2, 2)
    assert isinstance(
        obb.stable_probiou_loss(first.expand(2, 2, 6), first[:1].expand(3, 2, 6)),
        obb.Err,
    )


def test_no_tensor_value_scans(monkeypatch: pytest.MonkeyPatch) -> None:
    """Metadata checks never read tensor scalars or scan numerical finiteness."""
    boxes = torch.tensor([[0.0, 0, 2, 1, 1, 0]])
    logits = torch.zeros(1, 2)
    quality = torch.zeros_like(logits)

    def forbidden(*args: Tensor, **kwargs: bool) -> Tensor:
        """Fail if loss execution attempts a host-side value check."""
        pytest.fail("unexpected tensor value scan")

    monkeypatch.setattr(torch, "isfinite", forbidden)
    monkeypatch.setattr(Tensor, "item", forbidden)
    monkeypatch.setattr(Tensor, "__bool__", forbidden)
    for result in (
        obb.stable_probiou_loss(boxes, boxes),
        obb.stable_probiou(boxes, boxes),
        obb.box_l1_loss(boxes, boxes),
        obb.varifocal_loss(logits, quality),
        obb.quality_focal_loss(logits, quality),
    ):
        assert isinstance(result, obb.Ok)
