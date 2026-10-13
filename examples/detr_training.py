"""Integrate obb_loss at a DETR decoder's prediction-head boundary.

Run with: uv run --extra matching examples/detr_training.py
Synthetic decoder states exercise the final-layer, auxiliary-rematching and
shared-match training paths. Replace those states with your decoder's output;
no backbone or Transformer implementation is required by this example.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.optim
from einops import rearrange
from torch import Tensor, nn

import obb_loss as obb
from obb_loss.criterion_core import evaluate_losses, prepare_context
from obb_loss.losses.set_terms import (
    class_probability_cost,
    matched_box_inputs,
    one_hot_classification_inputs,
    pairwise_box_l1_cost,
    pairwise_probiou_cost,
    target_count,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def decode_boxes(raw: Tensor) -> Tensor[[int, int, int, 6]]:
    """Decode [layers,batch,queries,5] raw values into oriented geometry.

    Four sigmoid coordinates give cx,cy,width,height in common image-span units.
    The fifth value is an angle in radians; cos/sin produce a nonzero direction.
    Projection and decoding should run in FP32 outside autocast.
    """
    coordinates = raw[..., :4].sigmoid()
    angles = raw[..., 4]
    direction = torch.stack((angles.cos(), angles.sin()), dim=-1)
    return torch.cat((coordinates, direction), dim=-1)


def predict_layers(
    decoder_states: Tensor,
    class_head: nn.Linear[int, int],
    box_head: nn.Linear[int, int],
) -> tuple[obb.Predictions, ...]:
    """Project [layers,batch,queries,hidden] decoder states through shared FP32 heads.

    class_head produces foreground logits without a background column; box_head
    produces five raw coordinates. Keep both head parameters in FP32 when using
    autocast for the backbone/decoder. Returned tensors retain decoder gradients.
    """
    with torch.autocast(decoder_states.device.type, enabled=False):
        features = decoder_states.float()
        logits = class_head(features)
        boxes = decode_boxes(box_head(features))
    return tuple(
        obb.Predictions(layer_boxes, layer_logits)
        for layer_boxes, layer_logits in zip(
            boxes.unbind(0), logits.unbind(0), strict=True
        )
    )


def target_from_pixels(
    cxcywh: Tensor, angles: Tensor, labels: Tensor, *, height: int, width: int
) -> obb.Targets:
    """Convert one image's pixel boxes and radian angles to shared coordinate units.

    Use the same max(height,width) scale for centers and both side lengths.
    Independent x/y scaling would distort rotated geometry on nonsquare images.
    Labels are contiguous int64 foreground indices. Empty arrays are supported.
    """
    coordinates = cxcywh.float() / max(height, width)
    direction = torch.stack((angles.float().cos(), angles.float().sin()), dim=-1)
    return obb.Targets(torch.cat((coordinates, direction), dim=-1), labels)


def loss_with_shared_matches(
    criterion: obb.Criterion,
    predictions: obb.Predictions,
    targets: Sequence[obb.Targets],
    matches: tuple[obb.Match, ...],
) -> obb.Result[obb.LossOutput, obb.CriterionFailure]:
    """Evaluate an auxiliary layer with final-layer assignments, retaining gradients."""
    match prepare_context(predictions, targets, matches):
        case obb.Err(error):
            return obb.Err(error)
        case obb.Ok(context):
            match evaluate_losses(criterion.losses, context):
                case obb.Err(error):
                    return obb.Err(error)
                case obb.Ok(output):
                    return obb.Ok(output)


def require_loss(
    outcome: obb.Result[obb.LossOutput, obb.CriterionFailure],
) -> obb.LossOutput:
    """Abort this demonstration on a typed criterion error at the training boundary."""
    match outcome:
        case obb.Err(error):
            raise RuntimeError(f"criterion failed: {error}")
        case obb.Ok(output):
            return output


def auxiliary_loss(
    criterion: obb.Criterion,
    layers: tuple[obb.Predictions, ...],
    targets: Sequence[obb.Targets],
    final: obb.LossOutput,
    *,
    weights: tuple[float, ...],
    reuse_matches: bool,
) -> Tensor:
    """Sum explicit auxiliary weights with independent or shared final assignments.

    Supply one weight per auxiliary layer. Zero weights skip assignment and loss
    work. The caller adds this scalar to the final-layer total before backward.
    """
    total = final.total.new_zeros(())
    for predictions, weight in zip(layers, weights, strict=True):
        if weight == 0:
            continue
        if reuse_matches:
            outcome = loss_with_shared_matches(
                criterion, predictions, targets, final.matches
            )
        else:
            outcome = obb.criterion_loss(criterion, predictions, targets)
        total = total + weight * require_loss(outcome).total
    return total


def main() -> None:
    """Take one optimizer step per integration policy with explicit layer weights."""
    torch.manual_seed(7)
    hidden_width, classes = 32, 3
    class_head = nn.Linear(hidden_width, classes)
    box_head = nn.Linear(hidden_width, 5)
    optimizer = torch.optim.AdamW(
        (*class_head.parameters(), *box_head.parameters()), lr=1e-3
    )
    criterion = obb.Criterion(
        matcher=obb.Matcher(
            solver=obb.hungarian_assignment,
            costs=(
                obb.Cost("classes", class_probability_cost, weight=1.0),
                obb.Cost("l1", pairwise_box_l1_cost, weight=5.0),
                obb.Cost("probiou", pairwise_probiou_cost, weight=2.0),
            ),
        ),
        losses=(
            obb.Loss(
                "classes",
                obb.quality_focal_loss,
                one_hot_classification_inputs,
                weight=1.0,
                normalize=target_count,
            ),
            obb.Loss(
                "l1",
                obb.box_l1_loss,
                matched_box_inputs,
                weight=5.0,
                normalize=target_count,
            ),
            obb.Loss(
                "probiou",
                obb.stable_probiou_loss,
                matched_box_inputs,
                weight=2.0,
                normalize=target_count,
            ),
        ),
    )
    targets = (
        target_from_pixels(
            torch.tensor([[160.0, 120.0, 80.0, 40.0], [400.0, 300.0, 100.0, 60.0]]),
            torch.tensor([0.3, -0.5]),
            torch.tensor([0, 2]),
            height=480,
            width=640,
        ),
        target_from_pixels(
            torch.empty(0, 4),
            torch.empty(0),
            torch.empty(0, dtype=torch.int64),
            height=480,
            width=640,
        ),
    )

    for policy, weights, reuse_matches in (
        ("final_only", (0.0, 0.0), False),
        ("auxiliary_rematch", (0.5, 0.5), False),
        ("auxiliary_shared", (0.5, 0.5), True),
    ):
        optimizer.zero_grad(set_to_none=True)
        # Your DETR decoder supplies this tensor; synthetic leaves verify backward.
        states = torch.randn(3, 2, 8, hidden_width, requires_grad=True)
        layers = predict_layers(states, class_head, box_head)
        final = require_loss(obb.criterion_loss(criterion, layers[-1], targets))
        total = final.total + auxiliary_loss(
            criterion,
            layers[:-1],
            targets,
            final,
            weights=weights,
            reuse_matches=reuse_matches,
        )
        total.backward()
        optimizer.step()
        assert states.grad is not None
        layer_gradients = rearrange(states.grad, "l b q h -> l (b q h)").norm(dim=-1)
        print(
            f"{policy}: loss={total.detach().item():.6f}, "
            f"decoder_gradient_norms={layer_gradients.tolist()}"
        )
        print({name: value.detach().item() for name, value in final.raw.items()})


if __name__ == "__main__":
    main()
