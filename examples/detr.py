"""Integrate packed matching and losses with oriented DETR decoder outputs."""

from typing import NamedTuple

import torch
from einops import rearrange
from torch import Tensor

import obb_loss as obb


class DecoderOutput(NamedTuple):
    """One decoder layer: boxes `[batch, queries, 6]`, logits `[batch, queries, classes]`."""

    boxes: Tensor
    logits: Tensor


class Target(NamedTuple):
    """One sample: boxes `[count, 6]` and zero-based class labels `[count]`."""

    boxes: Tensor
    labels: Tensor


def layer_loss(
    output: DecoderOutput,
    targets: obb.PackedBoxes,
    labels: Tensor,
    matcher: obb.HungarianMatcher,
) -> obb.Result[Tensor, obb.InputError]:
    """Rematch one layer and use detached ProbIoU scores as positive VFL targets."""
    match obb.pack_dense_boxes(output.boxes):
        case obb.Err() as failure:
            return failure
        case obb.Ok(value=preds):
            pass
    match matcher.forward(preds, targets):
        case obb.Err() as failure:
            return failure
        case obb.Ok(value=assignment):
            pass
    first, second = obb.gather_matched(preds, targets, assignment)
    box_criterion = obb.ProbIoULoss(obb.ProbIoUConfig(reduction=obb.Reduction.SUM))
    match box_criterion.checked(first, second):
        case obb.Err() as failure:
            return failure
        case obb.Ok(value=box_loss):
            pass
    logits = rearrange(output.logits, "b q c -> (b q) c")
    quality = torch.zeros_like(logits)
    with torch.no_grad():
        scores = obb.stable_probiou(first.boxes, second.boxes)
        classes = labels[assignment.target_indices]
        quality[assignment.pred_indices, classes] = scores.to(quality.dtype)
    classification = obb.VarifocalLoss(obb.VFLConfig(reduction=obb.Reduction.SUM))
    match classification.checked(logits, quality):
        case obb.Err() as failure:
            return failure
        case obb.Ok(value=classification_loss):
            pass
    normalizer = max(targets.boxes.shape[0], 1)
    return obb.Ok((2.0 * box_loss + classification_loss) / normalizer)


def training_loss(
    final: DecoderOutput,
    targets: tuple[Target, ...],
    *,
    auxiliary: tuple[DecoderOutput, ...] = (),
) -> obb.Result[Tensor, obb.InputError]:
    """Pack targets once and combine final and auxiliary decoder-layer losses."""
    reference = final.boxes
    match obb.pack_boxes(
        [target.boxes for target in targets],
        device=reference.device,
        dtype=reference.dtype,
    ):
        case obb.Err() as failure:
            return failure
        case obb.Ok(value=packed):
            pass
    labels = (
        torch.cat([target.labels for target in targets])
        if targets
        else torch.empty(0, dtype=torch.int64, device=reference.device)
    )
    matcher = obb.HungarianMatcher(obb.MatcherConfig(chunk_size=128, max_pairs=10000))
    total = reference.double().sum() * 0.0
    outputs = (*auxiliary, final)
    for index, output in enumerate(outputs):
        match layer_loss(output, packed, labels, matcher):
            case obb.Err() as failure:
                return failure
            case obb.Ok(value=loss):
                weight = 1.0 if index == len(outputs) - 1 else 0.5
                total = total + weight * loss
    return obb.Ok(total)


def main() -> None:
    """Run two decoder layers and verify gradients, including an empty target sample."""
    boxes = torch.tensor(
        [
            [[0.1, 0.0, 2.0, 1.0, 1.0, 0.0], [3.0, 0.0, 2.0, 1.0, 1.0, 0.0]],
            [[0.2, 0.0, 2.0, 1.0, 1.0, 0.0], [4.0, 0.0, 2.0, 1.0, 1.0, 0.0]],
        ]
    )
    outputs = tuple(
        DecoderOutput(
            boxes.clone().requires_grad_(),
            torch.zeros(2, 2, 3, requires_grad=True),
        )
        for _ in range(2)
    )
    targets = (
        Target(torch.tensor([[0.0, 0.0, 2.0, 1.0, 1.0, 0.0]]), torch.tensor([1])),
        Target(torch.empty(0, 6), torch.empty(0, dtype=torch.int64)),
    )
    match training_loss(outputs[-1], targets, auxiliary=outputs[:-1]):
        case obb.Err(error=error):
            print(error)
            return
        case obb.Ok(value=loss):
            loss.backward()
    print(f"DETR training loss: {loss.detach().item():.6f}")
    for index, output in enumerate(outputs):
        assert output.boxes.grad is not None
        assert output.logits.grad is not None
        assert bool(torch.isfinite(output.boxes.grad).all())
        assert bool(torch.isfinite(output.logits.grad).all())
        print(f"layer {index}: finite box and classification gradients")


if __name__ == "__main__":
    main()
