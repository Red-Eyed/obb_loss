"""Use one set criterion per DETR decoder layer and aggregate losses in the caller."""

from typing import NamedTuple

import torch
from torch import Tensor
from typing_extensions import override

import obb_loss as obb

_VFL_PARAMETERS = obb.VFLParameters(alpha=0.75, gamma=2)


class DecoderOutput(NamedTuple):
    """One layer's `[batch, queries, 6]` geometry and `[batch, queries, classes]` logits."""

    boxes: Tensor
    logits: Tensor


def layer_loss(
    output: DecoderOutput,
    targets: obb.Targets,
    criterion: obb.SetCriterion,
) -> obb.Result[obb.SetLosses, obb.InputError]:
    """Adapt one dense decoder output and let the criterion match and supervise it."""
    match obb.pack_predictions(output.boxes, output.logits):
        case obb.Err() as failure:
            return failure
        case obb.Ok(value=preds):
            return criterion(
                preds,
                targets,
                reduction=obb.Reduction.NONE,
            )


class DetectorCriterion(obb.SetCriterion):
    """Choose detached localization quality for classification supervision."""

    @override
    def classification_quality(
        self,
        preds: obb.Predictions,
        targets: obb.Targets,
        assignment: obb.Assignment,
    ) -> obb.Result[Tensor, obb.InputError]:
        """Return this layer's matched ProbIoU scores without quality gradients."""
        first, second = obb.gather_matched(preds.boxes, targets.boxes, assignment)
        return obb.Ok(obb.stable_probiou(first.values, second.values).detach())


def main() -> None:
    """Train decoder layers with caller-selected quality computed after matching."""
    boxes = torch.tensor(
        [
            [[0.1, 0.0, 2.0, 1.0, 1.0, 0.0], [3.0, 0.0, 2.0, 1.0, 1.0, 0.0]],
            [[0.2, 0.0, 2.0, 1.0, 1.0, 0.0], [4.0, 0.0, 2.0, 1.0, 1.0, 0.0]],
        ]
    )
    outputs = tuple(
        DecoderOutput(
            boxes.clone().requires_grad_(), torch.zeros(2, 2, 3, requires_grad=True)
        )
        for _ in range(2)
    )
    match obb.pack_targets(
        [
            obb.TargetSample(
                torch.tensor([[0.0, 0.0, 2.0, 1.0, 1.0, 0.0]]), torch.tensor([1])
            ),
            obb.TargetSample(torch.empty(0, 6), torch.empty(0, dtype=torch.int64)),
        ],
        device=boxes.device,
        dtype=boxes.dtype,
    ):
        case obb.Err(error=error):
            print(error)
            return
        case obb.Ok(value=targets):
            pass
    criterion = DetectorCriterion(
        obb.SetCriterionConfig(
            matcher=obb.MatcherConfig(
                classification_weight=2, chunk_size=128, max_pairs=10000
            ),
            normalizer=obb.TargetCount.PER_SAMPLE,
            vfl=_VFL_PARAMETERS,
        )
    )
    total = boxes.double().sum() * 0
    for index, output in enumerate(outputs):
        match layer_loss(output, targets, criterion):
            case obb.Err(error=error):
                print(error)
                return
            case obb.Ok(value=losses):
                weight = 1.0 if index == len(outputs) - 1 else 0.5
                total = total + weight * losses.total.mean()
                print(
                    f"layer {index} per-sample totals: {losses.total.detach().tolist()}"
                )
    total.backward()
    for index, output in enumerate(outputs):
        assert output.boxes.grad is not None
        assert output.logits.grad is not None
        assert bool(torch.isfinite(output.boxes.grad).all())
        assert bool(torch.isfinite(output.logits.grad).all())
        print(f"layer {index}: finite box and classification gradients")
    print(f"DETR training loss: {total.detach().item():.6f}")


if __name__ == "__main__":
    main()
