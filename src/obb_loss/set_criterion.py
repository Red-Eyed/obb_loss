"""Per-layer matching and weighted losses for oriented set prediction.

Set supervision and L1: Carion et al., ECCV 2020, https://arxiv.org/abs/2005.12872.
ProbIoU and sigmoid quality classification replace original DETR's GIoU and CE.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import TYPE_CHECKING, NamedTuple

import torch
from torch import Tensor, nn
from typing_extensions import override

from obb_loss.config import QFLConfig, Reduction, SetCriterionConfig, VFLConfig
from obb_loss.losses.l1 import box_l1_loss
from obb_loss.losses.quality_focal import quality_focal_loss
from obb_loss.losses.stable_probiou import stable_probiou_loss, validate_pairs
from obb_loss.losses.varifocal import varifocal_loss
from obb_loss.matching import gather_matched
from obb_loss.packed import ErrorKind, InputError
from obb_loss.reduction import reduce_loss, sum_samples
from obb_loss.result import Err, Ok, Result
from obb_loss.set_matching import SetHungarianMatcher

if TYPE_CHECKING:
    from obb_loss.matching import Assignment
    from obb_loss.packed import PackedBoxes
    from obb_loss.set_data import PackedPredictions, PackedTargets

_DEFAULT_CONFIG = SetCriterionConfig()


class TargetCount(Enum):
    """Normalize by each sample's target count or the local batch target count."""

    PER_SAMPLE = "per_sample"
    PER_BATCH = "per_batch"


class SetLosses(NamedTuple):
    """Return reduced component losses, their weighted total, and assignment.

    Loss tensors are float64 [batch] for Reduction.NONE and scalars for SUM/MEAN.

    Attributes:
        total: Weighted sum of the three normalized loss components.
        classification: Normalized unweighted VFL/QFL loss over query/classes.
        l1: Normalized unweighted center/side L1 loss over matched boxes.
        probiou: Normalized unweighted ProbIoU loss over matched boxes.
        assignment: Detached global query/reference pair indices and cumulative
            sample pair counts.
    """

    total: Tensor
    classification: Tensor
    l1: Tensor
    probiou: Tensor
    assignment: Assignment


def _classification_loss(
    preds: PackedPredictions,
    targets: PackedTargets,
    assignment: Assignment,
    config: VFLConfig | QFLConfig,
) -> Tensor:
    """Return one classification sum per query using caller quality at matched classes."""
    quality = torch.zeros_like(preds.logits, dtype=torch.float64)
    classes = targets.labels[assignment.target_indices]
    quality[assignment.pred_indices, classes] = targets.quality[
        assignment.target_indices
    ].double()
    match config:
        case VFLConfig():
            return varifocal_loss(
                preds.logits,
                quality,
                alpha=config.alpha,
                gamma=config.gamma,
                quality_weighted=config.quality_weighted,
            ).sum(dim=-1)
        case QFLConfig():
            return quality_focal_loss(preds.logits, quality, beta=config.beta).sum(
                dim=-1
            )


def _box_losses(
    first: PackedBoxes, second: PackedBoxes, config: SetCriterionConfig
) -> Result[tuple[Tensor, Tensor], InputError]:
    """Compute enabled aligned box terms and skip disabled numerical domains."""
    zero = first.boxes[:, 0].double() * 0 + second.boxes[:, 0].double() * 0
    l1, probiou = zero, zero
    if config.weights.l1 > 0:
        l1 = box_l1_loss(first.boxes, second.boxes)
    if config.weights.probiou > 0:
        match validate_pairs(first.boxes, second.boxes):
            case Err() as failure:
                return failure
            case Ok():
                probiou = stable_probiou_loss(first.boxes, second.boxes)
    return Ok((l1, probiou))


def _sample_divisor(
    targets: PackedTargets, normalizer: float | TargetCount
) -> Result[Tensor, InputError]:
    """Resolve an explicit normalization policy to one positive divisor per sample."""
    offsets = targets.geometry.cu_seqlens
    counts = (offsets[1:] - offsets[:-1]).double()
    match normalizer:
        case TargetCount.PER_SAMPLE:
            return Ok(counts.clamp_min(1))
        case TargetCount.PER_BATCH:
            return Ok(torch.full_like(counts, max(targets.geometry.boxes.shape[0], 1)))
        case float() | int():
            if math.isfinite(normalizer) and normalizer > 0:
                return Ok(torch.full_like(counts, normalizer))
    return Err(
        InputError(
            ErrorKind.DOMAIN,
            "normalizer",
            "target-count policy or finite positive scalar required",
        )
    )


def _reduce_components(
    classification: Tensor,
    box_losses: tuple[Tensor, Tensor],
    preds: PackedPredictions,
    assignment: Assignment,
    divisor: Tensor,
    reduction: Reduction,
) -> tuple[Tensor, Tensor, Tensor]:
    """Aggregate queries/pairs per sample, normalize, then apply the batch reduction."""
    l1, probiou = box_losses
    return (
        reduce_loss(
            sum_samples(classification, preds.geometry.cu_seqlens) / divisor, reduction
        ),
        reduce_loss(sum_samples(l1, assignment.cu_seqlens) / divisor, reduction),
        reduce_loss(sum_samples(probiou, assignment.cu_seqlens) / divisor, reduction),
    )


class SetCriterion(nn.Module):
    """Match and supervise one decoder layer; callers aggregate auxiliary layers."""

    def __init__(self, config: SetCriterionConfig = _DEFAULT_CONFIG) -> None:
        """Configure matching and loss terms for one decoder layer.

        Args:
            config: Independent training weights, matcher weights/storage, and VFL/QFL
                formulation parameters. Classification weight/reduction retain defaults
                because this criterion applies them centrally.
        """
        super().__init__()
        self.config = config
        self.matcher = SetHungarianMatcher(config.matcher)

    @override
    # Torch stubs declare variadic forward; this module intentionally narrows it.
    def forward(  # pyrefly: ignore[bad-override]
        self,
        preds: PackedPredictions,
        targets: PackedTargets,
        *,
        reduction: Reduction = Reduction.SUM,
        normalizer: float | TargetCount = TargetCount.PER_BATCH,
    ) -> Result[SetLosses, InputError]:
        """Match one decoder layer and compute its normalized sample losses.

        Normalization precedes reduction. Classification supervises all query/classes; L1 and ProbIoU supervise matched pairs. L1 excludes direction. Float64 work disables autocast locally. Zero-weight terms skip computation. The caller aggregates decoder layers.

        Args:
            preds: Packed geometry and [queries, classes] logits for one decoder layer,
                sharing box order and device. Boxes and targets must use shared
                coordinate units.
            targets: Packed reference geometry, int64 class labels, and caller-supplied
                [targets] quality in [0, 1]; labels and quality follow box order. Sample
                count/device must match preds; quality is never detached.
            reduction: Operation across normalized sample losses: NONE returns [batch],
                SUM returns their sum, and MEAN averages all samples including empty
                ones. Zero samples reduce to a connected zero.
            normalizer: PER_SAMPLE divides each sample by its own target count;
                PER_BATCH divides all samples by the local batch target count. Counts
                are clamped to one. A finite positive scalar supplies a shared divisor,
                e.g. a distributed target-count normalization.

        Returns:
            Ok(SetLosses) with normalized unweighted classification/L1/ProbIoU
            components, their weighted total, and detached assignment. Loss fields are
            [batch] for NONE and scalar otherwise. Err(InputError) reports invalid
            records, labels, quality, costs, domains, normalization, reduction, storage
            limits, or nonfinite losses.
        """
        if not isinstance(reduction, Reduction):
            return Err(
                InputError(ErrorKind.CONFIG, "reduction", "expected Reduction enum")
            )
        match self.matcher.forward(preds, targets):
            case Err() as failure:
                return failure
            case Ok(value=assignment):
                pass
        match _sample_divisor(targets, normalizer):
            case Err() as failure:
                return failure
            case Ok(value=divisor):
                pass
        first, second = gather_matched(preds.geometry, targets.geometry, assignment)
        with torch.autocast(preds.logits.device.type, enabled=False):
            match _box_losses(first, second, self.config):
                case Err() as failure:
                    return failure
                case Ok(value=box_losses):
                    pass
            classification = (preds.logits.double() * 0).sum(dim=-1)
            if self.config.weights.classification > 0:
                classification = _classification_loss(
                    preds, targets, assignment, self.config.classification
                )
            classification, l1, probiou = _reduce_components(
                classification, box_losses, preds, assignment, divisor, reduction
            )
            weights = self.config.weights
            total = (
                weights.classification * classification
                + weights.l1 * l1
                + weights.probiou * probiou
            )
            if not bool(
                torch.isfinite(torch.stack((classification, l1, probiou, total))).all()
            ):
                return Err(
                    InputError(
                        ErrorKind.DOMAIN, "loss", "nonfinite loss or weighted total"
                    )
                )
            return Ok(SetLosses(total, classification, l1, probiou, assignment))
