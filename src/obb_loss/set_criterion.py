"""Subclassable packed set loss; DETR: https://arxiv.org/abs/2005.12872."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, NamedTuple

import torch
from torch import Tensor, nn
from typing_extensions import override

from obb_loss.config import Reduction, SetCriterionConfig, TargetCount
from obb_loss.losses.l1 import box_l1_loss
from obb_loss.losses.quality_focal import quality_focal_loss
from obb_loss.losses.stable_probiou import stable_probiou_loss, validate_pair_domain
from obb_loss.losses.varifocal import varifocal_loss
from obb_loss.matching import Assignment, gather_matched
from obb_loss.packed import ErrorKind, InputError, PackedBatch
from obb_loss.pairwise import validate_output
from obb_loss.reduction import reduce_loss, sum_samples
from obb_loss.result import Err, Ok, Result
from obb_loss.set_matching import SetHungarianMatcher

if TYPE_CHECKING:
    from collections.abc import Mapping

    from obb_loss.set_data import Predictions, Targets

_DEFAULT_CONFIG = SetCriterionConfig()


class SetLosses(NamedTuple):
    """Return normalized loss values and the assignment used for supervision.

    Attributes:
        total: Sum of weighted terms, [samples] for NONE or a scalar for SUM/MEAN.
        terms: Read-only named weighted losses, with the same reduction as total.
        assignment: Detached global query/reference indices and per-sample offsets.
    """

    total: Tensor
    terms: Mapping[str, Tensor]
    assignment: Assignment


class SetCriterion(nn.Module):
    """Configure once, then match and supervise one packed decoder layer per call."""

    def __init__(self, config: SetCriterionConfig = _DEFAULT_CONFIG) -> None:
        """Select numerical settings, independent weights and normalization.

        Args:
            config: Immutable training/matching configuration. Override loss methods
                for custom formulas; classification quality is supplied after matching.
        """
        super().__init__()
        self.config = config
        self.matcher = SetHungarianMatcher(config.matcher)

    def classification_quality(
        self, preds: Predictions, targets: Targets, assignment: Assignment
    ) -> Result[Tensor, InputError]:
        """Supply matched-pair classification quality by overriding this method.

        Args:
            preds: All packed query geometry/logits for this layer.
            targets: Packed reference geometry and labels; quality is not annotation data.
            assignment: Matched-pair order for the returned quality vector.

        Returns:
            A typed configuration error until overridden. Return Ok(float [matches]
            scores in [0, 1]) or Err; choose detachment explicitly in the override.
        """
        return Err(
            InputError(
                ErrorKind.CONFIG,
                "quality",
                "override classification_quality or disable classification",
            )
        )

    def classification_loss(
        self, preds: Predictions, targets: Targets, assignment: Assignment
    ) -> Result[Tensor, InputError]:
        """Calculate weighted VFL/QFL sums over every query/class per sample.

        Args:
            preds: Admitted packed queries and logits without a background column.
            targets: Admitted reference boxes and class labels.
            assignment: Global matched indices; unmatched query/classes are negative.

        Returns:
            Ok(weighted raw [samples] sums) or Err for invalid supplied quality.
            Zero weight or disabled classification skips quality and loss evaluation.
        """
        if (
            self.config.classification == "none"
            or self.config.classification_weight == 0
        ):
            return Ok(
                sum_samples(preds.logits[:, 0].double() * 0, preds.boxes.cu_seqlens)
            )
        match self.classification_quality(preds, targets, assignment):
            case Err() as failure:
                return failure
            case Ok(value=quality):
                pass
        match validate_output(
            quality, assignment.pred_indices.shape[0], preds.boxes.values.device
        ):
            case Err() as failure:
                return failure
            case Ok():
                pass
        if not bool(((quality >= 0) & (quality <= 1)).all()):
            return Err(
                InputError(ErrorKind.DOMAIN, "quality", "scores must be in [0,1]")
            )
        scores = torch.zeros_like(preds.logits, dtype=torch.float64)
        scores[assignment.pred_indices, targets.labels[assignment.target_indices]] = (
            quality.double()
        )
        if self.config.classification == "vfl":
            params = self.config.vfl
            values = varifocal_loss(
                preds.logits,
                scores,
                alpha=params.alpha,
                gamma=params.gamma,
                quality_weighted=params.quality_weighted,
            )
        else:
            values = quality_focal_loss(preds.logits, scores, beta=self.config.qfl.beta)
        return Ok(
            self.config.classification_weight
            * sum_samples(values.sum(-1), preds.boxes.cu_seqlens)
        )

    def l1_loss(
        self, preds: PackedBatch, targets: PackedBatch
    ) -> Result[Tensor, InputError]:
        """Calculate weighted center/side L1 sums for already matched packed boxes.

        Args:
            preds: Gathered matched predictions, with per-sample match offsets.
            targets: Gathered aligned references in the same coordinate units.

        Returns:
            Ok(weighted raw [samples] sums); zero weight skips L1 evaluation.
        """
        values = preds.values[:, 0].double() * 0
        if self.config.l1_weight > 0:
            values = self.config.l1_weight * box_l1_loss(preds.values, targets.values)
        return Ok(sum_samples(values, preds.cu_seqlens))

    def probiou_loss(
        self, preds: PackedBatch, targets: PackedBatch
    ) -> Result[Tensor, InputError]:
        """Calculate weighted ProbIoU sums for already matched packed boxes.

        Args:
            preds: Gathered matched predictions with per-sample match offsets.
            targets: Gathered aligned reference geometry, sharing units/device.

        Returns:
            Ok(weighted raw [samples] sums) or a numerical-domain error. Zero weight
            skips domain/loss evaluation; overrides can change formula or precision.
        """
        values = preds.values[:, 0].double() * 0
        if self.config.probiou_weight > 0:
            match validate_pair_domain(preds.values, targets.values):
                case Err() as failure:
                    return failure
                case Ok():
                    values = self.config.probiou_weight * stable_probiou_loss(
                        preds.values, targets.values
                    )
        return Ok(sum_samples(values, preds.cu_seqlens))

    def compute_losses(
        self, preds: Predictions, targets: Targets, assignment: Assignment
    ) -> Result[Mapping[str, Tensor], InputError]:
        """Calculate named loss vectors; override to add, replace or omit terms.

        Args:
            preds: All packed queries, including unmatched queries for custom regularizers.
            targets: Packed annotations containing geometry and labels only.
            assignment: This layer's detached global matches and sample offsets.

        Returns:
            Ok(named weighted raw [samples] vectors) or Err. Forward handles output
            validation, normalization and reduction once, after this method returns.
        """
        match self.classification_loss(preds, targets, assignment):
            case Err() as failure:
                return failure
            case Ok(value=classification):
                pass
        first, second = gather_matched(preds.boxes, targets.boxes, assignment)
        match self.l1_loss(first, second):
            case Err() as failure:
                return failure
            case Ok(value=l1):
                pass
        match self.probiou_loss(first, second):
            case Err() as failure:
                return failure
            case Ok(value=probiou):
                return Ok(
                    {"classification": classification, "l1": l1, "probiou": probiou}
                )

    @override
    # Native Torch stubs declare variadic forward; packed inputs narrow it.
    def forward(  # pyrefly: ignore[bad-override]
        self,
        preds: Predictions,
        targets: Targets,
        reduction: Reduction = Reduction.SUM,
    ) -> Result[SetLosses, InputError]:
        """Match one layer, calculate its losses, then normalize and reduce samples.

        Args:
            preds: Borrowed packed boxes, offsets and floating [queries, classes] logits.
            targets: Borrowed packed boxes, offsets and int64 [targets] class labels.
            reduction: NONE retains [samples]; SUM adds that vector; MEAN averages it,
                including empty samples. Empty scalar reductions return connected zero.

        Returns:
            Ok(total, named weighted terms and assignment) or typed input/domain/
            output failure. Matching runs without autograd; loss evaluation disables
            autocast locally. Inputs are never mutated or detached by the criterion.
        """
        if not isinstance(reduction, Reduction):
            return Err(
                InputError(ErrorKind.CONFIG, "reduction", "expected Reduction enum")
            )
        match self.matcher(preds, targets):
            case Err() as failure:
                return failure
            case Ok(value=assignment):
                pass
        counts = (targets.boxes.cu_seqlens[1:] - targets.boxes.cu_seqlens[:-1]).double()
        match self.config.normalizer:
            case TargetCount.PER_SAMPLE:
                divisor = counts.clamp_min(1)
            case TargetCount.PER_BATCH:
                divisor = torch.full_like(counts, max(targets.boxes.values.shape[0], 1))
            case float() as value:
                divisor = torch.full_like(counts, value)
        with torch.autocast(preds.boxes.values.device.type, enabled=False):
            match self.compute_losses(preds, targets, assignment):
                case Err() as failure:
                    return failure
                case Ok(value=outputs):
                    pass
            total = sum_samples(
                preds.boxes.values[:, 0].double() * 0 + preds.logits[:, 0].double() * 0,
                preds.boxes.cu_seqlens,
            )
            terms: dict[str, Tensor] = {}
            for name, values in outputs.items():
                match validate_output(
                    values, total.shape[0], preds.boxes.values.device
                ):
                    case Err(error=error):
                        return Err(InputError(error.kind, name, error.detail))
                    case Ok(value=values):
                        values = values.double()
                total = total + values
                terms[name] = reduce_loss(values / divisor, reduction)
            total = reduce_loss(total / divisor, reduction)
            if not bool(torch.isfinite(total).all()) or any(
                not bool(torch.isfinite(value).all()) for value in terms.values()
            ):
                return Err(
                    InputError(ErrorKind.DOMAIN, "loss", "nonfinite normalized loss")
                )
            return Ok(SetLosses(total, MappingProxyType(terms), assignment))
