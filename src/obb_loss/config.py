"""Immutable typed configuration and recoverable external parsing."""

from enum import Enum
from typing import Annotated, Literal, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
)

from obb_loss.packed import ErrorKind, InputError
from obb_loss.result import Err, Ok, Result

C = TypeVar("C", bound=BaseModel)
Nonnegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Positive = Annotated[int, Field(gt=0)]


class Reduction(Enum):
    """Explicit normalization; mean averages all unreduced elements."""

    NONE = "none"
    SUM = "sum"
    MEAN = "mean"


class LossConfig(BaseModel):
    """Configure loss reduction and weighting.

    Attributes:
        reduction: NONE retains elements; SUM adds them; MEAN averages elements,
            with empty scalar reductions returning zero.
        weight: Finite nonnegative multiplier applied after reduction; zero skips
            loss evaluation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    reduction: Reduction = Reduction.MEAN
    weight: Nonnegative = 1.0


class VFLParameters(BaseModel):
    """Configure Varifocal math independently of weighting and reduction.

    Attributes:
        alpha: Finite nonnegative negative-class BCE multiplier.
        gamma: Finite nonnegative exponent suppressing easy negatives.
        quality_weighted: Weight positive BCE by caller quality when true.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    alpha: Nonnegative = 0.75
    gamma: Nonnegative = 2.0
    quality_weighted: bool = True


class VFLConfig(LossConfig, VFLParameters):
    """Configure Varifocal modulation.

    Attributes:
        alpha: Finite nonnegative multiplier for negative-class weights.
        gamma: Finite nonnegative probability exponent controlling suppression of
            easy negatives.
        quality_weighted: Use target quality as positive-class BCE weight; false
            uses one. Reduction/global weight are inherited.
    """


class QFLParameters(BaseModel):
    """Configure Quality Focal math independently of weighting and reduction.

    Attributes:
        beta: Finite exponent >=1 controlling suppression near target quality.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    beta: Annotated[float, Field(ge=1, allow_inf_nan=False)] = 2.0


class QFLConfig(LossConfig, QFLParameters):
    """Configure Quality Focal modulation.

    Attributes:
        beta: Finite exponent >=1 on absolute quality error; controls suppression of
            predictions close to target quality. Reduction/global weight are
            inherited.
    """


class MatcherConfig(BaseModel):
    """Configure set matching costs and packed pair storage limits.

    Attributes:
        classification_weight: SetHungarianMatcher multiplier for negative target-class
            probability; geometry-only HungarianMatcher uses its supplied cost instead.
        l1_weight: Multiplier for candidate center/side L1 costs.
        probiou_weight: Multiplier for candidate ProbIoU costs.
        chunk_size: Positive maximum number of aligned pairs evaluated together;
            bounds temporary pair storage.
        max_pairs: Maximum sum of per-sample rectangular pair counts; bounds
            complete cost storage.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    classification_weight: Nonnegative = 1
    l1_weight: Nonnegative = 5
    probiou_weight: Nonnegative = 2
    chunk_size: Positive = 65536
    max_pairs: Annotated[int, Field(ge=0)] = 10_000_000


def parse_config(model: type[C], raw: JsonValue) -> Result[C, InputError]:
    """Validate external configuration into a typed model.

    Args:
        model: Pydantic configuration class selecting the schema and loss/matcher
            variant.
        raw: JSON-compatible configuration data to validate, usually a mapping of
            field names to values.

    Returns:
        Ok(validated model) or Err(InputError) containing Pydantic validation
        diagnostics.
    """
    try:
        return Ok(model.model_validate(raw))
    except ValidationError as error:
        return Err(InputError(ErrorKind.CONFIG, model.__name__, str(error)))


class TargetCount(Enum):
    """Choose per-sample or local-batch target-count normalization."""

    PER_SAMPLE = "per_sample"
    PER_BATCH = "per_batch"


class SetCriterionConfig(BaseModel):
    """Configure one layer's matching, loss math and target-count normalization.

    Attributes:
        matcher: Candidate storage limits and independent matching multipliers.
        classification: VFL, QFL or disabled classification; quality is supplied by
            overriding classification_quality after matching.
        classification_weight: Multiplier for raw classification sums.
        l1_weight: Multiplier for matched center/side L1 sums; zero skips evaluation.
        probiou_weight: Multiplier for matched ProbIoU sums; zero skips evaluation.
        vfl: Varifocal modulation parameters.
        qfl: Quality Focal modulation parameters.
        normalizer: Per-sample/batch count policy, or a positive shared scalar divisor.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    matcher: MatcherConfig = MatcherConfig()
    classification: Literal["vfl", "qfl", "none"] = "vfl"
    classification_weight: Nonnegative = 1
    l1_weight: Nonnegative = 5
    probiou_weight: Nonnegative = 2
    vfl: VFLParameters = VFLParameters()
    qfl: QFLParameters = QFLParameters()
    normalizer: TargetCount | Annotated[float, Field(gt=0, allow_inf_nan=False)] = (
        TargetCount.PER_BATCH
    )
