"""Immutable typed configuration and recoverable external parsing."""

from enum import Enum
from typing import Annotated, Literal, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
    model_validator,
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


class ProbIoUConfig(LossConfig):
    """Configure the stable ProbIoU loss.

    Attributes:
        kind: Serialization discriminator selecting stable ProbIoU. Inherits
            reduction and global weight from LossConfig.
    """

    kind: Literal["probiou"] = "probiou"


class VFLConfig(LossConfig):
    """Configure Varifocal modulation.

    Attributes:
        kind: Serialization discriminator selecting Varifocal loss.
        alpha: Finite nonnegative multiplier for negative-class weights.
        gamma: Finite nonnegative probability exponent controlling suppression of
            easy negatives.
        quality_weighted: Use target quality as positive-class BCE weight; false
            uses one. Reduction/global weight are inherited.
    """

    kind: Literal["vfl"] = "vfl"
    alpha: Nonnegative = 0.75
    gamma: Nonnegative = 2.0
    quality_weighted: bool = True


class QFLConfig(LossConfig):
    """Configure Quality Focal modulation.

    Attributes:
        kind: Serialization discriminator selecting Quality Focal loss.
        beta: Finite exponent >=1 on absolute quality error; controls suppression of
            predictions close to target quality. Reduction/global weight are
            inherited.
    """

    kind: Literal["qfl"] = "qfl"
    beta: Annotated[float, Field(ge=1, allow_inf_nan=False)] = 2.0


class MatcherConfig(BaseModel):
    """Configure packed matching memory limits.

    Attributes:
        chunk_size: Positive maximum number of aligned pairs evaluated together;
            bounds temporary pair storage.
        max_pairs: Maximum sum of per-sample rectangular pair counts; bounds
            complete cost storage.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    chunk_size: Positive = 65536
    max_pairs: Annotated[int, Field(ge=0)] = 10_000_000


class SetWeights(BaseModel):
    """Configure independent component multipliers.

    Attributes:
        classification: Nonnegative multiplier for classification cost or loss,
            according to its owning config.
        l1: Nonnegative multiplier for center/side L1 cost or loss.
        probiou: Nonnegative multiplier for ProbIoU cost or loss. Zero disables the
            corresponding computation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    classification: Nonnegative = 1.0
    l1: Nonnegative = 5.0
    probiou: Nonnegative = 2.0


class SetMatcherConfig(BaseModel):
    """Configure set assignment.

    Attributes:
        weights: Class/L1/ProbIoU assignment weights, independent of training
            weights; at least one must be positive.
        storage: Temporary chunk-size and complete-pair storage limits.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    weights: SetWeights = SetWeights()
    storage: MatcherConfig = MatcherConfig()

    @model_validator(mode="after")
    def require_cost(self) -> "SetMatcherConfig":
        """Require at least one enabled assignment cost."""
        if self.weights.classification + self.weights.l1 + self.weights.probiou == 0:
            raise ValueError("at least one matching cost must be enabled")
        return self


class SetCriterionConfig(BaseModel):
    """Configure one layer of set supervision.

    Attributes:
        weights: Class/L1/ProbIoU training multipliers; all-zero loss weights are
            allowed.
        matcher: Independent assignment weights and cost-storage limits.
        classification: VFL/QFL formulation settings; weight/reduction must retain
            defaults because the set criterion owns both operations.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    weights: SetWeights = SetWeights()
    matcher: SetMatcherConfig = SetMatcherConfig()
    classification: Annotated[VFLConfig | QFLConfig, Field(discriminator="kind")] = (
        VFLConfig()
    )

    @model_validator(mode="after")
    def require_unweighted_classification(self) -> "SetCriterionConfig":
        """Keep weighting and normalization exclusively in the set criterion."""
        if (
            self.classification.weight != 1
            or self.classification.reduction != Reduction.MEAN
        ):
            raise ValueError(
                "classification weight/reduction are controlled by the set criterion"
            )
        return self


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
