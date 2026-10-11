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
    """Shared immutable scalar weighting and reduction settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    reduction: Reduction = Reduction.MEAN
    weight: Nonnegative = 1.0


class ProbIoUConfig(LossConfig):
    """Stable float64 ProbIoU; numerical domain is fixed, not an epsilon knob."""

    kind: Literal["probiou"] = "probiou"


class VFLConfig(LossConfig):
    """Varifocal weighting for caller-supplied quality labels."""

    kind: Literal["vfl"] = "vfl"
    alpha: Nonnegative = 0.75
    gamma: Nonnegative = 2.0
    quality_weighted: bool = True


class QFLConfig(LossConfig):
    """Quality focal modulation for caller-supplied quality labels."""

    kind: Literal["qfl"] = "qfl"
    beta: Annotated[float, Field(ge=1, allow_inf_nan=False)] = 2.0


class MatcherConfig(BaseModel):
    """Bound temporary geometry and complete assignment cost storage independently."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    chunk_size: Positive = 65536
    max_pairs: Annotated[int, Field(ge=0)] = 10_000_000


class SetWeights(BaseModel):
    """Independent nonnegative weights for classification, L1, and ProbIoU."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    classification: Nonnegative = 1.0
    l1: Nonnegative = 5.0
    probiou: Nonnegative = 2.0


class SetMatcherConfig(BaseModel):
    """Matching weights and storage bounds, independent of training losses."""

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
    """Per-layer losses and matcher configuration with caller-supplied quality."""

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
    """Validate JSON-compatible settings, returning typed validation failures."""
    try:
        return Ok(model.model_validate(raw))
    except ValidationError as error:
        return Err(InputError(ErrorKind.CONFIG, model.__name__, str(error)))
