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


class ProbIoUConfig(LossConfig):
    """Configure the stable ProbIoU loss.

    Attributes:
        kind: Serialization discriminator selecting stable ProbIoU. Inherits
            reduction and global weight from LossConfig.
    """

    kind: Literal["probiou"] = "probiou"


class WeightConfig(BaseModel):
    """Configure one callback's multiplier.

    Attributes:
        weight: Finite nonnegative multiplier; zero omits the callback from composition.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
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
        kind: Serialization discriminator selecting Varifocal loss.
        alpha: Finite nonnegative multiplier for negative-class weights.
        gamma: Finite nonnegative probability exponent controlling suppression of
            easy negatives.
        quality_weighted: Use target quality as positive-class BCE weight; false
            uses one. Reduction/global weight are inherited.
    """

    kind: Literal["vfl"] = "vfl"


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
        kind: Serialization discriminator selecting Quality Focal loss.
        beta: Finite exponent >=1 on absolute quality error; controls suppression of
            predictions close to target quality. Reduction/global weight are
            inherited.
    """

    kind: Literal["qfl"] = "qfl"


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
