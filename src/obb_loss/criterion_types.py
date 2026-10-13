"""Immutable declarations and typed boundaries for set-prediction training."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Generic, Protocol, TypeAlias, TypeVarTuple, Unpack

from torch import Tensor

from obb_loss.result import Err, InputError, Ok, Result

Arguments = TypeVarTuple("Arguments")


@dataclass(frozen=True)
class Predictions:
    """Borrow boxes [batch, queries, 6] and logits [batch, queries, classes]."""

    boxes: Tensor[[int, int, 6]]
    logits: Tensor[[int, int, int]]


@dataclass(frozen=True)
class Targets:
    """Borrow one image's boxes [objects, 6] and int64 labels [objects]."""

    boxes: Tensor[[int, 6]]
    labels: Tensor[[int]]


@dataclass(frozen=True)
class ImagePredictions:
    """Borrow one image's boxes [queries, 6] and logits [queries, classes]."""

    boxes: Tensor[[int, 6]]
    logits: Tensor[[int, int]]


@dataclass(frozen=True)
class Match:
    """Unique, in-range int64 prediction/target indices on the input device."""

    predictions: Tensor[[int]]
    targets: Tensor[[int]]


@dataclass(frozen=True)
class LossContext:
    """Borrow batch inputs and retain differentiable gathered box pairs."""

    predictions: Predictions
    targets: tuple[Targets, ...]
    matches: tuple[Match, ...]
    predicted_boxes: Tensor[[int, 6]]
    target_boxes: Tensor[[int, 6]]


class CriterionErrorKind(Enum):
    """Separate declaration failures from assignment backend failures."""

    SPECIFICATION = "specification"
    BACKEND = "backend"


@dataclass(frozen=True)
class CriterionError:
    """Explain an invalid declaration or expected assignment failure."""

    kind: CriterionErrorKind
    field: str
    message: str


CriterionFailure: TypeAlias = InputError | CriterionError
Normalizer: TypeAlias = Callable[[LossContext], Tensor]
CostFunction: TypeAlias = Callable[
    [ImagePredictions, Targets], Result[Tensor, InputError]
]
AssignmentSolver: TypeAlias = Callable[[Tensor], Result[Match, CriterionFailure]]


@dataclass(frozen=True)
class Cost:
    """Declare a [queries, objects] cost and its independent matching weight."""

    name: str
    compute: CostFunction
    weight: float


@dataclass(frozen=True)
class Matcher:
    """Declare costs and the assignment backend, without computing on construction."""

    solver: AssignmentSolver
    costs: tuple[Cost, ...]


class LossTerm(Protocol):
    """Expose only what composition needs from a heterogeneously typed loss."""

    @property
    def name(self) -> str:
        """Identify the loss in named outputs."""
        ...

    @property
    def weight(self) -> float:
        """Scale the normalized contribution; zero disables all term work."""
        ...

    def values(self, context: LossContext) -> Result[Tensor, InputError]:
        """Project typed arguments and compute unreduced values."""
        ...

    def divisor(self, context: LossContext) -> Tensor:
        """Return a finite positive scalar on the prediction device."""
        ...


@dataclass(frozen=True)
class Loss(Generic[*Arguments]):
    """Declare a loss with statically paired input projection and compute function.

    Inputs and compute borrow tensors. Compute returns unreduced floating values
    or a metadata error. normalize returns a finite positive floating scalar on
    the prediction device. Numerical domains are the functions' responsibility.
    """

    name: str
    compute: Callable[[Unpack[Arguments]], Result[Tensor, InputError]]
    inputs: Callable[[LossContext], Result[tuple[*Arguments], InputError]]
    weight: float
    normalize: Normalizer

    def values(self, context: LossContext) -> Result[Tensor, InputError]:
        """Propagate projection failures without calling the numerical function."""
        match self.inputs(context):
            case Err() as failure:
                return failure
            case Ok(arguments):
                return self.compute(*arguments)

    def divisor(self, context: LossContext) -> Tensor:
        """Evaluate the declared normalization policy."""
        return self.normalize(context)


@dataclass(frozen=True)
class Criterion:
    """Declare the complete matching and loss objective without mutable state."""

    matcher: Matcher
    losses: tuple[LossTerm, ...]


@dataclass(frozen=True)
class LossOutput:
    """Return connected scalar losses and assignments; logging chooses detachment.

    raw contains normalized unweighted losses; weighted contains their scaled
    contributions. Disabled terms are omitted. Matches expose assignment policy.
    """

    total: Tensor
    raw: dict[str, Tensor]
    weighted: dict[str, Tensor]
    matches: tuple[Match, ...]
