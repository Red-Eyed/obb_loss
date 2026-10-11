"""Typed assembly of custom aligned functions and registered module components."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple, assert_never

import torch
from torch import Tensor, nn
from typing_extensions import override

from obb_loss.config import (
    LossConfig,
    MatcherConfig,
    ProbIoUConfig,
    QFLConfig,
    VFLConfig,
)
from obb_loss.criterion import ProbIoULoss, validate_alignment
from obb_loss.losses.quality_focal import QualityFocalLoss
from obb_loss.losses.varifocal import VarifocalLoss
from obb_loss.matching import HungarianMatcher
from obb_loss.packed import ErrorKind, InputError, PackedBoxes
from obb_loss.pairwise import validate_output
from obb_loss.reduction import reduce_loss
from obb_loss.result import Err, Ok, Result

if TYPE_CHECKING:
    from obb_loss.protocols import BoxFunction


class TermSpec(NamedTuple):
    """Keep serializable settings separate from live function/module references."""

    name: str
    function: BoxFunction
    weight: float


class BoxTerm(nn.Module):
    """Adapt one unreduced function; assignment of modules registers their state."""

    def __init__(self, spec: TermSpec) -> None:
        """Register one weighted custom box term.

        Args:
            spec: Term name, aligned function/module, and finite nonnegative multiplier;
                module state is registered as a child.
        """
        super().__init__()
        self.function = spec.function
        self.weight = spec.weight

    def evaluate(self, preds: Tensor, targets: Tensor) -> Tensor:
        """Call the configured unreduced box term.

        Args:
            preds: Predicted aligned boxes in (cx, cy, width, height, cos(theta),
                sin(theta)) order; positive sides and nonzero directions, in shared
                coordinate units. Float64 [pairs, 6] geometry.
            targets: Reference boxes in the same order and units as preds, paired
                elementwise with predictions. Float64 [pairs, 6] geometry.

        Returns:
            The callback tensor before weighting or reduction; custom callback failures
            propagate.
        """
        return self.function(preds, targets)

    @override
    # Torch stubs declare variadic forward; this module intentionally narrows it.
    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:  # pyrefly: ignore[bad-override]
        """Evaluate and weight a trusted box term.

        A callback returning a non-tensor violates the trusted contract and raises TypeError.

        Args:
            preds: Predicted aligned boxes in (cx, cy, width, height, cos(theta),
                sin(theta)) order; positive sides and nonzero directions, in shared
                coordinate units. Float64 [pairs, 6] geometry.
            targets: Reference boxes in the same order and units as preds, paired
                elementwise with predictions. Float64 [pairs, 6] geometry.

        Returns:
            Unreduced callback values multiplied by the configured term weight.
        """
        match self.evaluate(preds, targets):
            case Tensor() as values:
                return values * self.weight
            case _:
                raise TypeError(
                    "Trusted box component violated its tensor return contract"
                )


class CompositeBoxLoss(nn.Module):
    """Weighted custom terms with one reduction and the packed box forward contract."""

    def __init__(self, specs: tuple[TermSpec, ...], config: LossConfig) -> None:
        """Register enabled terms and configure their shared reduction.

        Args:
            specs: Validated uniquely named loss functions/modules with finite
                nonnegative term weights; modules register their parameters and buffers.
            config: Final reduction over matched pairs and global multiplier applied
                after combining weighted terms.
        """
        super().__init__()
        self.config = config
        self.terms = tuple(BoxTerm(spec) for spec in specs)
        for spec, term in zip(specs, self.terms, strict=True):
            self.add_module("term_" + spec.name, term)

    @override
    # Torch stubs declare variadic forward; this module intentionally narrows it.
    def forward(self, preds: PackedBoxes, targets: PackedBoxes) -> Tensor:  # pyrefly: ignore[bad-override]
        """Evaluate custom loss terms on aligned packed box pairs.

        Callbacks receive float64 [pairs, 6] tensors and return unreduced [pairs] losses. Inputs and callback contracts are trusted. Autocast is disabled locally. Custom code bugs propagate.

        Args:
            preds: Predicted packed geometry; sample boundaries and box order must match
                targets.
            targets: Already matched reference geometry in the same units, with the same
                cumulative offsets as preds.

        Returns:
            Float64 [pairs] values for NONE, or a weighted scalar for SUM/MEAN; empty
            scalar reductions are connected zeros.
        """
        with torch.autocast(preds.boxes.device.type, enabled=False):
            first, second = preds.boxes.double(), targets.boxes.double()
            values = first[:, 0] * 0 + second[:, 0] * 0
            for term in self.terms:
                values = values + term(first, second)
            return reduce_loss(values, self.config.reduction) * self.config.weight

    def checked(
        self, preds: PackedBoxes, targets: PackedBoxes
    ) -> Result[Tensor, InputError]:
        """Evaluate custom loss terms on aligned packed box pairs.

        Callbacks receive float64 [pairs, 6] tensors and return unreduced [pairs] losses. Fresh inputs and callback shape/device/finiteness are validated. Autocast is disabled locally. Custom code bugs propagate.

        Args:
            preds: Predicted packed geometry; sample boundaries and box order must match
                targets.
            targets: Already matched reference geometry in the same units, with the same
                cumulative offsets as preds.

        Returns:
            Ok(weighted/reduced loss) or Err(InputError) for invalid aligned input or
            observable callback output contracts.
        """
        match validate_alignment(preds, targets):
            case Err() as failure:
                return failure
            case Ok():
                pass
        with torch.autocast(preds.boxes.device.type, enabled=False):
            first, second = preds.boxes.double(), targets.boxes.double()
            values = first[:, 0] * 0 + second[:, 0] * 0
            for term in self.terms:
                match validate_output(
                    term.evaluate(first, second), first.shape[0], first.device
                ):
                    case Err() as failure:
                        return failure
                    case Ok(value=output):
                        values = values + output * term.weight
            return Ok(reduce_loss(values, self.config.reduction) * self.config.weight)


class BoxLossBuilder(NamedTuple):
    """Describe an immutable sequence of weighted custom loss terms.

    Attributes:
        terms: Ordered term specifications; add returns a new builder, and build
            validates names/weights and registers enabled modules.
    """

    terms: tuple[TermSpec, ...] = ()

    def add(
        self, name: str, function: BoxFunction, *, weight: float = 1.0
    ) -> BoxLossBuilder:
        """Describe an additional named loss term without changing this builder.

        Args:
            name: Unique Python identifier used to register the term and name its module
                state.
            function: Aligned box loss callable or module, returning finite [pairs]
                values from two [pairs, 6] float64 tensors. Module parameters and
                buffers are registered on build.
            weight: Finite nonnegative multiplier applied to this term before the shared
                reduction; zero omits the term entirely.

        Returns:
            A new builder containing the additional term. Names/weights are validated by
            build.
        """
        return BoxLossBuilder((*self.terms, TermSpec(name, function, weight)))

    def build(self, config: LossConfig) -> Result[CompositeBoxLoss, InputError]:
        """Validate and assemble a weighted composite loss.

        Disabled terms are not instantiated or invoked. Config serialization cannot reconstruct callbacks; retain or recreate them separately.

        Args:
            config: Reduction over combined pair losses and final global weight; zero
                global weight omits every term.

        Returns:
            Ok(CompositeBoxLoss) or Err(InputError) for duplicate/invalid names or
            negative/nonfinite term weights.
        """
        names: set[str] = set()
        enabled: list[TermSpec] = []
        for spec in self.terms:
            if not spec.name.isidentifier() or spec.name in names:
                return Err(
                    InputError(ErrorKind.CONFIG, "name", "unique identifier required")
                )
            if not math.isfinite(spec.weight) or spec.weight < 0:
                return Err(
                    InputError(
                        ErrorKind.CONFIG, "weight", "finite nonnegative weight required"
                    )
                )
            names.add(spec.name)
            if spec.weight > 0 and config.weight > 0:
                enabled.append(spec)
        return Ok(CompositeBoxLoss(tuple(enabled), config))


def build_loss(
    config: ProbIoUConfig | VFLConfig | QFLConfig,
) -> ProbIoULoss | VarifocalLoss | QualityFocalLoss:
    """Construct a configured built-in loss module.

    Args:
        config: Validated ProbIoU, VFL, or QFL settings selecting the formulation,
            its numerical parameters, reduction, and global weight.

    Returns:
        The corresponding ProbIoULoss, VarifocalLoss, or QualityFocalLoss module.
    """
    match config:
        case ProbIoUConfig():
            return ProbIoULoss(config)
        case VFLConfig():
            return VarifocalLoss(config)
        case QFLConfig():
            return QualityFocalLoss(config)
        case _:
            assert_never(config)


def build_matcher(config: MatcherConfig, cost: BoxFunction) -> HungarianMatcher:
    """Construct a geometry matcher with a custom cost function.

    Args:
        config: Validated temporary-pair and full-cost storage limits.
        cost: Aligned [pairs, 6] -> [pairs] finite same-device cost callable; lower
            costs are preferred.

    Returns:
        A HungarianMatcher using the supplied limits and cost.
    """
    return HungarianMatcher(config, cost)
