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
    function: BoxFunction | nn.Module
    weight: float


class BoxTerm(nn.Module):
    """Adapt one unreduced function; assignment of modules registers their state."""

    def __init__(self, spec: TermSpec) -> None:
        """Register module components instead of hiding them in a closure."""
        super().__init__()
        self.function = spec.function
        self.weight = spec.weight

    def evaluate(self, preds: Tensor, targets: Tensor) -> object:
        """Contain nn.Module's untyped __call__ return at the adapter boundary."""
        return self.function(preds, targets)

    @override
    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        """Invoke a trusted component whose contract guarantees a tensor result."""
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
        """Freeze the component graph and register every enabled child module."""
        super().__init__()
        self.config = config
        self.terms = tuple(BoxTerm(spec) for spec in specs)
        for spec, term in zip(specs, self.terms, strict=True):
            self.add_module("term_" + spec.name, term)

    @override
    def forward(self, preds: PackedBoxes, targets: PackedBoxes) -> Tensor:
        """Evaluate trusted aligned terms in float64 with autocast locally disabled.

        User functions receive float64 aligned geometry and return `[n]` values.
        Reduction/weighting happen once; caller owns valid input/output contracts.
        Layer iteration and normalization beyond the configured reduction are external.
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
        """Validate current aligned inputs and observable callback output contracts."""
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
    """Immutable fluent assembly; calling add never modifies an existing builder."""

    terms: tuple[TermSpec, ...] = ()

    def add(
        self, name: str, function: BoxFunction | nn.Module, *, weight: float = 1.0
    ) -> BoxLossBuilder:
        """Describe a named unreduced term; build validates names and weights."""
        return BoxLossBuilder((*self.terms, TermSpec(name, function, weight)))

    def build(self, config: LossConfig) -> Result[CompositeBoxLoss, InputError]:
        """Reject duplicate/invalid names or weights and omit zero-weight terms.

        Disabled terms create no wrappers and are never invoked. Custom functions
        cannot be JSON-serialized; callers reconstruct them before loading weights.
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
    """Build a built-in from already validated settings; no parsing in forward."""
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
    """Assemble the standalone matcher with a typed custom cost collaborator."""
    return HungarianMatcher(config, cost)
