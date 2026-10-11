"""Typed weighted callbacks shared by set losses and matching costs."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Generic, NamedTuple, Protocol, TypeVar

import torch
from torch import Tensor, nn
from typing_extensions import override

from obb_loss.config import WeightConfig
from obb_loss.packed import ErrorKind, InputError
from obb_loss.pairwise import validate_output
from obb_loss.result import Err, Ok, Result

if TYPE_CHECKING:
    from collections.abc import Mapping

_DEFAULT_WEIGHT = WeightConfig()


class CallbackContext(Protocol):
    """Provide the output layout and a differentiable zero for callback composition."""

    @property
    def output_count(self) -> int:
        """Return the required sample or aligned-pair output count."""
        ...

    @property
    def device(self) -> torch.device:
        """Return the required output device without allocating tensors."""
        ...

    def zero(self) -> Tensor[[int]]:
        """Return same-device float64 zeros connected to the prediction tensors."""
        ...


C = TypeVar("C", bound=CallbackContext)
C_contra = TypeVar("C_contra", bound=CallbackContext, contravariant=True)


class LossFunction(Protocol[C_contra]):
    """Evaluate weighted or unweighted values without normalization or reduction."""

    def __call__(self, context: C_contra, /) -> Result[Tensor[[int]], InputError]:
        """Return finite same-device values matching the context's zero layout.

        Args:
            context: Borrowed inputs and pair/sample ownership for this evaluation.

        Returns:
            Ok(one value per sample or cost pair), or a typed recoverable failure.
            Callback implementation bugs propagate; inputs must not be mutated.
        """
        ...


class DefaultComposition(Enum):
    """Select built-in callbacks when no explicit composition is supplied."""

    BUILTINS = "builtins"


class NamedLoss(NamedTuple):
    """Carry one callback's name and unreduced weighted output.

    Attributes:
        name: Caller-selected term name used in reports and error diagnostics.
        values: Same-device floating vector; normalization belongs to the caller.
    """

    name: str
    values: Tensor[[int]]


class WeightedLoss(nn.Module, Generic[C]):
    """Apply a validated multiplier while registering a callback module's state."""

    def __init__(
        self, function: LossFunction[C], config: WeightConfig = _DEFAULT_WEIGHT
    ) -> None:
        """Bind an unnormalized callback to its own weighting policy.

        Args:
            function: Function or typed module returning Ok(values) or Err(InputError).
                Modules become registered children, retaining hooks and trainable state.
            config: Immutable finite nonnegative multiplier; zero skips the callback.
        """
        super().__init__()
        self.function = function
        self.config = config

    @override
    # Native Torch stubs declare variadic forward; weighted callbacks narrow it.
    def forward(self, context: C) -> Result[Tensor[[int]], InputError]:  # pyrefly: ignore[bad-override]
        """Evaluate and weight the callback, or return connected zeros when disabled.

        Args:
            context: Borrowed inputs defining the callback's sample or pair layout.

        Returns:
            Ok(weighted values) or the callback's typed failure. Final output shape,
            dtype, device, and finiteness are validated by the owning composition.
        """
        if self.config.weight == 0:
            return Ok(context.zero())
        match self.function(context):
            case Err() as failure:
                return failure
            case Ok(value=values):
                if not isinstance(values, Tensor) or not values.is_floating_point():
                    return Err(
                        InputError(
                            ErrorKind.CALLBACK, "output", "expected floating tensor"
                        )
                    )
                return Ok(values.double() * self.config.weight)
            case _:
                return Err(
                    InputError(ErrorKind.CALLBACK, "callback", "expected Ok or Err")
                )


class LossComposition(nn.Module, Generic[C]):
    """Own registered callbacks and validate their observable vector contracts."""

    def __init__(self, functions: Mapping[str, LossFunction[C]]) -> None:
        """Copy names/callbacks and omit explicitly disabled weighted terms.

        Args:
            functions: Named unnormalized callbacks; mapping mutations after
                construction do not change the selection. Module state is registered.
        """
        super().__init__()
        terms: list[tuple[str, LossFunction[C]]] = []
        for name, function in functions.items():
            if isinstance(function, WeightedLoss) and function.config.weight == 0:
                continue
            if isinstance(function, nn.Module):
                self.add_module(f"term_{len(terms)}", function)
            terms.append((name, function))
        self.terms = tuple(terms)

    @override
    # Native Torch stubs declare variadic forward; this composition narrows it.
    def forward(self, context: C) -> Result[tuple[NamedLoss, ...], InputError]:  # pyrefly: ignore[bad-override]
        """Validate each named callback and retain its unreduced weighted values.

        Args:
            context: Borrowed inputs defining the required vector count and device.

        Returns:
            Ok(named float64 vectors) or a typed output/callback error prefixed with
            the caller's term name. No normalization or reduction is applied.
        """
        values: list[NamedLoss] = []
        for name, function in self.terms:
            match function(context):
                case Err(error=error):
                    return Err(
                        InputError(error.kind, f"{name}.{error.field}", error.detail)
                    )
                case Ok(value=output):
                    pass
                case _:
                    return Err(
                        InputError(ErrorKind.CALLBACK, name, "expected Ok or Err")
                    )
            match validate_output(output, context.output_count, context.device):
                case Err(error=error):
                    return Err(InputError(error.kind, name, error.detail))
                case Ok(value=output):
                    values.append(NamedLoss(name, output.double()))
        return Ok(tuple(values))
