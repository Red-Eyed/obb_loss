"""Narrow structural contracts for user-defined aligned geometry functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor


class BoxFunction(Protocol):
    """Pure unreduced aligned box loss/cost; library owns reduction and weighting."""

    def __call__(
        self,
        preds: Float[Tensor, "n 6"],  # noqa: F722
        targets: Float[Tensor, "n 6"],  # noqa: F722
        /,
    ) -> Float[Tensor, "n"]:  # noqa: F821, UP037
        """Return finite `[n]` same-device floating values without input mutation."""
        ...
