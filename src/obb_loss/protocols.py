"""Narrow structural contracts for user-defined aligned geometry functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from shape_extensions import IntVar
    from torch import Tensor

    N = IntVar("N")


class BoxFunction(Protocol):
    """Pure unreduced aligned box loss/cost; library owns reduction and weighting."""

    def __call__(
        self,
        preds: Tensor[[N, 6]],
        targets: Tensor[[N, 6]],
        /,
    ) -> Tensor[[N]]:
        """Evaluate a custom unreduced box loss or matching cost.

        Args:
            preds: Predicted aligned boxes in (cx, cy, width, height, cos(theta),
                sin(theta)) order; positive sides and nonzero directions, in shared
                coordinate units. Shape [pairs, 6].
            targets: Reference boxes in the same order and units as preds, paired
                elementwise with predictions. Same [pairs, 6] shape and device.

        Returns:
            Finite same-device floating [pairs] values; no input mutation. Matching
            interprets lower values as preferable.
        """
        ...
