"""CPU Hungarian assignment boundary using optional SciPy.

Rectangular assignment: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
Set prediction: Carion et al., https://arxiv.org/abs/2005.12872.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, cast

import torch

from obb_loss.criterion_core import empty_match
from obb_loss.criterion_types import (
    CriterionError,
    CriterionErrorKind,
    CriterionFailure,
    Match,
)
from obb_loss.result import Err, ErrorKind, InputError, Ok, Result

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    from numpy.typing import NDArray
    from torch import Tensor

    ScipySolver = Callable[
        [NDArray[np.float64]], tuple[NDArray[np.int64], NDArray[np.int64]]
    ]


def hungarian_assignment(costs: Tensor) -> Result[Match, CriterionFailure]:
    """Minimize finite [queries, objects] costs with at most min(Q,N) pairs.

    Borrow costs, detach and transfer to CPU float64 for SciPy, then return int64
    indices on the original device. CUDA use synchronizes. Empty matrices bypass
    SciPy. Missing optional dependencies and infeasible backend costs return Err.
    No value scans are added; SciPy may reject nonfinite or infeasible inputs.
    """
    if costs.ndim != 2:
        return Err(InputError(ErrorKind.SHAPE, "costs", "[queries, objects] required"))
    if not costs.is_floating_point():
        return Err(InputError(ErrorKind.DTYPE, "costs", "floating tensor required"))
    if costs.device.type not in ("cpu", "cuda"):
        return Err(InputError(ErrorKind.DEVICE, "costs", "CPU/CUDA required"))
    if 0 in costs.shape:
        return Ok(empty_match(costs))
    try:
        # SciPy's dynamic import is isolated here; the public solver contract is typed.
        solver = cast(
            "ScipySolver", import_module("scipy.optimize").linear_sum_assignment
        )
    except ModuleNotFoundError as error:
        if error.name not in ("scipy", "scipy.optimize"):
            raise
        return Err(
            CriterionError(
                CriterionErrorKind.BACKEND,
                "scipy",
                "install obb-loss[matching] for Hungarian assignment",
            )
        )
    matrix: NDArray[np.float64] = (
        costs.detach().to(device="cpu", dtype=torch.float64).numpy()
    )
    try:
        rows, columns = solver(matrix)
    except ValueError as error:
        return Err(CriterionError(CriterionErrorKind.BACKEND, "assignment", str(error)))
    return Ok(
        Match(
            torch.from_numpy(rows).to(costs.device),
            torch.from_numpy(columns).to(costs.device),
        )
    )
