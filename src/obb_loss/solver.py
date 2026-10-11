"""Contain SciPy's CPU assignment boundary and untyped array outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from einops import rearrange
from scipy.optimize import linear_sum_assignment

from obb_loss.packed import ErrorKind, InputError
from obb_loss.result import Err, Ok, Result

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from torch import Tensor

    from obb_loss.matching import Assignment
    from obb_loss.pairwise import PackedCosts


def solve_costs(costs: PackedCosts) -> Result[Assignment, InputError]:
    """Solve each packed sample rectangle on the CPU.

    Costs transfer once to CPU. SciPy allocates its own working storage; unexpected vendor failures propagate.

    Args:
        costs: Finite row-major cost values and ordered rectangular metadata; source
            offsets identify global prediction/reference rows.

    Returns:
        Ok(Assignment) with global int64 indices on the cost device, or
        Err(InputError) for recoverable solver/input response failures.
    """
    from obb_loss.matching import Assignment

    host = costs.values.detach().double().cpu()
    pred_indices: list[Tensor] = []
    target_indices: list[Tensor] = []
    offsets = [0]
    for block in costs.blocks:
        if block.rows == 0 or block.columns == 0:
            offsets.append(offsets[-1])
            continue
        matrix = rearrange(
            host[block.start : block.end], "(r c) -> r c", r=block.rows, c=block.columns
        )
        try:
            rows: NDArray[np.int64]
            columns: NDArray[np.int64]
            rows, columns = linear_sum_assignment(matrix.numpy())
        except ValueError as error:
            return Err(InputError(ErrorKind.DOMAIN, "assignment", str(error)))
        if not isinstance(rows, np.ndarray) or not isinstance(columns, np.ndarray):
            return Err(
                InputError(
                    ErrorKind.CALLBACK, "assignment", "solver must return arrays"
                )
            )
        if (
            rows.ndim != 1
            or columns.shape != rows.shape
            or rows.dtype != np.int64
            or columns.dtype != np.int64
        ):
            return Err(
                InputError(
                    ErrorKind.CALLBACK,
                    "assignment",
                    "solver must return aligned int64 vectors",
                )
            )
        pred_indices.append(torch.from_numpy(rows) + block.pred_start)
        target_indices.append(torch.from_numpy(columns) + block.target_start)
        offsets.append(offsets[-1] + rows.size)
    device = costs.values.device
    first = (
        torch.cat(pred_indices).to(device)
        if pred_indices
        else torch.empty(0, dtype=torch.int64, device=device)
    )
    second = (
        torch.cat(target_indices).to(device)
        if target_indices
        else torch.empty(0, dtype=torch.int64, device=device)
    )
    return Ok(
        Assignment(
            first, second, torch.tensor(offsets, dtype=torch.int64, device=device)
        )
    )
