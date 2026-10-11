"""Bounded seeded numerical probes for loss forward and backward stability."""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import assert_never, cast

import torch
import torch.autograd
from einops import rearrange
from torch import Tensor

from obb_loss import Err, Ok, Result
from tests.math_helpers import stable_probiou_loss


class InputDtype(Enum):
    """Explicit input precision independently of internal float64 computation."""

    FP16 = "float16"
    BF16 = "bfloat16"
    FP32 = "float32"
    FP64 = "float64"


def torch_dtype(value: InputDtype) -> torch.dtype:
    """Translate a closed precision selection without string reflection."""
    # Native Torch stubs currently expose dtype constants as Any.
    match value:
        case InputDtype.FP16:
            return cast("torch.dtype", torch.float16)
        case InputDtype.BF16:
            return cast("torch.dtype", torch.bfloat16)
        case InputDtype.FP32:
            return cast("torch.dtype", torch.float32)
        case InputDtype.FP64:
            return cast("torch.dtype", torch.float64)
        case _:
            assert_never(value)


@dataclass(frozen=True)
class StressSettings:
    """Specify bounded test distributions and a reproducible seed."""

    samples: int = 10000
    chunk_size: int = 512
    seed: int = 901
    device: str = "cpu"
    dtype: InputDtype = InputDtype.FP64
    min_log_scale: float = -30
    max_log_scale: float = 30
    max_log_aspect: float = 24
    quiet: bool = False


@dataclass
class StressReport:
    """Count finite outputs, input quantization and gradient-cast failures."""

    evaluated: int = 0
    rejected_input_quantization: int = 0
    forward_failures: int = 0
    backward_failures: int = 0
    gradient_cast_overflows: int = 0
    unexplained_backward_failures: int = 0
    first_failure_sample: Result[int, str] = field(
        default_factory=lambda: Err("No unexplained failure observed")
    )
    max_chunk_seen: int = 0
    analytic_max_error: float = 0


def sample_pair(
    count: int, settings: StressSettings, start: int
) -> tuple[Tensor, Tensor]:
    """Generate one bounded chunk: thirds equality, analytic shifts, general geometry.

    Scale/aspect are log-uniform. Analytic/equality cases are axis aligned; general
    pairs have random directions and side ratios. Chunk seed is seed+global start;
    reproduce with the same settings, including chunk_size.
    """
    generator = torch.Generator(device=settings.device).manual_seed(
        settings.seed + start
    )
    random = torch.rand(
        (count, 8), dtype=torch.float64, device=settings.device, generator=generator
    )
    scale = 10 ** (
        settings.min_log_scale
        + random[:, 0] * (settings.max_log_scale - settings.min_log_scale)
    )
    aspect = 10 ** (random[:, 1] * settings.max_log_aspect)
    angle = random[:, 2] * 6.283185307179586
    groups = torch.arange(start, start + count, device=settings.device) % 3
    angle = torch.where(groups == 2, angle, 0)
    first = torch.stack(
        (
            torch.zeros_like(scale),
            torch.zeros_like(scale),
            scale,
            scale / aspect,
            angle.cos(),
            angle.sin(),
        ),
        dim=-1,
    )
    second = first.clone()
    second[:, 1] = torch.where(groups == 1, 0.1 * scale / aspect, 0)
    general = groups == 2
    second[:, :2] += rearrange(general * scale, "n -> n 1") * random[:, 3:5]
    ratio = 10 ** (random[:, 5:7] * 2 - 1)
    second[:, 2:4] *= torch.where(rearrange(general, "n -> n 1"), ratio, 1)
    angle2 = angle + general * random[:, 7]
    second[:, 4], second[:, 5] = angle2.cos(), angle2.sin()
    dtype = torch_dtype(settings.dtype)
    return first.to(dtype), second.to(dtype)


def _valid_geometry(first: Tensor, second: Tensor) -> Tensor:
    """Distinguish input cast underflow/overflow from arithmetic failures."""
    return torch.stack(
        (
            torch.isfinite(first).all(-1),
            torch.isfinite(second).all(-1),
            (first[:, 2:4] > 0).all(-1),
            (second[:, 2:4] > 0).all(-1),
        )
    ).all(0)


def _silent_progress(done: int, total: int) -> None:
    """Provide an I/O-free default for tests and programmatic simulation."""


def _backward_diagnostics(
    loss: Tensor, first: Tensor, second: Tensor
) -> tuple[Tensor, Tensor]:
    """Separate nonfinite leaf gradients from float64-to-leaf cast overflow."""
    gradients = torch.autograd.grad(loss.sum(), (first, second))
    backward_bad = ~torch.isfinite(gradients[0]).all(-1) | ~torch.isfinite(
        gradients[1]
    ).all(-1)
    first64, second64 = (
        first.detach().double().requires_grad_(),
        second.detach().double().requires_grad_(),
    )
    references = torch.autograd.grad(
        stable_probiou_loss(first64, second64).sum(), (first64, second64)
    )
    limit = torch.finfo(first.dtype).max
    outside = (references[0].abs() > limit).any(-1) | (references[1].abs() > limit).any(
        -1
    )
    finite = torch.isfinite(references[0]).all(-1) & torch.isfinite(references[1]).all(
        -1
    )
    return backward_bad, backward_bad & outside & finite


def _analytic_error(loss: Tensor, indices: Tensor, dtype: InputDtype) -> float:
    """Measure center-shift accuracy only where float64 preserves the input geometry."""
    if dtype != InputDtype.FP64:
        return 0.0
    analytic = loss[indices % 3 == 1].detach()
    if analytic.numel() == 0:
        return 0.0
    expected = torch.sqrt(
        -torch.expm1(torch.tensor(-0.015, dtype=torch.float64, device=loss.device))
    )
    return float((analytic - expected).abs().max())


def simulate(
    settings: StressSettings, progress: Callable[[int, int], None] = _silent_progress
) -> StressReport:
    """Stream chunks and classify nonfinite backward by float64 reference gradients.

    The float64 gradient comparison diagnoses cast overflow, not independent math.
    Analytic accuracy is checked for representable float64 center-shift cases.
    Memory is O(chunk_size); only one first-failure index and counters are retained.
    """
    report = StressReport()
    for start in range(0, settings.samples, settings.chunk_size):
        count = min(settings.chunk_size, settings.samples - start)
        first, second = sample_pair(count, settings, start)
        valid = _valid_geometry(first, second)
        report.max_chunk_seen = max(report.max_chunk_seen, count)
        report.rejected_input_quantization += int((~valid).sum())
        indices = torch.arange(start, start + count, device=first.device)[valid]
        first, second = first[valid].requires_grad_(), second[valid].requires_grad_()
        loss = stable_probiou_loss(first, second)
        forward_bad = ~torch.isfinite(loss) | (loss < 0) | (loss > 1)
        backward_bad, cast_overflow = _backward_diagnostics(loss, first, second)
        report.evaluated += loss.numel()
        report.forward_failures += int(forward_bad.sum())
        report.backward_failures += int(backward_bad.sum())
        report.gradient_cast_overflows += int(cast_overflow.sum())
        unexplained = backward_bad & ~cast_overflow
        report.unexplained_backward_failures += int(unexplained.sum())
        failures = indices[forward_bad | unexplained]
        match report.first_failure_sample:
            case Err() if failures.numel():
                report.first_failure_sample = Ok(int(failures[0]))
        report.analytic_max_error = max(
            report.analytic_max_error, _analytic_error(loss, indices, settings.dtype)
        )
        progress(start + count, settings.samples)
    return report
