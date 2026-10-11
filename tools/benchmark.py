"""Explicit wall-clock stages for CPU/CUDA loss construction and assignment."""

from collections.abc import Callable
from datetime import UTC, datetime
from time import perf_counter
from typing import Annotated

import torch
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict

from obb_loss import (
    HungarianMatcher,
    MatcherConfig,
    PackedBoxes,
    ProbIoULoss,
    pairwise_costs,
)
from obb_loss.result import Ok
from obb_loss.solver import solve_costs
from tools.stress import StressSettings, sample_pair


class BenchmarkSettings(BaseSettings):
    """Specify workload and iteration count rather than hidden benchmark limits."""

    model_config = SettingsConfigDict(cli_implicit_flags=True, cli_kebab_case=True)

    images: Annotated[int, Field(gt=0)] = 4
    queries: Annotated[int, Field(gt=0)] = 128
    targets: Annotated[int, Field(gt=0)] = 32
    repeats: Annotated[int, Field(gt=0)] = 5
    device: str = "cpu"
    chunk_size: Annotated[int, Field(gt=0)] = 65536
    quiet: bool = False

    def cli_cmd(self) -> None:
        """Warm up and report stage timing, complete cost storage, and CUDA peak memory."""
        print(benchmark(self).model_dump_json(indent=2))


class BenchmarkReport(BaseModel):
    """Measured stage times; CPU solver includes explicit input transfer."""

    timestamp: datetime
    settings: BenchmarkSettings
    torch_version: str
    seconds: dict[str, float]
    cost_bytes: int
    cuda_peak_bytes: int


def _time(call: Callable[[], object], repeats: int, device: torch.device) -> float:
    """Synchronize CUDA around each measured stage and return mean elapsed time."""
    call()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = perf_counter()
    for _ in range(repeats):
        call()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (perf_counter() - start) / repeats


def benchmark(settings: BenchmarkSettings) -> BenchmarkReport:
    """Measure bounded ordinary geometry without claiming cross-hardware performance."""
    import sys

    device = torch.device(settings.device)
    count = settings.images * settings.queries
    geometry = StressSettings(
        samples=count,
        device=settings.device,
        min_log_scale=-1,
        max_log_scale=1,
        max_log_aspect=1,
    )
    first, second = sample_pair(count, geometry, 0)
    pred_cu = torch.arange(settings.images + 1, device=device) * settings.queries
    target_count = settings.images * settings.targets
    target_boxes, _ = sample_pair(target_count, geometry, 1000)
    target_cu = torch.arange(settings.images + 1, device=device) * settings.targets
    preds, targets = PackedBoxes(first, pred_cu), PackedBoxes(target_boxes, target_cu)
    aligned = PackedBoxes(second, pred_cu)
    config = MatcherConfig(chunk_size=settings.chunk_size)
    matcher, criterion = HungarianMatcher(config), ProbIoULoss()
    with torch.no_grad():
        costs = pairwise_costs(preds, targets, config)
    assert isinstance(costs, Ok)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    stages: dict[str, Callable[[], object]] = {
        "trusted_loss": lambda: criterion.forward(preds, aligned),
        "checked_loss": lambda: criterion.checked(preds, aligned),
        "pairwise_costs": lambda: pairwise_costs(preds, targets, config),
        "cpu_solver_with_transfer": lambda: solve_costs(costs.value),
        "complete_matcher": lambda: matcher.forward(preds, targets),
    }

    def forward_backward() -> None:
        """Include autograd and clear accumulated leaf gradients between repeats."""
        with torch.enable_grad():
            first.requires_grad_()
            first.grad = None
            criterion.forward(preds, aligned).backward()

    stages["loss_forward_backward"] = forward_backward
    results: dict[str, float] = {}
    with torch.no_grad():
        for name, call in stages.items():
            if not settings.quiet:
                print(f"benchmark {name}", file=sys.stderr)
            results[name] = _time(call, settings.repeats, device)
    return BenchmarkReport(
        timestamp=datetime.now(UTC),
        settings=settings,
        torch_version=str(torch.__version__),
        seconds=results,
        cost_bytes=costs.value.values.numel() * 8,
        cuda_peak_bytes=torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0,
    )


if __name__ == "__main__":
    CliApp.run(BenchmarkSettings)
