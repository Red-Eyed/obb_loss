"""Bounded simulation evidence and distribution reproducibility."""

import pytest

from tests.simulation import InputDtype, StressSettings, simulate


@pytest.mark.stress
@pytest.mark.parametrize("dtype", [InputDtype.FP32, InputDtype.FP64])
def test_bounded_reproducible_simulation(dtype: InputDtype) -> None:
    """Moderate representable geometry remains finite without unbounded arrays."""
    settings = StressSettings(
        samples=257,
        chunk_size=31,
        dtype=dtype,
        min_log_scale=-5,
        max_log_scale=5,
        max_log_aspect=6,
        quiet=True,
    )
    first, second = simulate(settings), simulate(settings)
    assert first.evaluated == 257
    assert first.max_chunk_seen == 31
    assert first.forward_failures == 0
    assert first.backward_failures == 0
    assert first.first_failure_sample == second.first_failure_sample
    assert first.analytic_max_error == second.analytic_max_error
    assert first.analytic_max_error < 1e-13


@pytest.mark.stress
def test_low_precision_failures_are_classified() -> None:
    """Input quantization and gradient cast overflow are reported separately."""
    report = simulate(
        StressSettings(samples=256, chunk_size=32, dtype=InputDtype.FP16, quiet=True)
    )
    assert report.rejected_input_quantization > 0
    assert report.evaluated + report.rejected_input_quantization == 256
    assert report.forward_failures == 0
    assert report.unexplained_backward_failures == 0
