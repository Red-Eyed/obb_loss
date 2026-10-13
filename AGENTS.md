# obb_loss

This repository contains pure PyTorch losses in `src/obb_loss/losses/` and
focused tests in `tests/`. The declarative criterion owns matching: keep tensor
computation in `criterion_core.py` and assignment effects in `criterion.py` and
`matching.py`. Extend it with immutable cost/loss declarations and ordinary
functions. Keep it small: no detector harness, packed records, configuration
framework, builders, registry, or callback machinery.

Public losses return the small `Ok`/`Err` union defined in `result.py`. Each loss
module owns a metadata validation helper for its shape, floating dtype and
shared CPU/CUDA device contract. Do not scan tensor values or add checked APIs.
Numerical domain requirements belong in function docstrings. Criterion validation
also checks integer label/assignment metadata; scalar declarations have typed
criterion errors. Optional SciPy stays behind the lazy assignment adapter.

Geometry is `(cx, cy, width, height, cos(theta), sin(theta))`. Inputs are borrowed
and never mutated. Computation disables autocast locally and uses float64.
Keep numerical reasoning next to the implementation and cite primary papers.
Use einops for rearrangement and native Pyrefly shapes under TYPE_CHECKING so
Python 3.11 runtime imports remain supported. Development defaults to 3.12.

Use uv for Python commands. Run `just check`: separate strict configurations
live in `ruff.toml`, `pyrefly.toml`, and `pytest.ini`. Native shape checking runs
through `uv run python -m tests.typecheck`. Keep independent mathematical oracles,
metadata contract tests, gradient tests, and seeded stability simulations.
CUDA tests skip when unavailable; CPU tests still run.

Follow the user's global instructions. Before committing, run Ruff check,
Ruff format, reread formatted files, Pyrefly, and relevant pytest tests.
On main, synchronize package version and lockfile for user-visible changes.
Only the user publishes packages. The ignored `plan/` is working material.
