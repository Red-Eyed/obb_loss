check: fmt-check lint types test

fmt:
    uv run ruff format src tools

fmt-check:
    uv run ruff format --check src tools

lint:
    uv run ruff check src tools

types:
    uv run pyrefly check

test:
    uv run pytest

stress *args:
    uv run python -m tools.stress {{args}}

benchmark *args:
    uv run python -m tools.benchmark {{args}}

test-cuda:
    uv run python -c 'import torch; assert torch.cuda.is_available(), "CUDA hardware required"'
    uv run pytest -m cuda

build:
    uv build
