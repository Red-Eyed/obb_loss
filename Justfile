check: fmt-check lint types test examples

fmt:
    uv run ruff format src tools examples

fmt-check:
    uv run ruff format --check src tools examples

lint:
    uv run ruff check src tools examples

types:
    uv run python -m tools.typecheck

test:
    uv run pytest

examples:
    uv run examples/matching.py
    uv run examples/classification.py
    uv run examples/detr.py
    uv run examples/custom_set_loss.py

stress *args:
    uv run python -m tools.stress {{args}}

benchmark *args:
    uv run python -m tools.benchmark {{args}}

test-cuda:
    uv run python -c 'import torch; assert torch.cuda.is_available(), "CUDA hardware required"'
    uv run pytest -m cuda

build:
    uv build
