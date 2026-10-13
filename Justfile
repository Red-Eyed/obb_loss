check: fmt-check lint types test example-detr

fmt:
    uv run ruff format src tests examples

fmt-check:
    uv run ruff format --check src tests examples

lint:
    uv run ruff check src tests examples

types:
    uv run python -m tests.typecheck

test:
    uv run --extra matching pytest

stress:
    uv run pytest -m stress

build:
    uv build

example-detr:
    uv run --extra matching examples/detr_training.py
