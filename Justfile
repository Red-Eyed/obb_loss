check: fmt-check lint types test

fmt:
    uv run ruff format src tests

fmt-check:
    uv run ruff format --check src tests

lint:
    uv run ruff check src tests

types:
    uv run python -m tests.typecheck

test:
    uv run --extra matching pytest

stress:
    uv run pytest -m stress

build:
    uv build
