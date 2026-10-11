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
