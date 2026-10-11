"""Shared explicit CPU/CUDA numerical test resources."""

import pytest
import torch
import torch.cuda


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
def device(request: pytest.FixtureRequest) -> torch.device:
    """Exercise CUDA when present and report its absence as a visible skip."""
    name: str = request.param
    if name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware unavailable")
    return torch.device(name)
