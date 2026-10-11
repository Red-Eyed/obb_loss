"""Verify installed package metadata."""

from importlib.metadata import version

import obb_loss


def test_version_matches_metadata() -> None:
    """Expose the installed version without a duplicate version declaration."""
    assert obb_loss.__version__ == version("obb-loss")
