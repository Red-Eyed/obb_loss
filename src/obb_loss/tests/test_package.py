"""Verify installed package metadata."""

from importlib.metadata import version

import obb_loss
import obb_loss as obb
from obb_loss.adapters import pack_boxes
from obb_loss.result import Err, Ok


def test_version_matches_metadata() -> None:
    """Expose the installed version without a duplicate version declaration."""
    assert obb_loss.__version__ == version("obb-loss")


def test_module_alias() -> None:
    """Expose adapters and result variants through the advertised module alias."""
    assert obb.pack_boxes is pack_boxes
    assert obb.Ok is Ok
    assert obb.Err is Err
