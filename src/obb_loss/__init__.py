"""Typed packed oriented-box losses for PyTorch detectors."""

from importlib.metadata import version

from obb_loss.adapters import boxes_from_angles, pack_boxes, pack_dense_boxes
from obb_loss.packed import InputError, PackedBoxes, make_packed_boxes
from obb_loss.result import Err, Ok, Result

__all__ = [
    "Err",
    "InputError",
    "Ok",
    "PackedBoxes",
    "Result",
    "boxes_from_angles",
    "make_packed_boxes",
    "pack_boxes",
    "pack_dense_boxes",
]

__version__ = version("obb-loss")
