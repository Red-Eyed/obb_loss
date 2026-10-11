"""Typed packed oriented-box losses for PyTorch detectors."""

from importlib.metadata import version

from obb_loss.adapters import boxes_from_angles, pack_boxes, pack_dense_boxes
from obb_loss.config import ProbIoUConfig, QFLConfig, Reduction, VFLConfig, parse_config
from obb_loss.criterion import ProbIoULoss
from obb_loss.losses.stable_probiou import (
    checked_probiou_loss,
    stable_probiou,
    stable_probiou_loss,
)
from obb_loss.packed import InputError, PackedBoxes, make_packed_boxes
from obb_loss.result import Err, Ok, Result

__all__ = [
    "Err",
    "InputError",
    "Ok",
    "PackedBoxes",
    "ProbIoUConfig",
    "ProbIoULoss",
    "QFLConfig",
    "Reduction",
    "Result",
    "VFLConfig",
    "boxes_from_angles",
    "checked_probiou_loss",
    "make_packed_boxes",
    "pack_boxes",
    "pack_dense_boxes",
    "parse_config",
    "stable_probiou",
    "stable_probiou_loss",
]

__version__ = version("obb-loss")
