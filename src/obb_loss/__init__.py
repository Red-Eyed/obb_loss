"""Numerically stable oriented-box and quality-classification loss functions."""

from importlib.metadata import version

from obb_loss.losses.l1 import box_l1_loss
from obb_loss.losses.quality_focal import quality_focal_loss
from obb_loss.losses.stable_probiou import stable_probiou, stable_probiou_loss
from obb_loss.losses.varifocal import varifocal_loss
from obb_loss.result import Err, ErrorKind, InputError, Ok, Result

__all__ = [
    "Err",
    "ErrorKind",
    "InputError",
    "Ok",
    "Result",
    "box_l1_loss",
    "quality_focal_loss",
    "stable_probiou",
    "stable_probiou_loss",
    "varifocal_loss",
]

__version__ = version("obb-loss")
