"""Numerically stable oriented-box and quality-classification loss functions."""

from importlib.metadata import version

from obb_loss.criterion import criterion_loss
from obb_loss.criterion_types import (
    Cost,
    Criterion,
    CriterionError,
    CriterionErrorKind,
    CriterionFailure,
    ImagePredictions,
    Loss,
    LossContext,
    LossOutput,
    Match,
    Matcher,
    Predictions,
    Targets,
)
from obb_loss.losses.l1 import box_l1_loss
from obb_loss.losses.quality_focal import quality_focal_loss
from obb_loss.losses.stable_probiou import stable_probiou, stable_probiou_loss
from obb_loss.losses.varifocal import varifocal_loss
from obb_loss.matching import hungarian_assignment
from obb_loss.result import Err, ErrorKind, InputError, Ok, Result

__all__ = [
    "Cost",
    "Criterion",
    "CriterionError",
    "CriterionErrorKind",
    "CriterionFailure",
    "Err",
    "ErrorKind",
    "ImagePredictions",
    "InputError",
    "Loss",
    "LossContext",
    "LossOutput",
    "Match",
    "Matcher",
    "Ok",
    "Predictions",
    "Result",
    "Targets",
    "box_l1_loss",
    "criterion_loss",
    "hungarian_assignment",
    "quality_focal_loss",
    "stable_probiou",
    "stable_probiou_loss",
    "varifocal_loss",
]

__version__ = version("obb-loss")
