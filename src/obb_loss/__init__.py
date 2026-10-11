"""Typed packed oriented-box losses for PyTorch detectors."""

from importlib.metadata import version

from obb_loss.adapters import boxes_from_angles, pack_boxes, pack_dense_boxes
from obb_loss.builder import BoxLossBuilder, CompositeBoxLoss, build_loss, build_matcher
from obb_loss.config import (
    LossConfig,
    MatcherConfig,
    ProbIoUConfig,
    QFLConfig,
    Reduction,
    VFLConfig,
    parse_config,
)
from obb_loss.criterion import ProbIoULoss
from obb_loss.losses.quality_focal import QualityFocalLoss, quality_focal_loss
from obb_loss.losses.stable_probiou import (
    checked_probiou_loss,
    stable_probiou,
    stable_probiou_loss,
)
from obb_loss.losses.varifocal import VarifocalLoss, varifocal_loss
from obb_loss.matching import Assignment, HungarianMatcher, gather_matched
from obb_loss.packed import InputError, PackedBoxes, make_packed_boxes
from obb_loss.pairwise import PackedCosts, pairwise_costs
from obb_loss.protocols import BoxFunction
from obb_loss.result import Err, Ok, Result

__all__ = [
    "Assignment",
    "BoxFunction",
    "BoxLossBuilder",
    "CompositeBoxLoss",
    "Err",
    "HungarianMatcher",
    "InputError",
    "LossConfig",
    "MatcherConfig",
    "Ok",
    "PackedBoxes",
    "PackedCosts",
    "ProbIoUConfig",
    "ProbIoULoss",
    "QFLConfig",
    "QualityFocalLoss",
    "Reduction",
    "Result",
    "VFLConfig",
    "VarifocalLoss",
    "boxes_from_angles",
    "build_loss",
    "build_matcher",
    "checked_probiou_loss",
    "gather_matched",
    "make_packed_boxes",
    "pack_boxes",
    "pack_dense_boxes",
    "pairwise_costs",
    "parse_config",
    "quality_focal_loss",
    "stable_probiou",
    "stable_probiou_loss",
    "varifocal_loss",
]

__version__ = version("obb-loss")
