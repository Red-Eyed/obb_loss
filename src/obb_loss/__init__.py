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
    SetCriterionConfig,
    SetMatcherConfig,
    SetWeights,
    VFLConfig,
    parse_config,
)
from obb_loss.criterion import ProbIoULoss
from obb_loss.losses.l1 import box_l1_loss
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
from obb_loss.set_criterion import SetCriterion, SetLosses, TargetCount
from obb_loss.set_data import (
    PackedPredictions,
    PackedTargets,
    TargetSample,
    make_packed_predictions,
    make_packed_targets,
    pack_predictions,
    pack_targets,
)
from obb_loss.set_matching import SetHungarianMatcher, set_pairwise_costs

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
    "PackedPredictions",
    "PackedTargets",
    "ProbIoUConfig",
    "ProbIoULoss",
    "QFLConfig",
    "QualityFocalLoss",
    "Reduction",
    "Result",
    "SetCriterion",
    "SetCriterionConfig",
    "SetHungarianMatcher",
    "SetLosses",
    "SetMatcherConfig",
    "SetWeights",
    "TargetCount",
    "TargetSample",
    "VFLConfig",
    "VarifocalLoss",
    "box_l1_loss",
    "boxes_from_angles",
    "build_loss",
    "build_matcher",
    "checked_probiou_loss",
    "gather_matched",
    "make_packed_boxes",
    "make_packed_predictions",
    "make_packed_targets",
    "pack_boxes",
    "pack_dense_boxes",
    "pack_predictions",
    "pack_targets",
    "pairwise_costs",
    "parse_config",
    "quality_focal_loss",
    "set_pairwise_costs",
    "stable_probiou",
    "stable_probiou_loss",
    "varifocal_loss",
]

__version__ = version("obb-loss")
