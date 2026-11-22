"""
Baseline change-detection modules for ObjectMemory.

Provides the independent pairwise baseline used as an experimental control
against the multi-visit temporal memory engine.
"""

from objectmemory.baseline.models import (
    PairwiseMatchConfig,
    PairwiseMatch,
    PairwiseComparisonReport,
)
from objectmemory.baseline.comparator import (
    PairwiseComparator,
    calculate_bbox_iou,
    calculate_centroid_distance,
    check_bbox_overlap_fraction,
)

__all__ = [
    "PairwiseMatchConfig",
    "PairwiseMatch",
    "PairwiseComparisonReport",
    "PairwiseComparator",
    "calculate_bbox_iou",
    "calculate_centroid_distance",
    "check_bbox_overlap_fraction",
]
