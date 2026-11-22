"""
Data contracts and configuration models for the Phase 4 Pairwise Change-Detection Baseline.

Safeguards:
- Match confidence is decoupled from detector confidence and alignment quality.
- Outcomes are restricted to BaselineChangeType (UNCHANGED, MOVED, ADDED, REMOVED, UNKNOWN).
- Numerical scores reject NaN and Infinity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from objectmemory.detection.models import Detection
from objectmemory.domain.states import BaselineChangeType
from objectmemory.registration.models import AlignmentStatus


def _validate_unit_interval(val: Optional[float], name: str) -> None:
    if val is not None:
        if not math.isfinite(val):
            raise ValueError(f"{name} must be a finite number, got {val}")
        if val < 0.0 or val > 1.0:
            raise ValueError(f"{name} must be in [0.0, 1.0], got {val}")


@dataclass
class PairwiseMatchConfig:
    """
    Configurable decision parameters for pairwise baseline comparison.
    All thresholds are documented and explicit.
    """
    # Spatial IoU threshold above which matched objects are considered UNCHANGED
    iou_unchanged_threshold: float = 0.40

    # Maximum centroid Euclidean pixel distance for UNCHANGED when IoU is low (e.g. small objects)
    distance_unchanged_threshold: float = 35.0

    # Minimum visual signature cosine similarity required to consider two detections an identity match
    signature_similarity_threshold: float = 0.65

    # Ambiguity margin: if the difference between top-1 and top-2 candidate similarity is less than
    # this margin, the assignment is declared ambiguous (UNKNOWN)
    ambiguity_margin: float = 0.12

    # Minimum detector confidence to consider a candidate detection
    min_detector_confidence: float = 0.35

    # Minimum inlier ratio required from registration before trusting canonical coordinates
    min_inlier_ratio: float = 0.20

    # Minimum overlap area fraction required to evaluate an object's canonical region
    min_overlap_fraction: float = 0.70

    # Vacancy check: if True, evaluates the query image patch at the reference canonical bbox
    # before confirming REMOVED
    vacancy_check_enabled: bool = True

    # Maximum similarity of query patch to reference object for vacancy to be confirmed
    # (similarity < threshold implies the object is genuinely no longer present)
    vacancy_similarity_max: float = 0.35

    # If True, missing detections without vacancy evidence are declared UNKNOWN.
    # Set to False only to emulate a naive detector that blindly equates missing detection to removal.
    require_vacancy_evidence: bool = True

    # Bipartite matching algorithm: "greedy" (default) or "optimal" (Kuhn-Munkres maximum weight)
    assignment_strategy: str = "greedy"

    def __post_init__(self) -> None:
        _validate_unit_interval(self.iou_unchanged_threshold, "iou_unchanged_threshold")
        _validate_unit_interval(self.signature_similarity_threshold, "signature_similarity_threshold")
        _validate_unit_interval(self.ambiguity_margin, "ambiguity_margin")
        _validate_unit_interval(self.min_detector_confidence, "min_detector_confidence")
        _validate_unit_interval(self.min_inlier_ratio, "min_inlier_ratio")
        _validate_unit_interval(self.min_overlap_fraction, "min_overlap_fraction")
        _validate_unit_interval(self.vacancy_similarity_max, "vacancy_similarity_max")
        if self.distance_unchanged_threshold < 0.0 or not math.isfinite(self.distance_unchanged_threshold):
            raise ValueError("distance_unchanged_threshold must be non-negative and finite")
        if self.assignment_strategy not in ("greedy", "optimal"):
            raise ValueError(f"assignment_strategy must be 'greedy' or 'optimal', got '{self.assignment_strategy}'")


@dataclass
class PairwiseMatch:
    """
    Represents the comparison outcome between a reference detection and a query detection
    (or an unmatched detection in either visit).
    """
    predicted_change: BaselineChangeType
    ref_detection: Optional[Detection] = None
    query_detection: Optional[Detection] = None
    match_confidence: Optional[float] = None
    spatial_distance: Optional[float] = None
    spatial_iou: Optional[float] = None
    signature_similarity: Optional[float] = None
    explanation: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.predicted_change, BaselineChangeType):
            self.predicted_change = BaselineChangeType(self.predicted_change)
        _validate_unit_interval(self.match_confidence, "match_confidence")
        if self.spatial_distance is not None and (
            self.spatial_distance < 0.0 or not math.isfinite(self.spatial_distance)
        ):
            raise ValueError(f"spatial_distance must be non-negative and finite, got {self.spatial_distance}")
        if self.spatial_iou is not None:
            _validate_unit_interval(self.spatial_iou, "spatial_iou")
        if self.signature_similarity is not None and not math.isfinite(self.signature_similarity):
            raise ValueError("signature_similarity must be finite")


@dataclass
class PairwiseComparisonReport:
    """
    Summary report of an independent pairwise baseline comparison between two sessions.
    """
    reference_session_id: str
    query_session_id: str
    alignment_status: AlignmentStatus
    matches: List[PairwiseMatch]
    summary_counts: Dict[str, int]
    is_synthetic: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.reference_session_id.strip():
            raise ValueError("reference_session_id cannot be empty")
        if not self.query_session_id.strip():
            raise ValueError("query_session_id cannot be empty")
