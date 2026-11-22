"""
Data contracts and configuration models for Phase 5: Multi-Visit Temporal Memory Engine.

Safeguards:
- Decouples single-visit observations from persistent entity state and change events.
- All confidence scores and probabilities are finite and bounded in [0.0, 1.0].
- Preserves explicit auditability of temporal decisions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

from objectmemory.detection.models import Detection
from objectmemory.domain.models import BoundingBox, WorkbenchEntity
from objectmemory.domain.states import EpistemicState, VisibilityStatus
from objectmemory.registration.models import RegistrationResult


def _validate_unit_interval(val: Optional[float], name: str) -> None:
    if val is not None:
        if not math.isfinite(val):
            raise ValueError(f"{name} must be a finite number, got {val}")
        if val < 0.0 or val > 1.0:
            raise ValueError(f"{name} must be in [0.0, 1.0], got {val}")


@dataclass
class TemporalEngineConfig:
    """
    Configurable hyperparameters and evidence policies for temporal belief updating.
    All thresholds are explicitly defined and documented.
    """
    # 1. Identity association thresholds
    signature_similarity_threshold: float = 0.65
    ambiguity_margin: float = 0.12
    iou_match_threshold: float = 0.40
    distance_match_threshold: float = 35.0  # max pixels for collocated match when IoU is low
    assignment_strategy: str = "optimal"    # "optimal" (Kuhn-Munkres) or "greedy"

    # 2. Minimum registration quality to trust spatial updates
    min_registration_inlier_ratio: float = 0.20
    min_overlap_fraction: float = 0.70
    max_reprojection_error: float = 4.0
    border_margin_px: float = 12.0

    # 3. Vacancy verification thresholds
    vacancy_similarity_max: float = 0.40  # patch similarity < max indicates genuinely vacant patch

    # 4. Multi-visit belief progression thresholds
    # Number of consecutive reliable vacancy observations required to confirm removal
    # Visit 1 unobserved -> PENDING_REMOVAL; Visit 2 verified vacancy -> CONFIRMED_REMOVED
    vacancy_visits_for_pending: int = 1
    vacancy_visits_for_removal: int = 2
    # Number of consecutive unobserved visits under reliable registration when vacancy is unverified
    unobserved_visits_for_removal: int = 3

    # Movement displacement threshold in pixels (canonical coordinates)
    movement_displacement_threshold: float = 35.0

    # 5. Boundary clipping and drift policies
    # Minimum displacement to consider genuine movement for edge-clipped objects
    # Displacements below this for boundary-truncated boxes are treated as boundary truncation artifacts
    clipped_movement_displacement_threshold: float = 100.0

    # Maximum displacement step in pixels between consecutive reliable visits to qualify as coordinate drift
    max_drift_step_displacement: float = 20.0

    # Minimum IoU overlap with initial position below which accumulated displacement cannot be considered drift
    drift_iou_threshold: float = 0.35

    # Background / surface classes that should not be treated as movable foreground objects
    excluded_background_classes: List[str] = field(default_factory=lambda: ["dining table"])

    # 6. Conservative cross-class association thresholds (for temporary detector classification fluctuation)
    enable_cross_class_association: bool = True
    cross_class_iou_threshold: float = 0.35
    cross_class_signature_similarity_threshold: float = 0.70
    cross_class_fallback_iou_threshold: float = 0.75
    cross_class_ambiguity_margin: float = 0.15

    def __post_init__(self) -> None:
        _validate_unit_interval(self.signature_similarity_threshold, "signature_similarity_threshold")
        _validate_unit_interval(self.ambiguity_margin, "ambiguity_margin")
        _validate_unit_interval(self.iou_match_threshold, "iou_match_threshold")
        _validate_unit_interval(self.min_registration_inlier_ratio, "min_registration_inlier_ratio")
        _validate_unit_interval(self.min_overlap_fraction, "min_overlap_fraction")
        _validate_unit_interval(self.vacancy_similarity_max, "vacancy_similarity_max")
        _validate_unit_interval(self.drift_iou_threshold, "drift_iou_threshold")
        _validate_unit_interval(self.cross_class_iou_threshold, "cross_class_iou_threshold")
        _validate_unit_interval(self.cross_class_signature_similarity_threshold, "cross_class_signature_similarity_threshold")
        _validate_unit_interval(self.cross_class_fallback_iou_threshold, "cross_class_fallback_iou_threshold")
        _validate_unit_interval(self.cross_class_ambiguity_margin, "cross_class_ambiguity_margin")
        if self.distance_match_threshold < 0.0 or not math.isfinite(self.distance_match_threshold):
            raise ValueError("distance_match_threshold must be non-negative and finite")
        if self.movement_displacement_threshold < 0.0 or not math.isfinite(self.movement_displacement_threshold):
            raise ValueError("movement_displacement_threshold must be non-negative and finite")
        if self.clipped_movement_displacement_threshold < 0.0 or not math.isfinite(self.clipped_movement_displacement_threshold):
            raise ValueError("clipped_movement_displacement_threshold must be non-negative and finite")
        if self.max_drift_step_displacement < 0.0 or not math.isfinite(self.max_drift_step_displacement):
            raise ValueError("max_drift_step_displacement must be non-negative and finite")
        if self.max_reprojection_error < 0.0 or not math.isfinite(self.max_reprojection_error):
            raise ValueError("max_reprojection_error must be non-negative and finite")
        if self.border_margin_px < 0.0 or not math.isfinite(self.border_margin_px):
            raise ValueError("border_margin_px must be non-negative and finite")
        if self.vacancy_visits_for_pending < 1:
            raise ValueError("vacancy_visits_for_pending must be at least 1")
        if self.vacancy_visits_for_removal < self.vacancy_visits_for_pending:
            raise ValueError("vacancy_visits_for_removal cannot be less than vacancy_visits_for_pending")
        if self.assignment_strategy not in ("optimal", "greedy"):
            raise ValueError(f"assignment_strategy must be 'optimal' or 'greedy', got '{self.assignment_strategy}'")


@dataclass
class VisitInput:
    """
    Public data contract representing one visit to the workbench.
    """
    session_id: str
    session_name: str
    timestamp: str
    image: np.ndarray
    registration: RegistrationResult
    detections: List[Detection]
    notes: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.session_id or not self.session_id.strip():
            raise ValueError("session_id cannot be empty")
        if not self.session_name or not self.session_name.strip():
            raise ValueError("session_name cannot be empty")
        if not self.timestamp or not self.timestamp.strip():
            raise ValueError("timestamp cannot be empty")


@dataclass
class AssociationMatch:
    """
    Structured outcome of associating a detected object with an existing entity.
    """
    entity_id: Optional[str]
    detection: Optional[Detection]
    is_associated: bool
    is_ambiguous: bool = False
    is_cross_class: bool = False
    confidence: float = 0.0
    signature_similarity: Optional[float] = None
    spatial_iou: Optional[float] = None
    spatial_distance: Optional[float] = None
    explanation: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_unit_interval(self.confidence, "confidence")


@dataclass
class EntityBeliefUpdate:
    """
    Summary of an entity's belief update during a single visit.
    """
    entity_id: str
    label: str
    prior_state: EpistemicState
    new_state: EpistemicState
    transition_occurred: bool
    observation_id: Optional[str] = None
    event_id: Optional[str] = None
    visibility_status: VisibilityStatus = VisibilityStatus.UNCERTAIN
    explanation: str = ""
    consecutive_vacant_visits: int = 0
    supporting_evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VisitProcessingReport:
    """
    Complete audit report produced after processing a single visit.
    """
    session_id: str
    session_name: str
    timestamp: str
    registration_status: str
    total_detections: int
    matched_detections: int
    new_entities_created: int
    state_transitions_count: int
    entity_updates: List[EntityBeliefUpdate] = field(default_factory=list)
    unassociated_detections: List[Detection] = field(default_factory=list)
    is_reprocessed: bool = False
    details: Dict[str, Any] = field(default_factory=dict)
