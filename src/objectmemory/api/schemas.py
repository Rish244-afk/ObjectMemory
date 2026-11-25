"""
Pydantic API schemas and data contracts for ObjectMemory backend.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Workspace Schemas
# ---------------------------------------------------------------------------

class WorkspaceCreate(BaseModel):
    workspace_id: str = Field(..., min_length=1, max_length=64, description="Unique slug for workspace (e.g. 'lab-bench-01')")
    name: str = Field(..., min_length=1, max_length=128, description="Human-readable name")
    description: Optional[str] = Field(None, description="Optional description of the environment")


class WorkspaceSummary(BaseModel):
    workspace_id: str
    name: str
    description: Optional[str] = None
    created_at: str
    total_sessions: int = 0
    total_entities: int = 0
    total_events: int = 0
    reference_session_id: Optional[str] = None
    last_session_timestamp: Optional[str] = None


class WorkspaceResponse(BaseModel):
    workspace: WorkspaceSummary
    entity_state_counts: Dict[str, int] = Field(default_factory=dict)


class WorkspaceExportResponse(BaseModel):
    workspace: WorkspaceSummary
    entity_state_counts: Dict[str, int] = Field(default_factory=dict)
    sessions: List[SessionDetailResponse] = Field(default_factory=list)
    entities: List[EntityDetailResponse] = Field(default_factory=list)
    events: List[ChangeEventSummary] = Field(default_factory=list)
    diagnostics: Optional[DiagnosticsResponse] = None
    exported_at: str = Field(..., description="ISO 8601 timestamp of export generation")


# ---------------------------------------------------------------------------
# Spatial / Detection Schemas
# ---------------------------------------------------------------------------

class BoundingBoxSchema(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class DetectionSummary(BaseModel):
    label: str
    confidence: float
    raw_bbox: BoundingBoxSchema
    canonical_bbox: Optional[BoundingBoxSchema] = None
    is_clipped: bool = False


class RegistrationSummary(BaseModel):
    success: bool
    status: str
    method_used: str
    inlier_ratio: float = 0.0
    mean_reprojection_error: Optional[float] = None
    inliers_count: int = 0
    ref_size: List[int]
    query_size: List[int]
    failure_reason: Optional[str] = None
    selected_keyframe_id: Optional[str] = None
    is_direct_to_canonical: bool = True
    spatial_coverage: Optional[float] = None
    decision_status: Optional[str] = None
    decision_reason: Optional[str] = None
    is_promoted: bool = False


# ---------------------------------------------------------------------------
# Temporal / Observation Schemas
# ---------------------------------------------------------------------------

class EntityBeliefUpdateSummary(BaseModel):
    entity_id: str
    label: str
    prior_state: str
    new_state: str
    transition_occurred: bool
    visibility_status: str
    explanation: str
    event_id: Optional[str] = None
    displacement_px: Optional[float] = None


class ChangeEventSummary(BaseModel):
    event_id: str
    entity_id: str
    from_session_id: Optional[str] = None
    to_session_id: str
    timestamp: str
    prior_state: str
    new_state: str
    source: str
    displacement_distance: Optional[float] = None
    explanation: str
    supporting_evidence: Dict[str, Any] = Field(default_factory=dict)


class ObservationSubmissionResult(BaseModel):
    session_id: str
    session_name: str
    timestamp: str
    image_url: str
    is_reference_frame: bool
    registration: RegistrationSummary
    detections: List[DetectionSummary]
    belief_updates: List[EntityBeliefUpdateSummary]
    events: List[ChangeEventSummary]
    total_active_entities: int
    latency_breakdown_ms: Dict[str, float]
    total_latency_ms: float


class ObservationItem(BaseModel):
    observation_id: str
    session_id: str
    entity_id: Optional[str] = None
    label: Optional[str] = None
    timestamp: str
    image_reference: str
    raw_bbox: Optional[BoundingBoxSchema] = None
    canonical_bbox: Optional[BoundingBoxSchema] = None
    visibility_status: str
    detector_confidence: Optional[float] = None
    identity_match_score: Optional[float] = None
    alignment_quality: Optional[float] = None
    raw_evidence: Optional[Dict[str, Any]] = None


class RawDetectionItem(BaseModel):
    detection_id: str
    session_id: str
    timestamp: str
    label: str
    confidence: float
    raw_bbox: BoundingBoxSchema
    canonical_bbox: Optional[BoundingBoxSchema] = None
    registration_status: str = "UNREGISTERED"
    metadata: Optional[Dict[str, Any]] = None


class SessionSummary(BaseModel):
    session_id: str
    session_name: str
    timestamp: str
    image_path: str
    image_url: str
    notes: Optional[str] = None
    observations_count: int = 0
    events_count: int = 0
    raw_detections_count: int = 0


class SessionDetailResponse(BaseModel):
    session: SessionSummary
    observations: List[ObservationItem]
    events: List[ChangeEventSummary]
    raw_detections: List[RawDetectionItem] = Field(default_factory=list)
    registration: Optional[RegistrationSummary] = None


# ---------------------------------------------------------------------------
# Entity Schemas
# ---------------------------------------------------------------------------

class EntitySummary(BaseModel):
    entity_id: str
    label: str
    raw_label: Optional[str] = None
    raw_confidence: Optional[float] = None
    confirmed_label: Optional[str] = None
    label_provenance: str = "MODEL_PREDICTED"
    label_confirmed_at: Optional[str] = None
    observed_labels: List[str] = Field(default_factory=list)
    class_uncertain: bool = False
    created_at: str
    current_state: str
    canonical_bbox: Optional[BoundingBoxSchema] = None
    last_seen_session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class EntityDetailResponse(BaseModel):
    entity: EntitySummary
    observations: List[ObservationItem]
    change_events: List[ChangeEventSummary]


# ---------------------------------------------------------------------------
# Diagnostics & Comparison Schemas
# ---------------------------------------------------------------------------

class DiagnosticsResponse(BaseModel):
    workspace_id: str
    detector_info: Dict[str, Any]
    total_sessions: int
    total_entities: int
    total_events: int
    entity_state_distribution: Dict[str, int]
    registration_health: Dict[str, Any]


class DetectorInfoSchema(BaseModel):
    name: str
    is_default: bool
    is_available: bool
    availability_error: Optional[str] = None
    is_open_vocabulary: bool = False
    is_synthetic: bool = False
    description: str = ""
    model_info: Dict[str, Any] = Field(default_factory=dict)


class DetectorListResponse(BaseModel):
    default_detector: str
    detectors: List[DetectorInfoSchema]


class EntityLabelUpdate(BaseModel):
    label: str = Field(..., min_length=1, max_length=64, description="User-confirmed semantic label for the entity")


class ConfirmLocationRequest(BaseModel):
    entity_id: str = Field(..., min_length=1, description="Persistent entity ID")
    bbox: BoundingBoxSchema = Field(..., description="User-provided bounding box in raw image pixel coordinates")
    notes: Optional[str] = Field(None, description="Optional operator note")


class ConfirmLocationResponse(BaseModel):
    workspace_id: str
    session_id: str
    entity_id: str
    evidence_source: str = "HUMAN_CONFIRMED"
    registration_reliable: bool
    prior_state: str
    new_state: str
    displacement_px: Optional[float] = None
    canonical_bbox: Optional[BoundingBoxSchema] = None
    raw_bbox: BoundingBoxSchema
    explanation: str


class ComparisonDiffItem(BaseModel):
    entity_id: str
    label: str
    raw_label: Optional[str] = None
    confirmed_label: Optional[str] = None
    label_provenance: str = "MODEL_PREDICTED"
    evidence_source: str = "MODEL_DETECTION"
    observed_labels: List[str] = Field(default_factory=list)
    class_uncertain: bool = False
    status_change: str
    movement_status: str = "NOT_MOVED"
    prior_state: str
    new_state: str
    displacement_px: Optional[float] = None
    ref_confidence: Optional[float] = None
    query_confidence: Optional[float] = None
    ref_bbox: Optional[BoundingBoxSchema] = None
    query_bbox: Optional[BoundingBoxSchema] = None
    ref_raw_bbox: Optional[BoundingBoxSchema] = None
    query_raw_bbox: Optional[BoundingBoxSchema] = None
    explanation: str


class ComparisonResponse(BaseModel):
    workspace_id: str
    reference_session_id: str
    query_session_id: str
    registration_status: str
    registration_inlier_ratio: float
    ref_detections_count: int = 0
    query_detections_count: int = 0
    cross_sequence_warning: Optional[str] = None
    items: List[ComparisonDiffItem]
