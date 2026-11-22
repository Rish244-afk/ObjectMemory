"""
Domain models for ObjectMemory.

Safeguards:
1. Strict dataclasses with validation logic.
2. Coordinates and confidence scores must be finite (rejects NaN, +Inf, -Inf).
3. ISO-8601 timestamps are validated and normalized to UTC.
4. Raw evidence is stored separately from inferred environmental states.
5. Metric confidence scores are optional and never manufactured.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple, Union

from objectmemory.domain.states import (
    BaselineChangeType,
    EpistemicState,
    VisibilityStatus,
)


def normalize_iso_timestamp(ts: Union[str, datetime]) -> str:
    """
    Validates and normalizes a timestamp to a UTC ISO-8601 string (YYYY-MM-DDTHH:MM:SSZ).

    Policy:
    - If a string, parses via datetime.fromisoformat (handling 'Z' suffix).
    - If timezone-aware, converts to UTC.
    - If naive (no tzinfo), deliberately interprets as UTC per system policy.
    - Rejects invalid timestamp strings with ValueError.
    """
    if isinstance(ts, datetime):
        dt = ts
    elif isinstance(ts, str):
        cleaned = ts.strip()
        if not cleaned:
            raise ValueError("Timestamp string cannot be empty")
        # Handle 'Z' suffix for fromisoformat in Python < 3.11 compatibility
        if cleaned.endswith("Z") or cleaned.endswith("z"):
            cleaned = cleaned[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(cleaned)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid ISO-8601 timestamp string: '{ts}'") from exc
    else:
        raise TypeError(f"Timestamp must be a string or datetime, got {type(ts)}")

    if dt.tzinfo is None:
        # Deliberate compatibility policy: naive timestamps are assigned UTC
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)

    # Format cleanly with trailing Z
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _validate_unit_interval(val: Optional[float], field_name: str) -> None:
    if val is not None:
        if not isinstance(val, (int, float)):
            raise TypeError(f"{field_name} must be a float or int, got {type(val)}")
        if not math.isfinite(val):
            raise ValueError(f"{field_name} must be finite (got {val})")
        if not (0.0 <= float(val) <= 1.0):
            raise ValueError(f"{field_name} must be in [0.0, 1.0], got {val}")


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned 2D bounding box (x_min, y_min, x_max, y_max)."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        for name, val in [("x_min", self.x_min), ("y_min", self.y_min), ("x_max", self.x_max), ("y_max", self.y_max)]:
            if not isinstance(val, (int, float)):
                raise TypeError(f"{name} must be a float or int, got {type(val)}")
            if not math.isfinite(val):
                raise ValueError(f"{name} must be finite, got {val}")
        if self.x_min > self.x_max:
            raise ValueError(f"x_min ({self.x_min}) cannot exceed x_max ({self.x_max})")
        if self.y_min > self.y_max:
            raise ValueError(f"y_min ({self.y_min}) cannot exceed y_max ({self.y_max})")
        if self.x_min < 0 or self.y_min < 0:
            raise ValueError(f"Coordinates cannot be negative: ({self.x_min}, {self.y_min})")

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x_min, self.y_min, self.x_max, self.y_max)

    @classmethod
    def from_tuple(cls, t: Tuple[float, float, float, float]) -> BoundingBox:
        if len(t) != 4:
            raise ValueError(f"Bounding box tuple must have exactly 4 elements, got {len(t)}")
        return cls(x_min=float(t[0]), y_min=float(t[1]), x_max=float(t[2]), y_max=float(t[3]))


@dataclass
class Session:
    """Represents a discrete visit or observation session of the workbench."""
    session_id: str
    session_name: str
    timestamp: str
    image_path: str
    notes: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if not self.session_id or not self.session_id.strip():
            raise ValueError("session_id cannot be empty")
        if not self.image_path or not self.image_path.strip():
            raise ValueError("image_path cannot be empty")
        self.timestamp = normalize_iso_timestamp(self.timestamp)


@dataclass
class RawDetection:
    """
    Represents an unadulterated detector output in native camera/frame pixel coordinates.
    Persisted independently of spatial registration success and cross-visit epistemic tracking.
    """
    detection_id: str
    session_id: str
    timestamp: str
    label: str
    confidence: float
    raw_bbox: BoundingBox
    canonical_bbox: Optional[BoundingBox] = None
    registration_status: str = "UNREGISTERED"
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if not self.detection_id or not self.detection_id.strip():
            raise ValueError("detection_id cannot be empty")
        if not self.session_id or not self.session_id.strip():
            raise ValueError("session_id cannot be empty")
        if not self.label or not self.label.strip():
            raise ValueError("label cannot be empty")
        self.timestamp = normalize_iso_timestamp(self.timestamp)
        _validate_unit_interval(self.confidence, "confidence")
        if not isinstance(self.raw_bbox, BoundingBox):
            raise TypeError(f"raw_bbox must be BoundingBox, got {type(self.raw_bbox)}")
        if self.canonical_bbox is not None and not isinstance(self.canonical_bbox, BoundingBox):
            raise TypeError(f"canonical_bbox must be BoundingBox or None, got {type(self.canonical_bbox)}")


@dataclass
class WorkbenchEntity:
    """
    A persistent real-world entity tracked across visits.
    Decouples persistent entity identity from transient per-frame observations.
    """
    entity_id: str
    label: str
    created_at: str
    canonical_bbox: Optional[BoundingBox] = None
    current_state: EpistemicState = EpistemicState.INSUFFICIENT_EVIDENCE
    last_seen_session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if not self.entity_id or not self.entity_id.strip():
            raise ValueError("entity_id cannot be empty")
        if not self.label or not self.label.strip():
            raise ValueError("label cannot be empty")
        self.created_at = normalize_iso_timestamp(self.created_at)
        if not isinstance(self.current_state, EpistemicState):
            self.current_state = EpistemicState(self.current_state)


@dataclass
class Observation:
    """
    A localized observation captured in a specific session.

    Safeguards:
    - Raw evidence is stored in raw_evidence, separate from inferred state.
    - Detector confidence, identity match score, and alignment quality can be None.
    - No scores are fabricated.
    - Scores must be finite numbers in [0.0, 1.0].
    """
    observation_id: str
    session_id: str
    timestamp: str
    image_reference: str
    entity_id: Optional[str] = None
    raw_bbox: Optional[BoundingBox] = None
    canonical_bbox: Optional[BoundingBox] = None
    visibility_status: VisibilityStatus = VisibilityStatus.UNCERTAIN
    detector_confidence: Optional[float] = None
    identity_match_score: Optional[float] = None
    alignment_quality: Optional[float] = None
    appearance_descriptor_ref: Optional[str] = None
    raw_evidence: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if not self.observation_id or not self.observation_id.strip():
            raise ValueError("observation_id cannot be empty")
        if not self.session_id or not self.session_id.strip():
            raise ValueError("session_id cannot be empty")
        self.timestamp = normalize_iso_timestamp(self.timestamp)
        if not isinstance(self.visibility_status, VisibilityStatus):
            self.visibility_status = VisibilityStatus(self.visibility_status)
        _validate_unit_interval(self.detector_confidence, "detector_confidence")
        _validate_unit_interval(self.identity_match_score, "identity_match_score")
        _validate_unit_interval(self.alignment_quality, "alignment_quality")


@dataclass
class ChangeEvent:
    """
    Records an explicit transition of an entity's persistent epistemic state.
    Provides complete auditability of decisions and their supporting evidence.
    """
    event_id: str
    entity_id: str
    to_session_id: str
    timestamp: str
    prior_state: EpistemicState
    new_state: EpistemicState
    from_session_id: Optional[str] = None
    source: str = "TEMPORAL_MEMORY"
    displacement_distance: Optional[float] = None
    supporting_evidence: Dict[str, Any] = field(default_factory=dict)
    explanation: str = ""

    def __post_init__(self) -> None:
        if not self.event_id or not self.event_id.strip():
            raise ValueError("event_id cannot be empty")
        if not self.entity_id or not self.entity_id.strip():
            raise ValueError("entity_id cannot be empty")
        if not self.to_session_id or not self.to_session_id.strip():
            raise ValueError("to_session_id cannot be empty")
        self.timestamp = normalize_iso_timestamp(self.timestamp)
        if not isinstance(self.prior_state, EpistemicState):
            self.prior_state = EpistemicState(self.prior_state)
        if not isinstance(self.new_state, EpistemicState):
            self.new_state = EpistemicState(self.new_state)


@dataclass
class PairwiseBaselineResult:
    """
    Stores an independent two-frame comparison result.
    Maintained strictly separate from multi-session temporal memory.
    """
    result_id: str
    reference_session_id: str
    query_session_id: str
    predicted_change: BaselineChangeType
    entity_id: Optional[str] = None
    confidence: Optional[float] = None
    details: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if not self.result_id or not self.result_id.strip():
            raise ValueError("result_id cannot be empty")
        if not isinstance(self.predicted_change, BaselineChangeType):
            self.predicted_change = BaselineChangeType(self.predicted_change)
        _validate_unit_interval(self.confidence, "confidence")
