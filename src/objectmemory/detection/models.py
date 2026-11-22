"""
Data models for object detection and visual signature extraction.

Safeguards:
1. Keeps detector confidence, identity-match confidence, alignment quality,
   and visibility strictly decoupled.
2. Clearly flags synthetic fixtures with is_synthetic=True.
3. Uses BoundingBox domain models with non-negative, finite coordinate guarantees.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Sequence
import numpy as np

from objectmemory.domain.models import BoundingBox
from objectmemory.detection.exceptions import DetectorConfigurationError


@dataclass
class DetectorConfiguration:
    """
    Standardized runtime configuration for object detector inference.

    Safeguards:
    1. Validates confidence and NMS thresholds in [0.0, 1.0].
    2. Enforces positive integer limits for max_detections.
    3. Normalizes and deduplicates target class prompts.
    """
    confidence_threshold: float = 0.25
    nms_threshold: float = 0.45
    target_classes: Optional[List[str]] = None
    max_detections: Optional[int] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.confidence_threshold) or not (0.0 <= self.confidence_threshold <= 1.0):
            raise DetectorConfigurationError(
                f"confidence_threshold must be finite and within [0.0, 1.0], got {self.confidence_threshold}"
            )
        if not math.isfinite(self.nms_threshold) or not (0.0 <= self.nms_threshold <= 1.0):
            raise DetectorConfigurationError(
                f"nms_threshold must be finite and within [0.0, 1.0], got {self.nms_threshold}"
            )
        if self.target_classes is not None:
            if not isinstance(self.target_classes, (list, tuple, set)):
                raise DetectorConfigurationError("target_classes must be a sequence of strings.")
            cleaned = []
            seen = set()
            for c in self.target_classes:
                s = str(c).strip()
                if s and s.lower() not in seen:
                    cleaned.append(s)
                    seen.add(s.lower())
            self.target_classes = cleaned
        if self.max_detections is not None:
            if not isinstance(self.max_detections, int) or self.max_detections <= 0:
                raise DetectorConfigurationError(
                    f"max_detections must be a positive integer, got {self.max_detections}"
                )



@dataclass
class VisualSignature:
    """
    Compact visual appearance descriptor combining color and texture information.
    Feature vectors are L2-normalized so that inner product equals cosine similarity.
    """
    color_histogram: np.ndarray        # e.g., HSV 3D histogram (float32)
    texture_descriptor: np.ndarray     # e.g., Local Binary Pattern or Sobel gradient histogram (float32)
    combined_vector: np.ndarray        # L2-normalized concatenation (float32)
    is_valid: bool = True              # Explicit flag: False if extracted from degenerate/empty crops
    failure_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if not np.all(np.isfinite(self.combined_vector)):
            raise ValueError("VisualSignature combined_vector contains non-finite values.")
        # Ensure float32 1D array
        if self.combined_vector.ndim != 1:
            self.combined_vector = self.combined_vector.ravel()

    def cosine_similarity(self, other: VisualSignature) -> float:
        """
        Computes cosine similarity in [0.0, 1.0] between two normalized signatures.
        Safeguard: If either signature is invalid (e.g. degenerate crop), similarity is 0.0.
        """
        if not self.is_valid or not other.is_valid:
            return 0.0
        dot = float(np.dot(self.combined_vector, other.combined_vector))
        return max(0.0, min(1.0, dot))


@dataclass
class Detection:
    """
    An individual localized object detected in an image frame.

    Coordinate conventions:
    - raw_bbox: BoundingBox in current image frame (x_min, y_min, x_max, y_max)
    - canonical_bbox: BoundingBox projected into canonical reference frame via homography
    """
    label: str
    confidence: float
    raw_bbox: BoundingBox
    canonical_bbox: Optional[BoundingBox] = None
    signature: Optional[VisualSignature] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.confidence) or not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Detection confidence must be finite in [0.0, 1.0], got {self.confidence}")
        if not self.label or not self.label.strip():
            raise ValueError("Detection label cannot be empty.")


@dataclass
class DetectionBatch:
    """
    Structured outcome of an object detector run on an image frame.
    """
    detections: List[Detection]
    image_size: Tuple[int, int]  # (width, height)
    latency_ms: float
    detector_name: str
    is_synthetic: bool = False   # Clearly labels synthetic/mock fixtures
    failure_reason: Optional[str] = None


@dataclass(frozen=True)
class SignatureMatch:
    """Represents an appearance and spatial comparison between two observations."""
    appearance_similarity: float
    spatial_distance_pixels: float
    raw_score: float
