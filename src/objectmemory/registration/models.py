"""
Data models and configuration for planar image co-registration.

Safeguards:
1. Coordinate convention is explicit:
   - Image dimensions are (width, height) = (cols, rows).
   - Points are (x, y) coordinates where x is horizontal column, y is vertical row.
   - Homography maps query points to reference points: p_ref ~ H @ p_query.
2. Thresholds are explicitly documented as engineering heuristics, not claimed as
   scientifically validated constants.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple
import numpy as np


class AlignmentStatus(str, Enum):
    """Discrete outcome states for image registration and quality evaluation."""
    SUCCESS = "SUCCESS"
    INSUFFICIENT_KEYPOINTS = "INSUFFICIENT_KEYPOINTS"
    INSUFFICIENT_MATCHES = "INSUFFICIENT_MATCHES"
    HOMOGRAPHY_ESTIMATION_FAILED = "HOMOGRAPHY_ESTIMATION_FAILED"
    NON_FINITE_HOMOGRAPHY = "NON_FINITE_HOMOGRAPHY"
    SINGULAR_HOMOGRAPHY = "SINGULAR_HOMOGRAPHY"
    IMPLAUSIBLE_WARP_GEOMETRY = "IMPLAUSIBLE_WARP_GEOMETRY"
    EXCESSIVE_REPROJECTION_ERROR = "EXCESSIVE_REPROJECTION_ERROR"
    INSUFFICIENT_INLIERS = "INSUFFICIENT_INLIERS"
    INSUFFICIENT_OVERLAP = "INSUFFICIENT_OVERLAP"
    EXECUTION_ERROR = "EXECUTION_ERROR"


@dataclass(frozen=True)
class ReprojectionStats:
    """Reprojection error statistics computed over RANSAC inlier keypoint pairs."""
    mean: float
    median: float
    std: float
    max_error: float
    inlier_count: int


@dataclass(frozen=True)
class AlignmentConfig:
    """
    Configurable parameters for feature matching, homography estimation,
    and quality gating.

    NOTE: The default thresholds below are engineering heuristics chosen for typical
    planar workbench scenarios. They are NOT claimed to be scientifically validated constants.
    """
    feature_method: str = "SIFT"  # 'SIFT' or 'ORB'
    max_features: int = 2000
    ratio_test_threshold: float = 0.75  # Lowe's ratio test threshold
    ransac_reproj_threshold: float = 3.0  # RANSAC maximum reprojection error in pixels
    min_tentative_matches: int = 10  # Minimum tentative matches required to attempt RANSAC
    min_inliers: int = 12  # Minimum absolute inlier count required for acceptance
    min_inlier_ratio: float = 0.20  # Minimum inlier fraction of tentative matches
    max_mean_reprojection_error: float = 4.0  # Pixels
    min_overlap_fraction: float = 0.15  # Minimum fraction of reference area covered by warp
    min_scale_ratio: float = 0.15  # Minimum area scale factor (avoids collapsed quadrilateral)
    max_scale_ratio: float = 6.0   # Maximum area scale factor (avoids blown-up quadrilateral)


@dataclass
class RegistrationResult:
    """
    Structured outcome of an image co-registration attempt.

    Coordinate conventions:
    - ref_size: (width, height)
    - query_size: (width, height)
    - homography: 3x3 matrix mapping (x_query, y_query, 1)^T -> (x_ref, y_ref, 1)^T
    - aligned_image: query image warped into reference image coordinate frame
    - overlap_mask: binary uint8 mask (0 or 255) where aligned query pixels overlap reference frame
    """
    success: bool
    status: AlignmentStatus
    method_used: str
    ref_size: Tuple[int, int]
    query_size: Tuple[int, int]
    homography: Optional[np.ndarray] = None
    num_keypoints_ref: int = 0
    num_keypoints_query: int = 0
    num_tentative_matches: int = 0
    num_inliers: int = 0
    inlier_ratio: float = 0.0
    reprojection_stats: Optional[ReprojectionStats] = None
    overlap_area_fraction: Optional[float] = None
    aligned_image: Optional[np.ndarray] = None
    overlap_mask: Optional[np.ndarray] = None
    failure_reason: Optional[str] = None
    latency_ms: float = 0.0
    selected_keyframe_id: Optional[str] = None
    is_direct_to_canonical: bool = True
    spatial_coverage: Optional[float] = None
    candidate_evaluations: Optional[list] = None
    is_promoted: bool = False
    decision_status: Optional[str] = None  # "ACCEPTED", "DEFERRED", "REJECTED"
    decision_reason: Optional[str] = None
