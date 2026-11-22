"""
Planar Image Co-Registration and Alignment Quality Gating module for ObjectMemory.

Provides:
- Feature extraction and matching (SIFT with ORB fallback).
- RANSAC planar homography estimation and coordinate warping.
- Rigorous multi-criteria alignment quality evaluation.
- Adaptive keyframe registration topology and quality scoring.
"""

from objectmemory.registration.models import (
    AlignmentConfig,
    AlignmentStatus,
    RegistrationResult,
    ReprojectionStats,
)
from objectmemory.registration.evaluator import AlignmentQualityEvaluator
from objectmemory.registration.registrar import PlanarRegistrar
from objectmemory.registration.keyframe import (
    AdaptiveKeyframeManager,
    KeyframeConfig,
    KeyframeRecord,
    KeyframeRegistrationResult,
    SpatialCoverageCalculator,
)

__all__ = [
    "AlignmentConfig",
    "AlignmentStatus",
    "RegistrationResult",
    "ReprojectionStats",
    "AlignmentQualityEvaluator",
    "PlanarRegistrar",
    "AdaptiveKeyframeManager",
    "KeyframeConfig",
    "KeyframeRecord",
    "KeyframeRegistrationResult",
    "SpatialCoverageCalculator",
]
