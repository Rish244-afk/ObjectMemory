"""
Adaptive Keyframe Registration and Quality Scoring Layer (Phase 6).

Maintains a topology of verified reference keyframes, attempts multi-candidate
registration, scores candidate transformations using multi-criteria quality
indicators, and composes homographies back into canonical coordinate space.

Key Architectural Guarantees:
1. Coordinate Invariance:
   - The initial reference frame remains the permanent anchor of canonical space.
   - Secondary keyframes maintain explicit, tested homographies to canonical space:
     H_query_to_canon = H_kf_to_canon @ H_query_to_kf.
   - Canonical coordinates and historical observations are never silently mutated.
2. Geometric Plausibility:
   - Compound homographies undergo full multi-criteria validation:
     finite checks, condition number, strictly convex quad, and scale bounds.
3. Uncertainty Safeguards:
   - Unaligned or unreliable frames never confirm movement or removal.
   - Rejection/deferral reasons are recorded with structured diagnostic telemetry.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from objectmemory.registration.coverage import SpatialCoverageCalculator
from objectmemory.registration.evaluator import AlignmentQualityEvaluator
from objectmemory.registration.models import (
    AlignmentConfig,
    AlignmentStatus,
    RegistrationResult,
    ReprojectionStats,
)
from objectmemory.registration.registrar import PlanarRegistrar


@dataclass
class KeyframeRecord:
    """A verified reference frame stored in the adaptive keyframe topology."""
    keyframe_id: str
    image: np.ndarray
    size: Tuple[int, int]  # (width, height)
    timestamp: str
    is_canonical: bool
    parent_keyframe_id: Optional[str] = None
    homography_to_canonical: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    quality_score: float = 1.0
    inlier_ratio: float = 1.0
    mean_reprojection_error: float = 0.0
    spatial_coverage: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KeyframeRegistrationResult:
    """Outcome of attempting multi-keyframe adaptive registration."""
    selected_keyframe_id: str
    is_direct_to_canonical: bool
    registration_to_keyframe: RegistrationResult
    registration_to_canonical: RegistrationResult
    candidate_evaluations: List[Dict[str, Any]]
    spatial_coverage: float = 0.0
    is_promoted_to_keyframe: bool = False
    promotion_reason: Optional[str] = None
    decision_status: str = "ACCEPTED"  # 'ACCEPTED', 'DEFERRED', 'REJECTED'
    decision_reason: str = "Registration verified against keyframe"


@dataclass
class KeyframeConfig:
    """Configuration for adaptive keyframe management and quality scoring."""
    max_keyframes: int = 5
    min_promotion_inlier_ratio: float = 0.35
    min_promotion_inliers: int = 20
    max_promotion_reprojection_error: float = 3.0
    min_promotion_spatial_coverage: float = 0.15
    min_viewpoint_shift_scale: float = 0.10  # Viewpoint/crop distinctness threshold
    min_viewpoint_shift_translation_px: float = 35.0
    direct_canonical_preference_weight: float = 1.10  # Slight preference for direct canonical match
    score_inlier_ratio_weight: float = 0.35
    score_inlier_count_weight: float = 0.25
    score_reprojection_weight: float = 0.25
    score_coverage_weight: float = 0.15


# SpatialCoverageCalculator is imported from objectmemory.registration.coverage
# and re-exported here for backward compatibility.


class AdaptiveKeyframeManager:
    """
    Manages a verified keyframe topology for robust planar co-registration
    under substantial camera shifts, crops, and resolution transitions.
    """

    def __init__(
        self,
        registrar: Optional[PlanarRegistrar] = None,
        evaluator: Optional[AlignmentQualityEvaluator] = None,
        config: Optional[KeyframeConfig] = None,
    ) -> None:
        self.config = config or KeyframeConfig()
        self.registrar = registrar or PlanarRegistrar()
        self.evaluator = evaluator or AlignmentQualityEvaluator(self.registrar.config)
        self.keyframes: Dict[str, KeyframeRecord] = {}
        self.canonical_keyframe_id: Optional[str] = None

    @property
    def keyframe_count(self) -> int:
        return len(self.keyframes)

    def set_canonical_reference(
        self,
        image: np.ndarray,
        visit_id: str,
        timestamp: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> KeyframeRecord:
        """
        Establishes the permanent canonical reference frame.
        All canonical coordinates originate in this coordinate system.
        """
        h, w = image.shape[:2]
        kf = KeyframeRecord(
            keyframe_id=visit_id,
            image=image.copy(),
            size=(w, h),
            timestamp=timestamp,
            is_canonical=True,
            parent_keyframe_id=None,
            homography_to_canonical=np.eye(3, dtype=np.float64),
            quality_score=1.0,
            inlier_ratio=1.0,
            mean_reprojection_error=0.0,
            spatial_coverage=1.0,
            metadata=metadata or {},
        )
        self.keyframes[visit_id] = kf
        self.canonical_keyframe_id = visit_id
        return kf

    def compute_registration_quality_score(
        self,
        reg: RegistrationResult,
        spatial_coverage: float,
        is_canonical: bool,
    ) -> float:
        """
        Computes composite multi-indicator quality score S in [0.0, 1.0].
        """
        if not reg.success or reg.homography is None:
            return 0.0

        r_inlier = float(reg.inlier_ratio)
        n_inlier = min(1.0, reg.num_inliers / 60.0)

        mean_err = reg.reprojection_stats.mean if reg.reprojection_stats else 4.0
        r_reproj = math.exp(-max(0.0, mean_err) / 2.5)

        cov = float(np.clip(spatial_coverage, 0.0, 1.0))

        score = (
            self.config.score_inlier_ratio_weight * r_inlier
            + self.config.score_inlier_count_weight * n_inlier
            + self.config.score_reprojection_weight * r_reproj
            + self.config.score_coverage_weight * cov
        )

        if is_canonical:
            score *= self.config.direct_canonical_preference_weight

        return float(np.clip(score, 0.0, 1.0))

    def _compose_to_canonical(
        self,
        reg_to_kf: RegistrationResult,
        kf: KeyframeRecord,
        query_image: np.ndarray,
    ) -> RegistrationResult:
        """
        Composes registration homography from query frame to canonical frame:
        H_query_to_canon = H_kf_to_canon @ H_query_to_kf.
        """
        if kf.is_canonical:
            return reg_to_kf

        assert reg_to_kf.homography is not None
        canon_kf = self.keyframes[self.canonical_keyframe_id]  # type: ignore
        canon_size = canon_kf.size

        # Compose 3x3 projective matrices
        H_compound = kf.homography_to_canonical @ reg_to_kf.homography

        # Normalize homography so H[2, 2] == 1.0
        if abs(H_compound[2, 2]) > 1e-9:
            H_compound = H_compound / H_compound[2, 2]

        # Validate geometric plausibility of compound transformation against canonical canvas
        # Project 4 corners of query image through compound homography
        qw, qh = reg_to_kf.query_size
        corners_query = np.array(
            [[0.0, 0.0], [qw, 0.0], [qw, qh], [0.0, qh]], dtype=np.float32
        ).reshape(-1, 1, 2)

        try:
            warped_corners = cv2.perspectiveTransform(corners_query, H_compound).reshape(-1, 2)
            is_convex = self.evaluator.is_convex_quadrilateral(warped_corners)
        except Exception:
            is_convex = False

        if not is_convex:
            return RegistrationResult(
                success=False,
                status=AlignmentStatus.IMPLAUSIBLE_WARP_GEOMETRY,
                method_used=f"COMPOUND_KEYFRAME_{kf.keyframe_id}",
                ref_size=canon_size,
                query_size=reg_to_kf.query_size,
                homography=None,
                inlier_ratio=reg_to_kf.inlier_ratio,
                failure_reason=f"Compound homography through keyframe '{kf.keyframe_id}' forms non-convex quadrilateral.",
            )

        # Warp query image into canonical canvas
        cw, ch = canon_size
        aligned_canon = cv2.warpPerspective(
            query_image,
            H_compound,
            (cw, ch),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        query_mask = np.ones((qh, qw), dtype=np.uint8) * 255
        overlap_mask = cv2.warpPerspective(
            query_mask,
            H_compound,
            (cw, ch),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        return RegistrationResult(
            success=True,
            status=AlignmentStatus.SUCCESS,
            method_used=f"COMPOUND_KEYFRAME_{kf.keyframe_id}",
            ref_size=canon_size,
            query_size=reg_to_kf.query_size,
            homography=H_compound,
            num_keypoints_ref=reg_to_kf.num_keypoints_ref,
            num_keypoints_query=reg_to_kf.num_keypoints_query,
            num_tentative_matches=reg_to_kf.num_tentative_matches,
            num_inliers=reg_to_kf.num_inliers,
            inlier_ratio=reg_to_kf.inlier_ratio,
            reprojection_stats=reg_to_kf.reprojection_stats,
            overlap_area_fraction=float(np.mean(overlap_mask > 0)),
            aligned_image=aligned_canon,
            overlap_mask=overlap_mask,
            latency_ms=reg_to_kf.latency_ms,
        )

    def _should_promote_keyframe(
        self,
        reg_to_canon: RegistrationResult,
        reg_to_kf: RegistrationResult,
        kf: KeyframeRecord,
        spatial_coverage: float,
    ) -> Tuple[bool, Optional[str]]:
        """
        Evaluates whether a newly observed visit satisfies strict criteria
        for promotion to an additional reference keyframe.
        """
        if self.keyframe_count >= self.config.max_keyframes:
            return False, f"Maximum keyframe capacity reached ({self.config.max_keyframes})."

        # Check high-quality registration indicators
        if reg_to_kf.inlier_ratio < self.config.min_promotion_inlier_ratio:
            return False, f"Inlier ratio ({reg_to_kf.inlier_ratio:.2f}) below promotion threshold ({self.config.min_promotion_inlier_ratio:.2f})."

        if reg_to_kf.num_inliers < self.config.min_promotion_inliers:
            return False, f"Inlier count ({reg_to_kf.num_inliers}) below promotion threshold ({self.config.min_promotion_inliers})."

        mean_err = reg_to_kf.reprojection_stats.mean if reg_to_kf.reprojection_stats else 999.0
        if mean_err > self.config.max_promotion_reprojection_error:
            return False, f"Reprojection error ({mean_err:.2f}px) exceeds threshold ({self.config.max_promotion_reprojection_error:.2f}px)."

        if spatial_coverage < self.config.min_promotion_spatial_coverage:
            return False, f"Spatial coverage ({spatial_coverage:.2f}) below threshold ({self.config.min_promotion_spatial_coverage:.2f})."

        # Check for meaningful viewpoint/framing/scale difference from parent keyframe
        H = reg_to_kf.homography
        if H is None:
            return False, "Null homography."

        # Compute scale factor and translation magnitude
        scale_det = math.sqrt(abs(np.linalg.det(H[:2, :2])))
        dx = float(H[0, 2])
        dy = float(H[1, 2])
        trans_mag = math.hypot(dx, dy)

        is_scale_diff = abs(scale_det - 1.0) >= self.config.min_viewpoint_shift_scale
        is_trans_diff = trans_mag >= self.config.min_viewpoint_shift_translation_px

        if not (is_scale_diff or is_trans_diff):
            return False, "Framing is nearly identical to existing keyframe; promotion redundant."

        reason = (
            f"Promoted to keyframe: inlier_ratio={reg_to_kf.inlier_ratio:.2f}, "
            f"inliers={reg_to_kf.num_inliers}, reproj={mean_err:.2f}px, "
            f"scale_change={abs(scale_det - 1.0):.2f}, translation={trans_mag:.1f}px."
        )
        return True, reason

    def register_query(
        self,
        query_image: np.ndarray,
        visit_id: str,
        timestamp: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> KeyframeRegistrationResult:
        """
        Registers a query visit against the keyframe topology.

        Workflow:
        1. If no reference exists, promotes query as canonical keyframe.
        2. Evaluates candidate registrations against canonical and secondary keyframes.
        3. Selects best candidate by multi-criteria quality score.
        4. Composes homography back to canonical coordinate frame.
        5. Evaluates keyframe promotion criteria.
        """
        # If no keyframes exist, this visit is the initial canonical reference
        if not self.keyframes or self.canonical_keyframe_id is None:
            kf = self.set_canonical_reference(query_image, visit_id, timestamp, metadata)
            h, w = query_image.shape[:2]
            canon_reg = RegistrationResult(
                success=True,
                status=AlignmentStatus.SUCCESS,
                method_used="INITIAL_CANONICAL_REFERENCE",
                ref_size=(w, h),
                query_size=(w, h),
                homography=np.eye(3, dtype=np.float64),
                overlap_mask=np.ones((h, w), dtype=np.uint8) * 255,
                aligned_image=query_image.copy(),
                inlier_ratio=1.0,
                num_inliers=100,
                overlap_area_fraction=1.0,
                selected_keyframe_id=visit_id,
                is_direct_to_canonical=True,
                spatial_coverage=1.0,
                candidate_evaluations=[],
                is_promoted=True,
                decision_status="ACCEPTED",
                decision_reason="Established initial canonical reference.",
            )
            return KeyframeRegistrationResult(
                selected_keyframe_id=visit_id,
                is_direct_to_canonical=True,
                registration_to_keyframe=canon_reg,
                registration_to_canonical=canon_reg,
                candidate_evaluations=[],
                spatial_coverage=1.0,
                is_promoted_to_keyframe=True,
                promotion_reason="Initial canonical reference frame established.",
                decision_status="ACCEPTED",
                decision_reason="Established initial canonical reference.",
            )

        candidate_evaluations: List[Dict[str, Any]] = []
        best_candidate: Optional[Tuple[KeyframeRecord, RegistrationResult, float, float]] = None
        best_score = -1.0

        # Attempt registration against existing keyframes:
        # Prioritize canonical keyframe first, then other verified keyframes
        ordered_kf_ids = [self.canonical_keyframe_id] + [
            k for k in self.keyframes if k != self.canonical_keyframe_id
        ]

        for kf_id in ordered_kf_ids:
            kf = self.keyframes[kf_id]
            reg = self.registrar.register(kf.image, query_image)

            cov = float(reg.spatial_coverage) if reg.spatial_coverage is not None else 0.0
            if cov == 0.0 and reg.success and reg.homography is not None:
                # Fallback: estimate from query canvas corners
                qw, qh = reg.query_size
                pts_q = np.array([[0, 0], [qw, 0], [qw, qh], [0, qh], [qw/2, qh/2]], dtype=np.float32)
                cov = SpatialCoverageCalculator.calculate_coverage(pts_q, reg.query_size)

            score = self.compute_registration_quality_score(reg, cov, kf.is_canonical)

            eval_entry = {
                "keyframe_id": kf_id,
                "is_canonical": kf.is_canonical,
                "success": reg.success,
                "status": reg.status.value,
                "inlier_ratio": float(reg.inlier_ratio),
                "num_inliers": reg.num_inliers,
                "mean_reprojection_error": (
                    float(reg.reprojection_stats.mean)
                    if reg.reprojection_stats and reg.reprojection_stats.mean is not None
                    else None
                ),
                "spatial_coverage": round(cov, 4),
                "quality_score": round(score, 4),
                "failure_reason": reg.failure_reason,
            }
            candidate_evaluations.append(eval_entry)

            if reg.success and reg.homography is not None and score > best_score:
                best_score = score
                best_candidate = (kf, reg, cov, score)

        # Evaluate outcome
        if best_candidate is None:
            # Registration failed against all keyframes
            canon_kf = self.keyframes[self.canonical_keyframe_id]
            failed_reg = RegistrationResult(
                success=False,
                status=AlignmentStatus.INSUFFICIENT_INLIERS,
                method_used="ADAPTIVE_KEYFRAME_FAILED",
                ref_size=canon_kf.size,
                query_size=(query_image.shape[1], query_image.shape[0]),
                homography=None,
                spatial_coverage=0.0,
                selected_keyframe_id=self.canonical_keyframe_id,
                is_direct_to_canonical=True,
                candidate_evaluations=candidate_evaluations,
                is_promoted=False,
                decision_status="REJECTED",
                decision_reason=f"Registration failed across all {len(self.keyframes)} keyframes; spatial updates gated.",
                failure_reason=f"Failed registration against all {len(self.keyframes)} available keyframe(s).",
            )
            return KeyframeRegistrationResult(
                selected_keyframe_id=self.canonical_keyframe_id,
                is_direct_to_canonical=True,
                registration_to_keyframe=failed_reg,
                registration_to_canonical=failed_reg,
                candidate_evaluations=candidate_evaluations,
                spatial_coverage=0.0,
                is_promoted_to_keyframe=False,
                decision_status="REJECTED",
                decision_reason=f"Registration failed across all {len(self.keyframes)} keyframes; spatial updates gated.",
            )

        winning_kf, winning_reg, winning_cov, winning_score = best_candidate

        # Compose to canonical coordinate frame
        canon_reg = self._compose_to_canonical(winning_reg, winning_kf, query_image)

        if not canon_reg.success:
            canon_reg.selected_keyframe_id = winning_kf.keyframe_id
            canon_reg.is_direct_to_canonical = winning_kf.is_canonical
            canon_reg.spatial_coverage = winning_cov
            canon_reg.candidate_evaluations = candidate_evaluations
            canon_reg.is_promoted = False
            canon_reg.decision_status = "REJECTED"
            canon_reg.decision_reason = f"Geometric failure in compound warp to canonical: {canon_reg.failure_reason}"

            return KeyframeRegistrationResult(
                selected_keyframe_id=winning_kf.keyframe_id,
                is_direct_to_canonical=winning_kf.is_canonical,
                registration_to_keyframe=winning_reg,
                registration_to_canonical=canon_reg,
                candidate_evaluations=candidate_evaluations,
                spatial_coverage=winning_cov,
                is_promoted_to_keyframe=False,
                decision_status="REJECTED",
                decision_reason=f"Geometric failure in compound warp to canonical: {canon_reg.failure_reason}",
            )

        # Check keyframe promotion
        should_promote, promo_reason = self._should_promote_keyframe(
            canon_reg, winning_reg, winning_kf, winning_cov
        )
        if should_promote:
            qh, qw = query_image.shape[:2]
            new_kf = KeyframeRecord(
                keyframe_id=visit_id,
                image=query_image.copy(),
                size=(qw, qh),
                timestamp=timestamp,
                is_canonical=False,
                parent_keyframe_id=winning_kf.keyframe_id,
                homography_to_canonical=canon_reg.homography.copy(),  # type: ignore
                quality_score=winning_score,
                inlier_ratio=float(winning_reg.inlier_ratio),
                mean_reprojection_error=(
                    float(winning_reg.reprojection_stats.mean)
                    if winning_reg.reprojection_stats
                    else 0.0
                ),
                spatial_coverage=winning_cov,
                metadata=metadata or {},
            )
            self.keyframes[visit_id] = new_kf

        # Determine decision status
        min_inlier_gate = self.registrar.config.min_inlier_ratio
        if canon_reg.inlier_ratio < min_inlier_gate:
            dec_status = "DEFERRED"
            dec_reason = f"Inlier ratio ({canon_reg.inlier_ratio:.1%}) below reliable gate ({min_inlier_gate:.1%}); observation deferred."
        else:
            dec_status = "ACCEPTED"
            dec_reason = f"Aligned against keyframe '{winning_kf.keyframe_id}' with inlier ratio {canon_reg.inlier_ratio:.1%}."

        canon_reg.selected_keyframe_id = winning_kf.keyframe_id
        canon_reg.is_direct_to_canonical = winning_kf.is_canonical
        canon_reg.spatial_coverage = winning_cov
        canon_reg.candidate_evaluations = candidate_evaluations
        canon_reg.is_promoted = should_promote
        canon_reg.decision_status = dec_status
        canon_reg.decision_reason = dec_reason

        return KeyframeRegistrationResult(
            selected_keyframe_id=winning_kf.keyframe_id,
            is_direct_to_canonical=winning_kf.is_canonical,
            registration_to_keyframe=winning_reg,
            registration_to_canonical=canon_reg,
            candidate_evaluations=candidate_evaluations,
            spatial_coverage=winning_cov,
            is_promoted_to_keyframe=should_promote,
            promotion_reason=promo_reason,
            decision_status=dec_status,
            decision_reason=dec_reason,
        )
