"""
Alignment quality evaluator for planar image co-registration.

Safeguards:
1. Rejects non-finite homography matrices (NaN, Inf).
2. Rejects near-singular and ill-conditioned matrices.
3. Rejects concave, folded, or self-intersecting quadrilateral projections.
4. Checks inlier count, inlier ratio, and reprojection error statistics together
   (never relies on inlier ratio alone).
5. Checks overlap area to ensure valid spatial coverage.
"""

from __future__ import annotations
import math
from typing import Optional, Tuple
import cv2
import numpy as np

from objectmemory.registration.models import (
    AlignmentConfig,
    AlignmentStatus,
    ReprojectionStats,
)


class AlignmentQualityEvaluator:
    """Evaluates the geometric plausibility and statistical accuracy of a homography."""

    def __init__(self, config: Optional[AlignmentConfig] = None) -> None:
        self.config = config or AlignmentConfig()

    def compute_reprojection_stats(
        self,
        homography: np.ndarray,
        pts_query: np.ndarray,
        pts_ref: np.ndarray,
    ) -> ReprojectionStats:
        """
        Computes reprojection error metrics for inlier correspondence pairs.
        Coordinates are (x, y) where x is horizontal column, y is vertical row.
        """
        if len(pts_query) == 0:
            return ReprojectionStats(
                mean=float("inf"), median=float("inf"), std=0.0, max_error=float("inf"), inlier_count=0
            )

        # Reshape to (N, 1, 2) for cv2.perspectiveTransform
        query_reshaped = pts_query.reshape(-1, 1, 2).astype(np.float64)
        projected = cv2.perspectiveTransform(query_reshaped, homography).reshape(-1, 2)

        # Euclidean error between projected query points and actual reference points
        errors = np.linalg.norm(projected - pts_ref, axis=1)

        return ReprojectionStats(
            mean=float(np.mean(errors)),
            median=float(np.median(errors)),
            std=float(np.std(errors)),
            max_error=float(np.max(errors)),
            inlier_count=len(errors),
        )

    def is_convex_quadrilateral(self, pts: np.ndarray) -> bool:
        """
        Checks if 4 ordered 2D points (x, y) form a strictly convex quadrilateral
        with non-zero area and no self-intersection.
        """
        if pts.shape != (4, 2):
            return False

        # Compute z-component of cross products of consecutive edges
        cross_products = []
        for i in range(4):
            p1 = pts[i]
            p2 = pts[(i + 1) % 4]
            p3 = pts[(i + 2) % 4]
            v1 = p2 - p1
            v2 = p3 - p2
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            cross_products.append(cross)

        # Must preserve original positive orientation and be strictly convex
        # Negative cross products indicate reflection/inversion; mixed signs indicate self-intersection/concavity.
        return all(cp > 1e-5 for cp in cross_products)

    def evaluate_homography(
        self,
        homography: Optional[np.ndarray],
        inlier_pts_query: np.ndarray,
        inlier_pts_ref: np.ndarray,
        num_tentative_matches: int,
        ref_size: Tuple[int, int],
        query_size: Tuple[int, int],
    ) -> Tuple[bool, AlignmentStatus, Optional[str], Optional[ReprojectionStats], Optional[float]]:
        """
        Runs comprehensive multi-criteria validation on an estimated homography.

        Returns:
            (is_valid, status, failure_reason, reprojection_stats, overlap_fraction)
        """
        # 1. Null / Shape check
        if homography is None or homography.shape != (3, 3):
            return (
                False,
                AlignmentStatus.HOMOGRAPHY_ESTIMATION_FAILED,
                "Homography matrix is None or invalid shape.",
                None,
                None,
            )

        # 2. Finite coefficient check
        if not np.all(np.isfinite(homography)):
            return (
                False,
                AlignmentStatus.NON_FINITE_HOMOGRAPHY,
                "Homography contains non-finite (NaN or Inf) values.",
                None,
                None,
            )

        # 3. Determinant and condition number check
        det = float(np.linalg.det(homography))
        if abs(det) < 1e-8:
            return (
                False,
                AlignmentStatus.SINGULAR_HOMOGRAPHY,
                f"Homography is near-singular (det={det:.2e}).",
                None,
                None,
            )

        cond = float(np.linalg.cond(homography))
        if cond > 1e7:
            return (
                False,
                AlignmentStatus.SINGULAR_HOMOGRAPHY,
                f"Homography is ill-conditioned (condition number={cond:.2e}).",
                None,
                None,
            )

        # 4. Inlier count and ratio check
        num_inliers = len(inlier_pts_query)
        if num_inliers < self.config.min_inliers:
            return (
                False,
                AlignmentStatus.INSUFFICIENT_INLIERS,
                f"Inlier count ({num_inliers}) is below minimum threshold ({self.config.min_inliers}).",
                None,
                None,
            )

        inlier_ratio = num_inliers / max(num_tentative_matches, 1)
        if inlier_ratio < self.config.min_inlier_ratio:
            return (
                False,
                AlignmentStatus.INSUFFICIENT_INLIERS,
                f"Inlier ratio ({inlier_ratio:.2f}) is below minimum threshold ({self.config.min_inlier_ratio:.2f}).",
                None,
                None,
            )

        # 5. Reprojection error computation and check
        stats = self.compute_reprojection_stats(homography, inlier_pts_query, inlier_pts_ref)
        if stats.mean > self.config.max_mean_reprojection_error:
            return (
                False,
                AlignmentStatus.EXCESSIVE_REPROJECTION_ERROR,
                f"Mean reprojection error ({stats.mean:.2f}px) exceeds threshold ({self.config.max_mean_reprojection_error:.2f}px).",
                stats,
                None,
            )

        # 6. Geometric plausibility: project corners
        qw, qh = query_size
        corners_query = np.array(
            [[0.0, 0.0], [float(qw), 0.0], [float(qw), float(qh)], [0.0, float(qh)]],
            dtype=np.float64,
        ).reshape(-1, 1, 2)

        try:
            warped_corners = cv2.perspectiveTransform(corners_query, homography).reshape(-1, 2)
        except cv2.error as exc:
            return (
                False,
                AlignmentStatus.IMPLAUSIBLE_WARP_GEOMETRY,
                f"OpenCV perspective transform failed for query corners: {exc}",
                stats,
                None,
            )

        if not np.all(np.isfinite(warped_corners)):
            return (
                False,
                AlignmentStatus.NON_FINITE_HOMOGRAPHY,
                "Projected query corners contain non-finite values.",
                stats,
                None,
            )

        # Verify convexity
        if not self.is_convex_quadrilateral(warped_corners):
            return (
                False,
                AlignmentStatus.IMPLAUSIBLE_WARP_GEOMETRY,
                "Transformed query frame is self-intersecting, concave, or inverted.",
                stats,
                None,
            )

        # Compute shoelace area of warped corners
        x = warped_corners[:, 0]
        y = warped_corners[:, 1]
        warped_area = 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
        original_area = float(qw * qh)
        scale_ratio = warped_area / max(original_area, 1.0)

        if scale_ratio < self.config.min_scale_ratio or scale_ratio > self.config.max_scale_ratio:
            return (
                False,
                AlignmentStatus.IMPLAUSIBLE_WARP_GEOMETRY,
                f"Warped area scale ratio ({scale_ratio:.2f}) out of plausible bounds "
                f"[{self.config.min_scale_ratio}, {self.config.max_scale_ratio}].",
                stats,
                None,
            )

        # 7. Overlap fraction check with reference frame
        rw, rh = ref_size
        ref_area = float(rw * rh)
        # Create a binary mask of query image and warp it to reference frame
        query_mask = np.ones((qh, qw), dtype=np.uint8) * 255
        warped_mask = cv2.warpPerspective(
            query_mask,
            homography,
            (rw, rh),
            flags=cv2.INTER_NEAREST,
        )
        overlap_pixels = int(np.count_nonzero(warped_mask > 0))
        overlap_fraction = overlap_pixels / max(ref_area, 1.0)

        if overlap_fraction < self.config.min_overlap_fraction:
            return (
                False,
                AlignmentStatus.INSUFFICIENT_OVERLAP,
                f"Overlap area fraction ({overlap_fraction:.2f}) is below minimum threshold ({self.config.min_overlap_fraction:.2f}).",
                stats,
                overlap_fraction,
            )

        # All checks passed
        return (True, AlignmentStatus.SUCCESS, None, stats, overlap_fraction)
