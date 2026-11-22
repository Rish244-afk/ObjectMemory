"""
Planar image co-registration engine for ObjectMemory.

Estimates projective homography transformations using SIFT (with ORB fallback)
and RANSAC, mapping query visits into the reference visit coordinate frame.

Coordinate conventions:
- Points are (x, y) where x is horizontal column, y is vertical row.
- Dimensions are (width, height) = (cols, rows).
- Homography matrix H maps query points to reference: p_ref ~ H @ p_query.
- Aligned image dimensions match reference image dimensions.
"""

from __future__ import annotations
import time
from typing import Optional, Tuple
import cv2
import numpy as np

from objectmemory.registration.coverage import SpatialCoverageCalculator
from objectmemory.registration.evaluator import AlignmentQualityEvaluator
from objectmemory.registration.models import (
    AlignmentConfig,
    AlignmentStatus,
    RegistrationResult,
)


class PlanarRegistrar:
    """
    Co-registers repeated workbench images into a canonical reference frame.
    Executes purely on CPU.
    """

    def __init__(self, config: Optional[AlignmentConfig] = None) -> None:
        self.config = config or AlignmentConfig()
        self.evaluator = AlignmentQualityEvaluator(self.config)

    def _get_detector(self) -> Tuple[cv2.Feature2D, str, int]:
        """
        Instantiates feature detector. Prefers SIFT where available; falls back to ORB.
        Returns (detector, method_name, norm_type).
        """
        requested = self.config.feature_method.upper()
        if requested == "SIFT":
            try:
                detector = cv2.SIFT_create(nfeatures=self.config.max_features)
                return detector, "SIFT", cv2.NORM_L2
            except (AttributeError, cv2.error):
                # Fallback to ORB if SIFT is unavailable in OpenCV build
                pass

        # ORB extractor
        detector = cv2.ORB_create(nfeatures=self.config.max_features)
        return detector, "ORB", cv2.NORM_HAMMING

    def register(
        self,
        reference_image: np.ndarray,
        query_image: np.ndarray,
    ) -> RegistrationResult:
        """
        Co-registers query_image into reference_image's coordinate frame.

        Args:
            reference_image: 2D (grayscale) or 3D (color) numpy array. Not modified in-place.
            query_image: 2D (grayscale) or 3D (color) numpy array. Not modified in-place.

        Returns:
            Structured RegistrationResult.
        """
        start_time = time.perf_counter()

        ref_h, ref_w = reference_image.shape[:2]
        query_h, query_w = query_image.shape[:2]
        ref_size = (ref_w, ref_h)
        query_size = (query_w, query_h)

        # Convert to grayscale for detection if color
        gray_ref = (
            reference_image
            if reference_image.ndim == 2
            else cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
        )
        gray_query = (
            query_image
            if query_image.ndim == 2
            else cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
        )

        detector, method_used, norm_type = self._get_detector()

        # 1. Feature detection and description
        try:
            kp_ref, desc_ref = detector.detectAndCompute(gray_ref, None)
            kp_query, desc_query = detector.detectAndCompute(gray_query, None)
        except cv2.error as exc:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return RegistrationResult(
                success=False,
                status=AlignmentStatus.EXECUTION_ERROR,
                method_used=method_used,
                ref_size=ref_size,
                query_size=query_size,
                failure_reason=f"OpenCV detectAndCompute failed: {exc}",
                latency_ms=elapsed,
            )

        num_kp_ref = len(kp_ref) if kp_ref else 0
        num_kp_query = len(kp_query) if kp_query else 0

        if desc_ref is None or desc_query is None or num_kp_ref < 4 or num_kp_query < 4:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return RegistrationResult(
                success=False,
                status=AlignmentStatus.INSUFFICIENT_KEYPOINTS,
                method_used=method_used,
                ref_size=ref_size,
                query_size=query_size,
                num_keypoints_ref=num_kp_ref,
                num_keypoints_query=num_kp_query,
                failure_reason=f"Insufficient keypoints extracted (ref={num_kp_ref}, query={num_kp_query}).",
                latency_ms=elapsed,
            )

        # 2. Descriptor matching (query -> ref)
        matcher = cv2.BFMatcher(norm_type)
        try:
            raw_matches = matcher.knnMatch(desc_query, desc_ref, k=2)
        except cv2.error as exc:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return RegistrationResult(
                success=False,
                status=AlignmentStatus.EXECUTION_ERROR,
                method_used=method_used,
                ref_size=ref_size,
                query_size=query_size,
                num_keypoints_ref=num_kp_ref,
                num_keypoints_query=num_kp_query,
                failure_reason=f"Feature matching failed: {exc}",
                latency_ms=elapsed,
            )

        # Lowe's ratio test
        good_matches = []
        for pair in raw_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < self.config.ratio_test_threshold * n.distance:
                    good_matches.append(m)

        num_tentative = len(good_matches)
        if num_tentative < self.config.min_tentative_matches:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return RegistrationResult(
                success=False,
                status=AlignmentStatus.INSUFFICIENT_MATCHES,
                method_used=method_used,
                ref_size=ref_size,
                query_size=query_size,
                num_keypoints_ref=num_kp_ref,
                num_keypoints_query=num_kp_query,
                num_tentative_matches=num_tentative,
                failure_reason=(
                    f"Tentative matches ({num_tentative}) below minimum threshold "
                    f"({self.config.min_tentative_matches})."
                ),
                latency_ms=elapsed,
            )

        # 3. Extract correspondence coordinate arrays
        pts_query = np.float32([kp_query[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        pts_ref = np.float32([kp_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

        # 4. RANSAC Homography Estimation
        try:
            H, mask = cv2.findHomography(
                pts_query,
                pts_ref,
                cv2.RANSAC,
                self.config.ransac_reproj_threshold,
            )
        except cv2.error as exc:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return RegistrationResult(
                success=False,
                status=AlignmentStatus.HOMOGRAPHY_ESTIMATION_FAILED,
                method_used=method_used,
                ref_size=ref_size,
                query_size=query_size,
                num_keypoints_ref=num_kp_ref,
                num_keypoints_query=num_kp_query,
                num_tentative_matches=num_tentative,
                failure_reason=f"cv2.findHomography threw an error: {exc}",
                latency_ms=elapsed,
            )

        if H is None or mask is None:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return RegistrationResult(
                success=False,
                status=AlignmentStatus.HOMOGRAPHY_ESTIMATION_FAILED,
                method_used=method_used,
                ref_size=ref_size,
                query_size=query_size,
                num_keypoints_ref=num_kp_ref,
                num_keypoints_query=num_kp_query,
                num_tentative_matches=num_tentative,
                failure_reason="RANSAC failed to find a valid geometric consensus.",
                latency_ms=elapsed,
            )

        # Inlier filtering
        inlier_mask = mask.ravel() == 1
        inlier_pts_query = pts_query[inlier_mask]
        inlier_pts_ref = pts_ref[inlier_mask]
        num_inliers = len(inlier_pts_query)
        inlier_ratio = num_inliers / max(num_tentative, 1)

        # Compute spatial feature coverage across query frame
        spatial_cov = SpatialCoverageCalculator.calculate_coverage(inlier_pts_query, query_size)

        # Normalize homography matrix so H[2, 2] == 1.0 if non-zero
        if np.all(np.isfinite(H)) and abs(H[2, 2]) > 1e-12:
            H = H / H[2, 2]

        # 5. Quality evaluation gate
        is_valid, status, failure_reason, stats, overlap_fraction = (
            self.evaluator.evaluate_homography(
                homography=H,
                inlier_pts_query=inlier_pts_query,
                inlier_pts_ref=inlier_pts_ref,
                num_tentative_matches=num_tentative,
                ref_size=ref_size,
                query_size=query_size,
            )
        )

        elapsed = (time.perf_counter() - start_time) * 1000.0

        if not is_valid:
            return RegistrationResult(
                success=False,
                status=status,
                method_used=method_used,
                ref_size=ref_size,
                query_size=query_size,
                homography=H,
                num_keypoints_ref=num_kp_ref,
                num_keypoints_query=num_kp_query,
                num_tentative_matches=num_tentative,
                num_inliers=num_inliers,
                inlier_ratio=inlier_ratio,
                reprojection_stats=stats,
                spatial_coverage=spatial_cov,
                overlap_area_fraction=overlap_fraction,
                failure_reason=failure_reason,
                latency_ms=elapsed,
            )

        # 6. Transform query image and compute overlap mask
        aligned_image = cv2.warpPerspective(query_image, H, (ref_w, ref_h))
        query_mask_canvas = np.ones((query_h, query_w), dtype=np.uint8) * 255
        overlap_mask = cv2.warpPerspective(
            query_mask_canvas,
            H,
            (ref_w, ref_h),
            flags=cv2.INTER_NEAREST,
        )

        return RegistrationResult(
            success=True,
            status=AlignmentStatus.SUCCESS,
            method_used=method_used,
            ref_size=ref_size,
            query_size=query_size,
            homography=H,
            num_keypoints_ref=num_kp_ref,
            num_keypoints_query=num_kp_query,
            num_tentative_matches=num_tentative,
            num_inliers=num_inliers,
            inlier_ratio=inlier_ratio,
            reprojection_stats=stats,
            spatial_coverage=spatial_cov,
            overlap_area_fraction=overlap_fraction,
            aligned_image=aligned_image,
            overlap_mask=overlap_mask,
            failure_reason=None,
            latency_ms=elapsed,
        )
