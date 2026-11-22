"""
Automated unit and regression tests for Phase 2 planar image co-registration.

Validates:
1. Identity, translation, and perspective transformations.
2. Blank and texture-poor image failure modes (without crashing).
3. Evaluator rejection of NaN/Inf, singular, folded, and high-error homographies.
4. Coordinate conventions, output dimensions, masks, and latency reporting.
5. SIFT and ORB feature extractors.
"""

import sys
import unittest
from pathlib import Path
import cv2
import numpy as np

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from objectmemory.registration.evaluator import AlignmentQualityEvaluator
from objectmemory.registration.models import (
    AlignmentConfig,
    AlignmentStatus,
    ReprojectionStats,
)
from objectmemory.registration.registrar import PlanarRegistrar


def _generate_synthetic_workbench(width: int = 400, height: int = 300, seed: int = 42) -> np.ndarray:
    """
    Generates a deterministic synthetic workbench image with rich geometric texture,
    grid lines, simulated PCB pads, and high-contrast corner features.
    """
    rng = np.random.RandomState(seed)
    img = np.ones((height, width, 3), dtype=np.uint8) * 45  # Dark workbench surface

    # Draw grid lines (anti-static mat simulation)
    for x in range(0, width, 20):
        cv2.line(img, (x, 0), (x, height), (70, 70, 70), 1)
    for y in range(0, height, 20):
        cv2.line(img, (0, y), (width, y), (70, 70, 70), 1)

    # Draw simulated PCB rectangle
    cv2.rectangle(img, (50, 50), (220, 180), (30, 120, 40), -1)
    cv2.rectangle(img, (50, 50), (220, 180), (200, 200, 200), 2)

    # Draw simulated chips and components with high-contrast edges
    cv2.rectangle(img, (80, 80), (130, 140), (20, 20, 20), -1)
    for px in range(85, 125, 8):
        cv2.circle(img, (px, 75), 2, (200, 180, 50), -1)
        cv2.circle(img, (px, 145), 2, (200, 180, 50), -1)

    # Draw simulated multimeter body
    cv2.rectangle(img, (260, 60), (360, 240), (20, 180, 240), -1)  # Yellow casing
    cv2.rectangle(img, (280, 80), (340, 130), (50, 60, 50), -1)    # Screen
    cv2.circle(img, (310, 180), 25, (40, 40, 40), -1)              # Dial

    # Add scattered high-contrast circular fiducial markers / solder points
    for _ in range(30):
        cx = rng.randint(20, width - 20)
        cy = rng.randint(20, height - 20)
        radius = rng.randint(3, 8)
        color = (int(rng.randint(150, 255)), int(rng.randint(150, 255)), int(rng.randint(150, 255)))
        cv2.circle(img, (cx, cy), radius, color, -1)
        cv2.circle(img, (cx, cy), radius + 2, (0, 0, 0), 1)

    return img


class TestPlanarRegistration(unittest.TestCase):

    def setUp(self):
        self.ref_img = _generate_synthetic_workbench(400, 300, seed=101)
        self.registrar = PlanarRegistrar()

    def test_identity_transformation(self):
        """Registering an image with itself must recover the identity homography."""
        query_img = self.ref_img.copy()
        result = self.registrar.register(self.ref_img, query_img)

        self.assertTrue(result.success)
        self.assertEqual(result.status, AlignmentStatus.SUCCESS)
        self.assertIsNotNone(result.homography)
        self.assertGreater(result.num_inliers, 15)
        self.assertGreater(result.inlier_ratio, 0.5)

        # H must be approximately the 3x3 identity matrix
        eye = np.eye(3, dtype=np.float64)
        np.testing.assert_allclose(result.homography, eye, atol=1e-1)

        # Reprojection error should be very low
        self.assertIsNotNone(result.reprojection_stats)
        self.assertLess(result.reprojection_stats.mean, 1.0)

        # Verify output dimensions and masks
        self.assertEqual(result.aligned_image.shape, self.ref_img.shape)
        self.assertEqual(result.overlap_mask.shape, self.ref_img.shape[:2])
        self.assertGreater(result.overlap_area_fraction, 0.95)

        # Verify measured latency
        self.assertGreater(result.latency_ms, 0.0)

    def test_known_translation(self):
        """Query image shifted by (dx, dy) must have shift accurately recovered."""
        dx, dy = 15.0, 10.0
        h, w = self.ref_img.shape[:2]
        T = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        query_img = cv2.warpAffine(self.ref_img, T, (w, h))

        result = self.registrar.register(self.ref_img, query_img)

        self.assertTrue(result.success)
        self.assertEqual(result.status, AlignmentStatus.SUCCESS)
        self.assertIsNotNone(result.homography)

        # Since query is shifted by (+dx, +dy) from ref:
        # p_query = p_ref + (dx, dy) => p_ref = p_query - (dx, dy)
        # Therefore H should have H[0, 2] ~= -dx, H[1, 2] ~= -dy
        recovered_dx = -result.homography[0, 2]
        recovered_dy = -result.homography[1, 2]
        self.assertAlmostEqual(recovered_dx, dx, delta=1.5)
        self.assertAlmostEqual(recovered_dy, dy, delta=1.5)

    def test_known_perspective_warp(self):
        """Synthetic perspective transformation within reasonable bounds is recovered."""
        h, w = self.ref_img.shape[:2]
        src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        # Apply modest tilt perspective
        dst_pts = np.float32([[15, 10], [w - 20, 5], [w - 10, h - 15], [20, h - 10]])
        H_true = cv2.getPerspectiveTransform(src_pts, dst_pts)
        query_img = cv2.warpPerspective(self.ref_img, H_true, (w, h))

        result = self.registrar.register(self.ref_img, query_img)

        self.assertTrue(result.success)
        self.assertEqual(result.status, AlignmentStatus.SUCCESS)
        self.assertIsNotNone(result.reprojection_stats)
        self.assertLess(result.reprojection_stats.mean, 2.5)

    def test_blank_images_fail_gracefully(self):
        """Completely uniform blank images must fail with INSUFFICIENT_KEYPOINTS, no crash."""
        blank_ref = np.zeros((300, 400, 3), dtype=np.uint8)
        blank_query = np.zeros((300, 400, 3), dtype=np.uint8)

        result = self.registrar.register(blank_ref, blank_query)

        self.assertFalse(result.success)
        self.assertIn(
            result.status,
            [AlignmentStatus.INSUFFICIENT_KEYPOINTS, AlignmentStatus.INSUFFICIENT_MATCHES],
        )
        self.assertIsNone(result.homography)
        self.assertIsNone(result.aligned_image)
        self.assertIsNotNone(result.failure_reason)

    def test_texture_poor_images_rejected(self):
        """Low-contrast smooth gradient image fails gracefully."""
        grad = np.tile(np.linspace(100, 110, 400, dtype=np.uint8), (300, 1))
        grad_3ch = cv2.cvtColor(grad, cv2.COLOR_GRAY2BGR)

        result = self.registrar.register(self.ref_img, grad_3ch)

        self.assertFalse(result.success)
        self.assertIn(
            result.status,
            [AlignmentStatus.INSUFFICIENT_KEYPOINTS, AlignmentStatus.INSUFFICIENT_MATCHES],
        )

    def test_orb_fallback_execution(self):
        """ORB detector configuration executes properly on CPU and produces valid alignment."""
        config_orb = AlignmentConfig(feature_method="ORB", min_inliers=8, min_inlier_ratio=0.15)
        registrar_orb = PlanarRegistrar(config_orb)

        result = registrar_orb.register(self.ref_img, self.ref_img)

        self.assertEqual(result.method_used, "ORB")
        self.assertTrue(result.success)
        self.assertEqual(result.status, AlignmentStatus.SUCCESS)

    def test_input_preservation(self):
        """Ensure input arrays are not modified in-place."""
        ref_copy = self.ref_img.copy()
        query_copy = self.ref_img.copy()

        self.registrar.register(ref_copy, query_copy)

        np.testing.assert_array_equal(ref_copy, self.ref_img)
        np.testing.assert_array_equal(query_copy, self.ref_img)


class TestAlignmentQualityEvaluator(unittest.TestCase):

    def setUp(self):
        self.evaluator = AlignmentQualityEvaluator()
        self.ref_size = (400, 300)
        self.query_size = (400, 300)

    def test_reject_none_and_invalid_shape(self):
        valid, status, reason, _, _ = self.evaluator.evaluate_homography(
            homography=None,
            inlier_pts_query=np.zeros((20, 2)),
            inlier_pts_ref=np.zeros((20, 2)),
            num_tentative_matches=30,
            ref_size=self.ref_size,
            query_size=self.query_size,
        )
        self.assertFalse(valid)
        self.assertEqual(status, AlignmentStatus.HOMOGRAPHY_ESTIMATION_FAILED)

    def test_reject_non_finite_coefficients(self):
        """Rejects homography matrices containing NaN or Inf."""
        for bad_val in [float("nan"), float("inf"), float("-inf")]:
            H_bad = np.eye(3)
            H_bad[0, 2] = bad_val

            valid, status, reason, _, _ = self.evaluator.evaluate_homography(
                homography=H_bad,
                inlier_pts_query=np.zeros((20, 2)),
                inlier_pts_ref=np.zeros((20, 2)),
                num_tentative_matches=30,
                ref_size=self.ref_size,
                query_size=self.query_size,
            )
            self.assertFalse(valid)
            self.assertEqual(status, AlignmentStatus.NON_FINITE_HOMOGRAPHY)

    def test_reject_singular_homography(self):
        """Rejects matrices with zero determinant or extreme condition number."""
        H_singular = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        valid, status, reason, _, _ = self.evaluator.evaluate_homography(
            homography=H_singular,
            inlier_pts_query=np.zeros((20, 2)),
            inlier_pts_ref=np.zeros((20, 2)),
            num_tentative_matches=30,
            ref_size=self.ref_size,
            query_size=self.query_size,
        )
        self.assertFalse(valid)
        self.assertEqual(status, AlignmentStatus.SINGULAR_HOMOGRAPHY)

    def test_reject_insufficient_inliers(self):
        """Rejects fits where inlier count is below threshold."""
        H_valid = np.eye(3)
        pts_q = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])  # Only 3 points
        pts_r = pts_q.copy()

        valid, status, reason, _, _ = self.evaluator.evaluate_homography(
            homography=H_valid,
            inlier_pts_query=pts_q,
            inlier_pts_ref=pts_r,
            num_tentative_matches=30,
            ref_size=self.ref_size,
            query_size=self.query_size,
        )
        self.assertFalse(valid)
        self.assertEqual(status, AlignmentStatus.INSUFFICIENT_INLIERS)

    def test_reject_excessive_reprojection_error(self):
        """Rejects homography when mean reprojection error exceeds threshold."""
        H_valid = np.eye(3)
        pts_q = np.random.uniform(20, 280, (25, 2))
        pts_r = pts_q + 15.0  # 15px error exceeds max_mean_reprojection_error (4.0px)

        valid, status, reason, stats, _ = self.evaluator.evaluate_homography(
            homography=H_valid,
            inlier_pts_query=pts_q,
            inlier_pts_ref=pts_r,
            num_tentative_matches=30,
            ref_size=self.ref_size,
            query_size=self.query_size,
        )
        self.assertFalse(valid)
        self.assertEqual(status, AlignmentStatus.EXCESSIVE_REPROJECTION_ERROR)
        self.assertGreater(stats.mean, 10.0)

    def test_reject_concave_or_folded_quadrilateral(self):
        """Rejects homography that flips or folds the query boundaries."""
        # A homography that reflects one axis, causing clockwise/counter-clockwise inversion
        H_inverted = np.array([[-1.0, 0.0, 400.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        # Mock inliers with zero error
        pts_q = np.array([[100.0, 100.0]] * 20)
        pts_r = np.array([[300.0, 100.0]] * 20)

        valid, status, reason, _, _ = self.evaluator.evaluate_homography(
            homography=H_inverted,
            inlier_pts_query=pts_q,
            inlier_pts_ref=pts_r,
            num_tentative_matches=25,
            ref_size=self.ref_size,
            query_size=self.query_size,
        )
        self.assertFalse(valid)
        self.assertEqual(status, AlignmentStatus.IMPLAUSIBLE_WARP_GEOMETRY)


if __name__ == "__main__":
    unittest.main()
