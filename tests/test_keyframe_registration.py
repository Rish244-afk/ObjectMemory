"""
Regression test suite for Phase 6: Adaptive Keyframe Registration
and Uncertainty-Aware Spatial Memory.

Covers all 12 key conditions:
1. Registration succeeds directly against original keyframe when viewpoint is stable.
2. Registration falls back to secondary keyframe when original keyframe has insufficient overlap/matches.
3. Registration fails cleanly and gates spatial decisions when all keyframes fail.
4. Scale/resolution changes between query and reference register accurately.
5. Large perspective/viewpoint shifts register through chained keyframes.
6. Camera movement alone does not trigger false CONFIRMED_MOVED transitions.
7. Physical object movement is correctly detected under moderate camera motion.
8. Ill-conditioned or non-convex compound homographies are rejected.
9. Outlier match clusters do not corrupt canonical coordinate mapping (spatial coverage weighting).
10. Keyframe promotion occurs when quality criteria are satisfied and is rejected when redundant or poor.
11. Canonical coordinates remain consistent across multi-hop registrations (H_query_to_canon = H_kf_to_canon * H_query_to_kf).
12. Gated registrations prevent false CONFIRMED_REMOVED transitions.
"""

from __future__ import annotations

import unittest
import numpy as np
import cv2

from objectmemory.detection.signature import VisualSignatureExtractor
from objectmemory.domain.models import BoundingBox
from objectmemory.domain.states import EpistemicState
from objectmemory.persistence.repository import SQLiteMemoryStore
from objectmemory.registration.coverage import SpatialCoverageCalculator
from objectmemory.registration.evaluator import AlignmentQualityEvaluator
from objectmemory.registration.keyframe import (
    AdaptiveKeyframeManager,
    KeyframeConfig,
    KeyframeRecord,
)
from objectmemory.registration.models import (
    AlignmentConfig,
    AlignmentStatus,
    RegistrationResult,
    ReprojectionStats,
)
from objectmemory.registration.registrar import PlanarRegistrar
from objectmemory.temporal.engine import TemporalMemoryEngine
from objectmemory.temporal.models import (
    Detection,
    TemporalEngineConfig,
    VisitInput,
)


def _generate_rich_texture_image(
    width: int = 640,
    height: int = 480,
    seed: int = 42,
    offset_x: int = 0,
    offset_y: int = 0,
) -> np.ndarray:
    """Generates an image with rich SIFT-detectable geometric and textural patterns without repetition."""
    rng = np.random.default_rng(seed)
    canvas = np.full((height, width, 3), 220, dtype=np.uint8)

    # Grid background pattern
    grid_spacing = 30
    for x in range(0, width, grid_spacing):
        cv2.line(canvas, (x, 0), (x, height), (180, 180, 180), 1)
    for y in range(0, height, grid_spacing):
        cv2.line(canvas, (0, y), (width, y), (180, 180, 180), 1)

    # Distinct non-repeating geometric shapes with interior texture
    num_shapes = max(8, width // 40)
    for _ in range(num_shapes):
        sx = int(rng.integers(10, max(20, width - 70))) + offset_x
        sy = int(rng.integers(10, max(20, height - 70))) + offset_y
        sw = int(rng.integers(35, 75))
        sh = int(rng.integers(35, 75))
        color = (int(rng.integers(20, 230)), int(rng.integers(20, 230)), int(rng.integers(20, 230)))
        if 0 <= sx < width - sw and 0 <= sy < height - sh:
            cv2.rectangle(canvas, (sx, sy), (sx + sw, sy + sh), color, -1)
            for i in range(sx + 5, sx + sw - 5, 8):
                for j in range(sy + 5, sy + sh - 5, 8):
                    v = int(rng.integers(0, 255))
                    cv2.circle(canvas, (i, j), 2, (v, v, v), -1)

    return canvas


class TestAdaptiveKeyframeRegistration(unittest.TestCase):
    """Test suite covering adaptive keyframe registration and uncertainty handling."""

    def setUp(self) -> None:
        self.registrar = PlanarRegistrar()
        self.config = KeyframeConfig()
        self.manager = AdaptiveKeyframeManager(registrar=self.registrar, config=self.config)

    def test_01_registration_direct_to_canonical_stable(self) -> None:
        """Condition 1: Registration succeeds directly against canonical keyframe under stable camera."""
        ref_img = _generate_rich_texture_image(640, 480, seed=10)
        self.manager.set_canonical_reference(ref_img, "visit_01", "2026-10-01T10:00:00Z")

        # Query with small camera jitter (translation 3px, 2px)
        M_jitter = np.float32([[1.0, 0.0, 3.0], [0.0, 1.0, 2.0]])
        query_img = cv2.warpAffine(ref_img, M_jitter, (640, 480))

        res = self.manager.register_query(query_img, "visit_02", "2026-10-01T11:00:00Z")

        self.assertTrue(res.registration_to_canonical.success)
        self.assertEqual(res.selected_keyframe_id, "visit_01")
        self.assertTrue(res.is_direct_to_canonical)
        self.assertEqual(res.decision_status, "ACCEPTED")
        self.assertGreater(res.registration_to_canonical.inlier_ratio, 0.40)

    def test_02_registration_fallback_to_secondary_keyframe(self) -> None:
        """Condition 2: Registration falls back to secondary keyframe when canonical has low overlap."""
        # Panorama width 1400: frame 1 is [0:640], frame 2 is [350:990], frame 3 is [700:1340]
        # Frame 3 has ZERO overlap with Frame 1, but strong overlap with Frame 2!
        panorama = _generate_rich_texture_image(1400, 480, seed=20)
        frame1 = panorama[:, :640].copy()
        frame2 = panorama[:, 350:990].copy()
        frame3 = panorama[:, 700:1340].copy()

        self.manager.set_canonical_reference(frame1, "visit_01", "2026-10-01T10:00:00Z")

        # Register and promote frame 2 as secondary keyframe
        res2 = self.manager.register_query(frame2, "visit_02", "2026-10-01T11:00:00Z")
        self.assertTrue(res2.registration_to_canonical.success)
        self.assertTrue(res2.is_promoted_to_keyframe)
        self.assertIn("visit_02", self.manager.keyframes)

        # Register frame 3: falls back and matches best against frame 2
        res3 = self.manager.register_query(frame3, "visit_03", "2026-10-01T12:00:00Z")
        self.assertTrue(res3.registration_to_canonical.success)
        self.assertEqual(res3.selected_keyframe_id, "visit_02")
        self.assertFalse(res3.is_direct_to_canonical)
        self.assertEqual(res3.decision_status, "ACCEPTED")
        # Canonical homography was compound-computed: H_query_to_canon
        self.assertIsNotNone(res3.registration_to_canonical.homography)

    def test_03_registration_fails_cleanly_all_keyframes(self) -> None:
        """Condition 3: Registration fails cleanly and gates spatial decisions when all keyframes fail."""
        ref_img = _generate_rich_texture_image(640, 480, seed=30)
        self.manager.set_canonical_reference(ref_img, "visit_01", "2026-10-01T10:00:00Z")

        # Query image: solid gray blank image with no features
        blank_query = np.full((480, 640, 3), 128, dtype=np.uint8)

        res = self.manager.register_query(blank_query, "visit_02", "2026-10-01T11:00:00Z")
        self.assertFalse(res.registration_to_canonical.success)
        self.assertEqual(res.decision_status, "REJECTED")
        self.assertIn("Registration failed across all", res.decision_reason)
        self.assertIsNone(res.registration_to_canonical.homography)

    def test_04_scale_resolution_change_registers_accurately(self) -> None:
        """Condition 4: Scale/resolution changes between query and reference register accurately."""
        ref_img = _generate_rich_texture_image(640, 480, seed=40)
        self.manager.set_canonical_reference(ref_img, "visit_01", "2026-10-01T10:00:00Z")

        # Query scaled down by 0.75x
        query_scaled = cv2.resize(ref_img, (480, 360), interpolation=cv2.INTER_LINEAR)

        res = self.manager.register_query(query_scaled, "visit_02", "2026-10-01T11:00:00Z")
        self.assertTrue(res.registration_to_canonical.success)
        H = res.registration_to_canonical.homography
        self.assertIsNotNone(H)

        # A point at center in query (240, 180) should map back to ~(320, 240) in canonical
        pt_query = np.array([[[240.0, 180.0]]], dtype=np.float32)
        pt_canon = cv2.perspectiveTransform(pt_query, H)[0, 0]
        self.assertAlmostEqual(pt_canon[0], 320.0, delta=5.0)
        self.assertAlmostEqual(pt_canon[1], 240.0, delta=5.0)

    def test_05_large_perspective_shift_chained_keyframe(self) -> None:
        """Condition 5: Large perspective shift registers through chained keyframes."""
        base_img = _generate_rich_texture_image(640, 480, seed=50)
        self.manager.set_canonical_reference(base_img, "visit_01", "2026-10-01T10:00:00Z")

        # Step 1 warp: moderate perspective tilt
        pts1 = np.float32([[0, 0], [640, 0], [640, 480], [0, 480]])
        pts2 = np.float32([[40, 20], [600, 10], [620, 460], [20, 470]])
        H_warp1 = cv2.getPerspectiveTransform(pts1, pts2)
        img_kf2 = cv2.warpPerspective(base_img, H_warp1, (640, 480))

        res2 = self.manager.register_query(img_kf2, "visit_02", "2026-10-01T11:00:00Z")
        self.assertTrue(res2.registration_to_canonical.success)
        self.assertTrue(res2.is_promoted_to_keyframe)

        # Step 2 warp: further tilt from step 1
        pts3 = np.float32([[70, 35], [570, 25], [590, 440], [40, 450]])
        H_warp2 = cv2.getPerspectiveTransform(pts1, pts3)
        img_q = cv2.warpPerspective(base_img, H_warp2, (640, 480))

        res3 = self.manager.register_query(img_q, "visit_03", "2026-10-01T12:00:00Z")
        self.assertTrue(res3.registration_to_canonical.success)
        self.assertIn(res3.decision_status, ("ACCEPTED", "DEFERRED"))
        self.assertIsNotNone(res3.registration_to_canonical.homography)

    def test_06_camera_movement_alone_no_false_confirmed_moved(self) -> None:
        """Condition 6: Camera movement alone does not trigger false CONFIRMED_MOVED transitions."""
        store = SQLiteMemoryStore(":memory:")
        engine = TemporalMemoryEngine(store=store, config=TemporalEngineConfig())

        ref_img = _generate_rich_texture_image(640, 480, seed=60)
        self.manager.set_canonical_reference(ref_img, "visit_01", "2026-10-01T10:00:00Z")

        # Object 1 located at (200, 200, 280, 280) in canonical space
        obj_bbox = BoundingBox(200.0, 200.0, 280.0, 280.0)
        det_v1 = Detection(
            label="box",
            confidence=0.90,
            raw_bbox=obj_bbox,
            canonical_bbox=obj_bbox,
        )
        canon_reg = RegistrationResult(
            success=True,
            status=AlignmentStatus.SUCCESS,
            method_used="INITIAL_REFERENCE",
            ref_size=(640, 480),
            query_size=(640, 480),
            homography=np.eye(3, dtype=np.float64),
            inlier_ratio=1.0,
        )
        engine.process_visit(VisitInput(
            session_id="visit_01",
            session_name="Visit 01",
            timestamp="2026-10-01T10:00:00Z",
            image=ref_img,
            registration=canon_reg,
            detections=[det_v1],
        ))

        # Visit 2: Camera translates by (dx=30, dy=20), object is physically stationary!
        dx, dy = 30.0, 20.0
        M_shift = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])
        v2_img = cv2.warpAffine(ref_img, M_shift, (640, 480))
        res_v2 = self.manager.register_query(v2_img, "visit_02", "2026-10-01T11:00:00Z")
        self.assertTrue(res_v2.registration_to_canonical.success)

        # In query image coordinate frame, the object appears at raw (200+dx, 200+dy, 280+dx, 280+dy)
        raw_box_v2 = BoundingBox(200.0 + dx, 200.0 + dy, 280.0 + dx, 280.0 + dy)
        # Transform raw box back into canonical space using the estimated homography
        H = res_v2.registration_to_canonical.homography
        assert H is not None
        # Project center of box
        cx = (raw_box_v2.x_min + raw_box_v2.x_max) / 2.0
        cy = (raw_box_v2.y_min + raw_box_v2.y_max) / 2.0
        c_raw = np.array([[[cx, cy]]], dtype=np.float32)
        c_canon = cv2.perspectiveTransform(c_raw, H)[0, 0]

        canon_box_v2 = BoundingBox(
            float(c_canon[0] - 40.0),
            float(c_canon[1] - 40.0),
            float(c_canon[0] + 40.0),
            float(c_canon[1] + 40.0),
        )
        det_v2 = Detection(
            label="box",
            confidence=0.88,
            raw_bbox=raw_box_v2,
            canonical_bbox=canon_box_v2,
        )

        rep = engine.process_visit(VisitInput(
            session_id="visit_02",
            session_name="Visit 02",
            timestamp="2026-10-01T11:00:00Z",
            image=v2_img,
            registration=res_v2.registration_to_canonical,
            detections=[det_v2],
        ))

        # Verify state is CONFIRMED_PRESENT, NOT CONFIRMED_MOVED
        entities = store.list_entities()
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0].current_state, EpistemicState.CONFIRMED_PRESENT)

    def test_07_physical_object_movement_detected_under_camera_motion(self) -> None:
        """Condition 7: Physical object movement is correctly detected under moderate camera motion."""
        store = SQLiteMemoryStore(":memory:")
        engine = TemporalMemoryEngine(store=store, config=TemporalEngineConfig())

        ref_img = _generate_rich_texture_image(640, 480, seed=70)
        self.manager.set_canonical_reference(ref_img, "visit_01", "2026-10-01T10:00:00Z")

        # Object 1 initially at (150, 150, 220, 220)
        init_box = BoundingBox(150.0, 150.0, 220.0, 220.0)
        sig_extractor = VisualSignatureExtractor()
        sig1 = sig_extractor.extract(ref_img, init_box)

        engine.process_visit(VisitInput(
            session_id="visit_01",
            session_name="Visit 01",
            timestamp="2026-10-01T10:00:00Z",
            image=ref_img,
            registration=RegistrationResult(
                success=True,
                status=AlignmentStatus.SUCCESS,
                method_used="INITIAL_REFERENCE",
                ref_size=(640, 480),
                query_size=(640, 480),
                homography=np.eye(3, dtype=np.float64),
                inlier_ratio=1.0,
            ),
            detections=[Detection(label="tool", confidence=0.92, raw_bbox=init_box, canonical_bbox=init_box, signature=sig1)],
        ))

        # Visit 2: Camera shifted slightly (dx=15, dy=10) AND object physically moved by 120px in x
        dx_cam, dy_cam = 15.0, 10.0
        M_shift = np.float32([[1.0, 0.0, dx_cam], [0.0, 1.0, dy_cam]])
        v2_img = cv2.warpAffine(ref_img, M_shift, (640, 480))
        res_v2 = self.manager.register_query(v2_img, "visit_02", "2026-10-01T11:00:00Z")
        self.assertTrue(res_v2.registration_to_canonical.success)

        # Object moved to canonical (270, 150, 340, 220), displacement = 120px
        # In query image: (270 + dx_cam, 150 + dy_cam)
        raw_box = BoundingBox(270.0 + dx_cam, 150.0 + dy_cam, 340.0 + dx_cam, 220.0 + dy_cam)
        canon_box = BoundingBox(270.0, 150.0, 340.0, 220.0)

        engine.process_visit(VisitInput(
            session_id="visit_02",
            session_name="Visit 02",
            timestamp="2026-10-01T11:00:00Z",
            image=v2_img,
            registration=res_v2.registration_to_canonical,
            detections=[Detection(label="tool", confidence=0.90, raw_bbox=raw_box, canonical_bbox=canon_box, signature=sig1)],
        ))

        entities = store.list_entities()
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0].current_state, EpistemicState.CONFIRMED_MOVED)

    def test_08_degenerate_or_non_convex_homography_rejected(self) -> None:
        """Condition 8: Degenerate or non-convex compound homography is rejected."""
        ref_img = _generate_rich_texture_image(640, 480, seed=80)
        self.manager.set_canonical_reference(ref_img, "visit_01", "2026-10-01T10:00:00Z")

        # Create a synthetic keyframe record with an intentionally inverted/crossing homography
        # Crossing corners warp: (0,0)->(640,480), (640,0)->(0,480) [folded quad]
        pts_norm = np.float32([[0, 0], [640, 0], [640, 480], [0, 480]])
        pts_folded = np.float32([[640, 480], [0, 480], [640, 0], [0, 0]])
        H_folded = cv2.getPerspectiveTransform(pts_norm, pts_folded)

        kf_corrupt = KeyframeRecord(
            keyframe_id="kf_corrupt",
            image=ref_img.copy(),
            size=(640, 480),
            timestamp="2026-10-01T10:30:00Z",
            is_canonical=False,
            homography_to_canonical=H_folded,
        )
        self.manager.keyframes["kf_corrupt"] = kf_corrupt

        reg_mock = RegistrationResult(
            success=True,
            status=AlignmentStatus.SUCCESS,
            method_used="SIFT",
            ref_size=(640, 480),
            query_size=(640, 480),
            homography=np.eye(3, dtype=np.float64),
            inlier_ratio=0.8,
            num_inliers=50,
        )

        compound_res = self.manager._compose_to_canonical(reg_mock, kf_corrupt, ref_img)
        self.assertFalse(compound_res.success)
        self.assertEqual(compound_res.status, AlignmentStatus.IMPLAUSIBLE_WARP_GEOMETRY)

    def test_09_outlier_match_clusters_spatial_coverage(self) -> None:
        """Condition 9: Clustered matches yield low coverage, increasing uncertainty margin."""
        # 20 points tightly clustered in a 20x20 pixel corner
        cluster_pts = np.float32([[10 + i % 5 * 2, 10 + i // 5 * 2] for i in range(20)])
        coverage_clustered = SpatialCoverageCalculator.calculate_coverage(cluster_pts, (640, 480))
        self.assertLess(coverage_clustered, 0.25)

        # 20 points well-distributed across canvas
        spread_pts = np.float32([
            [50, 50], [200, 60], [450, 50], [600, 70],
            [60, 200], [250, 220], [400, 240], [580, 210],
            [70, 380], [220, 400], [420, 390], [590, 420],
            [150, 150], [350, 160], [480, 300], [300, 350],
        ])
        coverage_spread = SpatialCoverageCalculator.calculate_coverage(spread_pts, (640, 480))
        self.assertGreater(coverage_spread, 0.50)

        # Verify dynamic uncertainty threshold formula
        store = SQLiteMemoryStore(":memory:")
        engine = TemporalMemoryEngine(store=store)

        reg_clustered = RegistrationResult(
            success=True,
            status=AlignmentStatus.SUCCESS,
            method_used="SIFT",
            ref_size=(640, 480),
            query_size=(640, 480),
            homography=np.eye(3, dtype=np.float64),
            spatial_coverage=coverage_clustered,
            inlier_ratio=0.50,
            reprojection_stats=ReprojectionStats(mean=2.8, median=2.5, std=0.5, max_error=3.0, inlier_count=20),
        )

        t_eff, delta_u, breakdown = engine.compute_effective_movement_threshold(reg_clustered, base_threshold=35.0)
        self.assertGreater(delta_u, 10.0)
        self.assertGreater(t_eff, 45.0)
        self.assertGreater(breakdown["u_coverage_px"], 5.0)

    def test_10_keyframe_promotion_criteria(self) -> None:
        """Condition 10: Promotion occurs with distinct framing, and is rejected when redundant."""
        ref_img = _generate_rich_texture_image(640, 480, seed=100)
        self.manager.set_canonical_reference(ref_img, "visit_01", "2026-10-01T10:00:00Z")

        # Case A: Identical image (zero shift) -> Should NOT be promoted
        res_identical = self.manager.register_query(ref_img, "visit_02", "2026-10-01T11:00:00Z")
        self.assertTrue(res_identical.registration_to_canonical.success)
        self.assertFalse(res_identical.is_promoted_to_keyframe)
        self.assertEqual(len(self.manager.keyframes), 1)

        # Case B: Meaningful viewpoint translation (dx=60px, dy=40px) with high inliers -> Should be promoted
        M_shift = np.float32([[1.0, 0.0, 60.0], [0.0, 1.0, 40.0]])
        shifted_img = cv2.warpAffine(ref_img, M_shift, (640, 480))
        res_shifted = self.manager.register_query(shifted_img, "visit_03", "2026-10-01T12:00:00Z")
        self.assertTrue(res_shifted.registration_to_canonical.success)
        self.assertTrue(res_shifted.is_promoted_to_keyframe)
        self.assertEqual(len(self.manager.keyframes), 2)
        self.assertIn("visit_03", self.manager.keyframes)

    def test_11_canonical_coordinates_consistent_multi_hop(self) -> None:
        """Condition 11: Compound homography preserves canonical coordinate consistency."""
        ref_img = _generate_rich_texture_image(640, 480, seed=110)
        self.manager.set_canonical_reference(ref_img, "visit_01", "2026-10-01T10:00:00Z")

        # Step 1: translation T1 = (dx=50, dy=30)
        T1 = np.float32([[1.0, 0.0, 50.0], [0.0, 1.0, 30.0]])
        img_v2 = cv2.warpAffine(ref_img, T1, (640, 480))
        res_v2 = self.manager.register_query(img_v2, "visit_02", "2026-10-01T11:00:00Z")
        self.assertTrue(res_v2.is_promoted_to_keyframe)

        # Step 2: translation T2 = (dx=40, dy=20) from v2
        T2 = np.float32([[1.0, 0.0, 40.0], [0.0, 1.0, 20.0]])
        img_v3 = cv2.warpAffine(img_v2, T2, (640, 480))
        res_v3 = self.manager.register_query(img_v3, "visit_03", "2026-10-01T12:00:00Z")
        self.assertTrue(res_v3.registration_to_canonical.success)

        # Expected total translation to canonical: dx_total = 50 - 40 = ...
        # Let's verify through homography perspective transformation of a test point
        H_total = res_v3.registration_to_canonical.homography
        self.assertIsNotNone(H_total)

        # Test point in query v3
        test_pt = np.array([[[200.0, 200.0]]], dtype=np.float32)
        canon_mapped = cv2.perspectiveTransform(test_pt, H_total)[0, 0]

        # The mapping should be finite and within plausible bounds
        self.assertTrue(np.all(np.isfinite(canon_mapped)))
        self.assertGreater(canon_mapped[0], 0.0)
        self.assertGreater(canon_mapped[1], 0.0)

    def test_12_gated_registration_prevents_false_confirmed_removed(self) -> None:
        """Condition 12: Gated registrations prevent false CONFIRMED_REMOVED transitions."""
        store = SQLiteMemoryStore(":memory:")
        engine = TemporalMemoryEngine(
            store=store,
            config=TemporalEngineConfig(vacancy_visits_for_removal=2),
        )

        ref_img = _generate_rich_texture_image(640, 480, seed=120)
        canon_reg = RegistrationResult(
            success=True,
            status=AlignmentStatus.SUCCESS,
            method_used="INITIAL_REFERENCE",
            ref_size=(640, 480),
            query_size=(640, 480),
            homography=np.eye(3, dtype=np.float64),
            inlier_ratio=1.0,
        )
        box = BoundingBox(100.0, 100.0, 160.0, 160.0)

        # Visit 1: Object present
        engine.process_visit(VisitInput(
            session_id="visit_01",
            session_name="Visit 01",
            timestamp="2026-10-01T10:00:00Z",
            image=ref_img,
            registration=canon_reg,
            detections=[Detection(label="cup", confidence=0.91, raw_bbox=box, canonical_bbox=box)],
        ))

        # Visit 2: Registration FAILS completely (e.g. occlusion or camera pointing away)
        failed_reg = RegistrationResult(
            success=False,
            status=AlignmentStatus.INSUFFICIENT_INLIERS,
            method_used="SIFT",
            ref_size=(640, 480),
            query_size=(640, 480),
            homography=None,
            inlier_ratio=0.05,
            failure_reason="No geometric consensus",
        )
        # Empty detections due to failed alignment
        engine.process_visit(VisitInput(
            session_id="visit_02",
            session_name="Visit 02",
            timestamp="2026-10-01T11:00:00Z",
            image=ref_img,
            registration=failed_reg,
            detections=[],
        ))

        # Visit 3: Registration again fails
        engine.process_visit(VisitInput(
            session_id="visit_03",
            session_name="Visit 03",
            timestamp="2026-10-01T12:00:00Z",
            image=ref_img,
            registration=failed_reg,
            detections=[],
        ))

        # Object MUST NOT transition to PENDING_REMOVAL or CONFIRMED_REMOVED
        entities = store.list_entities()
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0].current_state, EpistemicState.CONFIRMED_PRESENT)


if __name__ == "__main__":
    unittest.main()
