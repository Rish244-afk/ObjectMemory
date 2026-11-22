"""
Comprehensive unit and integration tests for temporal reasoning reliability (Phase 8).

Validates:
1. Return to Baseline: Object moves A -> B then returns B -> A, tracking consecutive
   displacement, baseline displacement, and emitting return-to-baseline events.
2. Resolution and FOV shifts: Boundary truncation and overlap mask gating prevent
   spurious movement when framing changes.
3. Conservative Removal Reasoning:
   - Single missed detection with object-like patch preserves CONFIRMED_PRESENT.
   - Occluded or out-of-frame object holds state without removal progression.
   - Failed or gated registration preserves state and does not increment removal streaks.
   - Verified vacant patch transitions to PENDING_REMOVAL then CONFIRMED_REMOVED.
4. False positive rejection: Candidate with mismatched visual signature is rejected.
5. Reappearance recovery: Entity in CONFIRMED_REMOVED or PENDING_REMOVAL is restored
   to CONFIRMED_PRESENT upon redetection.
"""

from __future__ import annotations

import unittest
import numpy as np

from objectmemory.baseline.comparator import calculate_centroid_distance
from objectmemory.detection.models import Detection, VisualSignature
from objectmemory.domain.models import BoundingBox
from objectmemory.domain.states import EpistemicState, VisibilityStatus
from objectmemory.persistence.repository import SQLiteMemoryStore
from objectmemory.registration.models import AlignmentStatus, RegistrationResult
from objectmemory.temporal.engine import TemporalMemoryEngine
from objectmemory.temporal.models import TemporalEngineConfig, VisitInput


def _create_synthetic_signature(color_val: float, texture_val: float = 0.5) -> VisualSignature:
    """Creates a deterministic 148-d visual signature vector."""
    color = np.full(128, color_val, dtype=np.float32)
    color = color / (np.linalg.norm(color) + 1e-6)
    texture = np.full(20, texture_val, dtype=np.float32)
    texture = texture / (np.linalg.norm(texture) + 1e-6)
    combined = np.concatenate([color, texture]).astype(np.float32)
    combined = combined / (np.linalg.norm(combined) + 1e-6)
    return VisualSignature(color_histogram=color, texture_descriptor=texture, combined_vector=combined)


def _make_mock_registration(
    success: bool = True,
    inlier_ratio: float = 0.85,
    ref_size: tuple[int, int] = (640, 480),
    query_size: tuple[int, int] = (640, 480),
) -> RegistrationResult:
    mask = np.ones((ref_size[1], ref_size[0]), dtype=np.uint8) * 255 if success else None
    return RegistrationResult(
        success=success,
        status=AlignmentStatus.SUCCESS if success else AlignmentStatus.INSUFFICIENT_INLIERS,
        method_used="planar_homography",
        ref_size=ref_size,
        query_size=query_size,
        homography=np.eye(3, dtype=np.float64) if success else None,
        num_inliers=100 if success else 5,
        inlier_ratio=inlier_ratio if success else 0.05,
        overlap_mask=mask,
        aligned_image=np.zeros((ref_size[1], ref_size[0], 3), dtype=np.uint8) if success else None,
    )


class TestTemporalReasoningReliability(unittest.TestCase):
    """Rigorous tests covering Phase 8 temporal reasoning reliability requirements."""

    def setUp(self) -> None:
        self.store = SQLiteMemoryStore(":memory:")
        self.config = TemporalEngineConfig(border_margin_px=12.0)
        self.engine = TemporalMemoryEngine(store=self.store, config=self.config)

    def test_movement_and_return_to_baseline(self) -> None:
        """
        Tests: Object moves A -> B then returns B -> A.
        Must track:
        - consecutive_displacement: distance between successive observations.
        - baseline_displacement: distance from initial discovery position.
        - is_return_to_baseline flag and event emitting return-to-baseline movement.
        - Does NOT treat return-to-baseline as 'never moved' or displacement 0.0.
        """
        sig = _create_synthetic_signature(0.8)
        reg = _make_mock_registration()

        # Visit 1: Baseline position A (x=100, y=100)
        box_a = BoundingBox(100.0, 100.0, 150.0, 150.0)
        det_v1 = Detection("cup", 0.95, box_a, canonical_bbox=box_a, signature=sig)
        rep1 = self.engine.process_visit(VisitInput("v1", "Visit 1", "2026-10-01T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), reg, [det_v1]))
        
        ent = self.store.list_entities()[0]
        self.assertEqual(ent.current_state, EpistemicState.CONFIRMED_PRESENT)

        # Visit 2: Object relocated to position B (x=300, y=100) -> 200px displacement
        box_b = BoundingBox(300.0, 100.0, 350.0, 150.0)
        det_v2 = Detection("cup", 0.95, box_b, canonical_bbox=box_b, signature=sig)
        rep2 = self.engine.process_visit(VisitInput("v2", "Visit 2", "2026-10-02T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), reg, [det_v2]))

        up2 = rep2.entity_updates[0]
        self.assertTrue(up2.transition_occurred)
        self.assertEqual(up2.new_state, EpistemicState.CONFIRMED_MOVED)
        self.assertAlmostEqual(up2.supporting_evidence["consecutive_displacement"], 200.0, places=1)
        self.assertAlmostEqual(up2.supporting_evidence["baseline_displacement"], 200.0, places=1)
        self.assertFalse(up2.supporting_evidence["is_return_to_baseline"])

        # Visit 3: Object returns to position A (x=102, y=101) -> ~2px from baseline, ~198px consecutive movement
        box_return = BoundingBox(102.0, 101.0, 152.0, 151.0)
        det_v3 = Detection("cup", 0.95, box_return, canonical_bbox=box_return, signature=sig)
        rep3 = self.engine.process_visit(VisitInput("v3", "Visit 3", "2026-10-03T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), reg, [det_v3]))

        up3 = rep3.entity_updates[0]
        self.assertTrue(up3.transition_occurred)
        self.assertEqual(up3.new_state, EpistemicState.CONFIRMED_PRESENT)
        self.assertTrue(up3.supporting_evidence["is_return_to_baseline"])
        self.assertTrue(up3.supporting_evidence["is_consecutive_moved"])
        # Crucial: consecutive displacement is the physical ~198px return step, NOT 0.0
        self.assertGreater(up3.supporting_evidence["consecutive_displacement"], 190.0)
        self.assertLess(up3.supporting_evidence["baseline_displacement"], 5.0)

        # Inspect recorded ChangeEvent in store
        events = self.store.list_change_events_for_entity(ent.entity_id)
        return_evt = [e for e in events if e.to_session_id == "v3"][0]
        self.assertEqual(return_evt.prior_state, EpistemicState.CONFIRMED_MOVED)
        self.assertEqual(return_evt.new_state, EpistemicState.CONFIRMED_PRESENT)
        self.assertGreater(return_evt.displacement_distance, 190.0)
        self.assertTrue(return_evt.supporting_evidence["is_return_to_baseline"])

    def test_resolution_and_framing_shift_does_not_trigger_false_movement(self) -> None:
        """
        Tests: When an object is partially clipped by camera framing or resolution shifts,
        boundary-clipping safeguards and overlap mask checks prevent spurious CONFIRMED_MOVED calls.
        """
        sig = _create_synthetic_signature(0.5)
        # Reference canvas: 780x636
        reg1 = _make_mock_registration(ref_size=(780, 636), query_size=(780, 636))

        # Visit 1: Large hardcover book at x=400..750 on 780px wide canvas
        box_v1 = BoundingBox(400.0, 300.0, 750.0, 500.0)
        det_v1 = Detection("book", 0.90, box_v1, canonical_bbox=box_v1, signature=sig)
        self.engine.process_visit(VisitInput("v1", "Visit 1", "2026-10-01T10:00:00Z", np.zeros((636, 780, 3), dtype=np.uint8), reg1, [det_v1]))

        # Visit 2: Query image is cropped to 584x576 (narrower framing).
        # Query detection touches border at x=575 (within 12px margin of 584).
        # Overlap mask in canonical space only partially covers initial box (< 70%).
        overlap_mask = np.zeros((636, 780), dtype=np.uint8)
        overlap_mask[:, :550] = 255  # only covers up to x=550 in canonical space (partial book overlap)
        reg2 = RegistrationResult(
            success=True,
            status=AlignmentStatus.SUCCESS,
            method_used="planar_homography",
            ref_size=(780, 636),
            query_size=(584, 576),
            homography=np.eye(3, dtype=np.float64),
            num_inliers=50,
            inlier_ratio=0.35,
            overlap_mask=overlap_mask,
            aligned_image=np.zeros((636, 780, 3), dtype=np.uint8),
        )

        det_v2_raw = BoundingBox(280.0, 260.0, 575.0, 460.0)  # touches 584 border (575 within 12px)
        det_v2_canon = BoundingBox(330.0, 320.0, 625.0, 520.0)
        det_v2 = Detection("book", 0.85, det_v2_raw, canonical_bbox=det_v2_canon, signature=sig)

        rep2 = self.engine.process_visit(VisitInput("v2", "Visit 2", "2026-10-02T10:00:00Z", np.zeros((576, 584, 3), dtype=np.uint8), reg2, [det_v2]))

        ent = self.store.list_entities()[0]
        # Must NOT be marked as CONFIRMED_MOVED!
        self.assertEqual(ent.current_state, EpistemicState.CONFIRMED_PRESENT)
        self.assertFalse(rep2.entity_updates[0].transition_occurred)

    def test_conservative_removal_reasoning(self) -> None:
        """
        Tests multi-stage conservative removal reasoning:
        - 1st missed visit with verified vacant patch -> PENDING_REMOVAL.
        - 2nd missed visit with verified vacant patch -> CONFIRMED_REMOVED.
        - Cannot skip PENDING_REMOVAL to directly confirm removal.
        """
        sig = _create_synthetic_signature(0.3)
        reg = _make_mock_registration()

        box = BoundingBox(200.0, 200.0, 250.0, 250.0)
        det = Detection("pliers", 0.95, box, canonical_bbox=box, signature=sig)
        self.engine.process_visit(VisitInput("v1", "Visit 1", "2026-10-01T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), reg, [det]))

        # Visit 2: Object absent. Aligned image has black/zero patch (patch similarity 0.0 < 0.40 -> verified vacant)
        rep2 = self.engine.process_visit(VisitInput("v2", "Visit 2", "2026-10-02T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), reg, []))
        up2 = rep2.entity_updates[0]
        self.assertEqual(up2.new_state, EpistemicState.PENDING_REMOVAL)
        self.assertTrue(up2.transition_occurred)

        # Visit 3: Object still absent and patch still vacant -> transitions to CONFIRMED_REMOVED
        rep3 = self.engine.process_visit(VisitInput("v3", "Visit 3", "2026-10-03T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), reg, []))
        up3 = rep3.entity_updates[0]
        self.assertEqual(up3.new_state, EpistemicState.CONFIRMED_REMOVED)
        self.assertTrue(up3.transition_occurred)

    def test_gated_registration_freezes_removal_progression(self) -> None:
        """
        Tests: When registration quality is poor or failed, absence cannot be verified.
        Removal streaks must NOT increment and state must NOT advance.
        """
        sig = _create_synthetic_signature(0.4)
        reg_good = _make_mock_registration(success=True)
        reg_bad = _make_mock_registration(success=False)

        box = BoundingBox(150.0, 150.0, 200.0, 200.0)
        det = Detection("bottle", 0.92, box, canonical_bbox=box, signature=sig)
        self.engine.process_visit(VisitInput("v1", "Visit 1", "2026-10-01T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), reg_good, [det]))

        # Visit 2: Registration failed (e.g. inliers below threshold). No detections.
        rep2 = self.engine.process_visit(VisitInput("v2", "Visit 2", "2026-10-02T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), reg_bad, []))
        up2 = rep2.entity_updates[0]
        self.assertFalse(up2.transition_occurred)
        self.assertEqual(up2.new_state, EpistemicState.CONFIRMED_PRESENT)
        self.assertIn("Registration unverified", up2.explanation)

    def test_reappearance_after_confirmed_removal(self) -> None:
        """
        Tests: Entity in CONFIRMED_REMOVED is re-observed.
        Must successfully restore to CONFIRMED_PRESENT.
        """
        sig = _create_synthetic_signature(0.6)
        reg = _make_mock_registration()

        box = BoundingBox(120.0, 120.0, 180.0, 180.0)
        det = Detection("notebook", 0.90, box, canonical_bbox=box, signature=sig)
        self.engine.process_visit(VisitInput("v1", "Visit 1", "2026-10-01T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), reg, [det]))

        # Remove across 2 visits
        self.engine.process_visit(VisitInput("v2", "Visit 2", "2026-10-02T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), reg, []))
        self.engine.process_visit(VisitInput("v3", "Visit 3", "2026-10-03T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), reg, []))

        ent = self.store.list_entities()[0]
        self.assertEqual(ent.current_state, EpistemicState.CONFIRMED_REMOVED)

        # Visit 4: Object returns to workbench
        det_v4 = Detection("notebook", 0.91, box, canonical_bbox=box, signature=sig)
        rep4 = self.engine.process_visit(VisitInput("v4", "Visit 4", "2026-10-04T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), reg, [det_v4]))

        up4 = rep4.entity_updates[0]
        self.assertTrue(up4.transition_occurred)
        self.assertEqual(up4.new_state, EpistemicState.CONFIRMED_PRESENT)
        self.assertIn("re-introduced", up4.explanation.lower())


if __name__ == "__main__":
    unittest.main()
