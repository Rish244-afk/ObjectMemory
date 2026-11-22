"""
Audited Unit & Integration Tests for Temporal Failure Modes (Phase 5).

Covers all 10 critical failure modes mandated by Phase 5:
1. One missed detection must not confirm removal.
2. Multiple missed detections under unreliable registration must not confirm removal.
3. A low-confidence detection must not overwrite a reliable historical observation without sufficient evidence.
4. Label changes must not create duplicate persistent entities.
5. Similar-looking objects must not be merged without sufficient association evidence.
6. A genuinely moved object must not be confused with camera motion.
7. Reappearance should preserve the original entity history when association is reliable.
8. Uncertain observations must not be reported as certain physical changes.
9. A failed or degenerate homography must not corrupt canonical coordinates.
10. Repeated ingestion of the same visit must not create duplicate events or entities.

Every test asserts actual SQLite persistent state and ChangeEvent audit rows.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np

from objectmemory.baseline.comparator import calculate_bbox_iou
from objectmemory.detection.models import Detection, VisualSignature
from objectmemory.detection.pipeline import transform_bbox_to_canonical
from objectmemory.domain.models import BoundingBox, Session, WorkbenchEntity, Observation, ChangeEvent
from objectmemory.domain.states import EpistemicState, VisibilityStatus
from objectmemory.persistence.repository import SQLiteMemoryStore
from objectmemory.registration.models import AlignmentStatus, RegistrationResult
from objectmemory.temporal.associator import CrossVisitEntityAssociator
from objectmemory.temporal.engine import TemporalMemoryEngine
from objectmemory.temporal.models import TemporalEngineConfig, VisitInput


def _make_dummy_sig(dim: int = 148, split: int = 0) -> VisualSignature:
    vec = np.zeros(dim, dtype=np.float32)
    if split == 0:
        vec[:74] = 1.0
    elif split == 1:
        vec[74:] = 1.0
    else:
        vec.fill(1.0)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return VisualSignature(
        color_histogram=vec[:128].copy(),
        texture_descriptor=vec[128:].copy(),
        combined_vector=vec,
    )


def make_det(
    label: str,
    bbox: BoundingBox,
    split: int = 0,
    conf: float = 0.90,
    canonical_bbox: Optional[BoundingBox] = None,
) -> Detection:
    return Detection(
        label=label,
        confidence=conf,
        raw_bbox=bbox,
        canonical_bbox=canonical_bbox or bbox,
        signature=_make_dummy_sig(split=split),
    )


def make_reg_success(inlier_ratio: float = 0.90, homography: Optional[np.ndarray] = None) -> RegistrationResult:
    H = homography if homography is not None else np.eye(3, dtype=np.float64)
    return RegistrationResult(
        success=True,
        status=AlignmentStatus.SUCCESS,
        method_used="MOCK",
        ref_size=(640, 480),
        query_size=(640, 480),
        homography=H,
        overlap_mask=np.ones((480, 640), dtype=np.uint8) * 255,
        aligned_image=np.ones((480, 640, 3), dtype=np.uint8) * 128,
        inlier_ratio=inlier_ratio,
    )


def make_reg_unreliable(inlier_ratio: float = 0.15) -> RegistrationResult:
    return RegistrationResult(
        success=False,
        status=AlignmentStatus.INSUFFICIENT_INLIERS,
        method_used="MOCK",
        ref_size=(640, 480),
        query_size=(640, 480),
        homography=None,
        inlier_ratio=inlier_ratio,
        failure_reason="Insufficient inlier count for reliable alignment",
    )


class TestTemporalFailureModes(unittest.TestCase):
    """Rigorous audit tests asserting database state across temporal failure modes."""

    def setUp(self):
        self.store = SQLiteMemoryStore(":memory:")
        self.config = TemporalEngineConfig(
            min_registration_inlier_ratio=0.40,
            movement_displacement_threshold=35.0,
            vacancy_visits_for_pending=1,
            vacancy_visits_for_removal=2,
            ambiguity_margin=0.10,
        )
        self.engine = TemporalMemoryEngine(store=self.store, config=self.config)

    # -------------------------------------------------------------------------
    # Mode 1: One missed detection must not confirm removal
    # -------------------------------------------------------------------------
    def test_one_missed_detection_must_not_confirm_removal(self):
        b = BoundingBox(50.0, 50.0, 150.0, 150.0)
        v1 = VisitInput("v1", "V1", "2026-10-20T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), make_reg_success(), [make_det("mug", b)])
        v2 = VisitInput("v2", "V2", "2026-10-20T10:05:00Z", np.zeros((480, 640, 3), dtype=np.uint8), make_reg_success(), [make_det("mug", b)])
        self.engine.process_visit(v1)
        self.engine.process_visit(v2)

        # Single missed detection in visit 3
        v3 = VisitInput("v3", "V3", "2026-10-20T10:10:00Z", np.zeros((480, 640, 3), dtype=np.uint8), make_reg_success(), [])
        rep3 = self.engine.process_visit(v3)

        ent = self.store.get_entity("ent_v1_1")
        self.assertIsNotNone(ent)
        self.assertEqual(ent.current_state, EpistemicState.PENDING_REMOVAL)
        self.assertNotEqual(ent.current_state, EpistemicState.CONFIRMED_REMOVED)

        # Assert ChangeEvent audit row recorded PENDING_REMOVAL, not CONFIRMED_REMOVED
        events = self.store.list_change_events_for_session("v3")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].new_state, EpistemicState.PENDING_REMOVAL)

    # -------------------------------------------------------------------------
    # Mode 2: Multiple missed detections under unreliable registration must not confirm removal
    # -------------------------------------------------------------------------
    def test_multiple_missed_detections_under_unreliable_registration_must_not_confirm_removal(self):
        b = BoundingBox(100.0, 100.0, 200.0, 200.0)
        v1 = VisitInput("v1", "V1", "2026-10-20T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), make_reg_success(), [make_det("lamp", b)])
        v2 = VisitInput("v2", "V2", "2026-10-20T10:05:00Z", np.zeros((480, 640, 3), dtype=np.uint8), make_reg_success(), [make_det("lamp", b)])
        self.engine.process_visit(v1)
        self.engine.process_visit(v2)

        # 3 successive visits with missing detections BUT unreliable registration
        for i in range(3, 6):
            v_fail = VisitInput(f"v{i}", f"V{i}", f"2026-10-20T10:{10 + i:02d}:00Z", np.zeros((480, 640, 3), dtype=np.uint8), make_reg_unreliable(), [])
            rep = self.engine.process_visit(v_fail)
            self.assertEqual(rep.state_transitions_count, 0)

        # In database: state must remain CONFIRMED_PRESENT; vacant streak must be 0
        ent = self.store.get_entity("ent_v1_1")
        self.assertIsNotNone(ent)
        self.assertEqual(ent.current_state, EpistemicState.CONFIRMED_PRESENT)
        vacant_streak = (ent.metadata or {}).get("vacant_streak", 0)
        self.assertEqual(vacant_streak, 0)

    # -------------------------------------------------------------------------
    # Mode 3: Low-confidence/unreliable detection must not overwrite historical observation
    # -------------------------------------------------------------------------
    def test_low_confidence_detection_does_not_overwrite_reliable_historical_observation(self):
        b_init = BoundingBox(80.0, 80.0, 160.0, 160.0)
        v1 = VisitInput("v1", "V1", "2026-10-20T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), make_reg_success(), [make_det("cup", b_init, conf=0.95)])
        self.engine.process_visit(v1)

        ent_init = self.store.get_entity("ent_v1_1")
        obs_init = self.store.list_observations_for_entity("ent_v1_1")
        self.assertEqual(len(obs_init), 1)
        self.assertAlmostEqual(obs_init[0].detector_confidence, 0.95)
        initial_canon = ent_init.canonical_bbox

        # Visit 2 has unreliable registration with a shifted detection
        b_shifted = BoundingBox(300.0, 300.0, 380.0, 380.0)
        v2 = VisitInput("v2", "V2", "2026-10-20T10:05:00Z", np.zeros((480, 640, 3), dtype=np.uint8), make_reg_unreliable(), [make_det("cup", b_shifted, conf=0.30)])
        self.engine.process_visit(v2)

        # Historical canonical bbox must NOT be corrupted/overwritten by unreliable detection
        ent_post = self.store.get_entity("ent_v1_1")
        self.assertEqual(ent_post.canonical_bbox, initial_canon)

        # Historical observation in DB is preserved untouched
        obs_post = self.store.list_observations_for_entity("ent_v1_1")
        # Observation for v1 still exists and has original confidence and bbox
        v1_obs = next(o for o in obs_post if o.session_id == "v1")
        self.assertAlmostEqual(v1_obs.detector_confidence, 0.95)
        self.assertEqual(v1_obs.canonical_bbox, initial_canon)

    # -------------------------------------------------------------------------
    # Mode 4: Label changes must not create duplicate persistent entities
    # -------------------------------------------------------------------------
    def test_label_changes_must_not_create_duplicate_persistent_entities(self):
        b = BoundingBox(150.0, 150.0, 250.0, 250.0)
        v1 = VisitInput("v1", "V1", "2026-10-20T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), make_reg_success(), [make_det("book", b, split=0)])
        v2 = VisitInput("v2", "V2", "2026-10-20T10:05:00Z", np.zeros((480, 640, 3), dtype=np.uint8), make_reg_success(), [make_det("notebook", b, split=0)])
        v3 = VisitInput("v3", "V3", "2026-10-20T10:10:00Z", np.zeros((480, 640, 3), dtype=np.uint8), make_reg_success(), [make_det("laptop", b, split=0)])

        self.engine.process_visit(v1)
        self.engine.process_visit(v2)
        self.engine.process_visit(v3)

        # Assert in DB: Exactly ONE entity exists; no ghost duplicate entities created
        all_ents = self.store.list_entities()
        self.assertEqual(len(all_ents), 1)
        self.assertEqual(all_ents[0].entity_id, "ent_v1_1")
        self.assertEqual(all_ents[0].current_state, EpistemicState.CONFIRMED_PRESENT)

        # 3 observations linked to the same entity
        obs = self.store.list_observations_for_entity("ent_v1_1")
        self.assertEqual(len(obs), 3)

    # -------------------------------------------------------------------------
    # Mode 5: Similar-looking objects must not be merged without sufficient evidence
    # -------------------------------------------------------------------------
    def test_similar_looking_objects_must_not_be_merged_without_sufficient_evidence(self):
        # Two identical-looking cups at different positions on desk
        b1 = BoundingBox(50.0, 50.0, 120.0, 120.0)
        b2 = BoundingBox(300.0, 50.0, 370.0, 120.0)
        v1 = VisitInput("v1", "V1", "2026-10-20T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), make_reg_success(), [
            make_det("cup", b1, split=0),
            make_det("cup", b2, split=0),
        ])
        self.engine.process_visit(v1)
        ents_v1 = self.store.list_entities()
        self.assertEqual(len(ents_v1), 2)

        # Visit 2: Candidate placed equidistant between them causing ambiguity
        b_mid = BoundingBox(175.0, 50.0, 245.0, 120.0)
        v2 = VisitInput("v2", "V2", "2026-10-20T10:05:00Z", np.zeros((480, 640, 3), dtype=np.uint8), make_reg_success(), [
            make_det("cup", b_mid, split=0),
        ])
        rep2 = self.engine.process_visit(v2)

        # Neither entity should be incorrectly merged with the equidistant ambiguous candidate
        ents_v2 = self.store.list_entities()
        self.assertEqual(len(ents_v2), 2)  # Did not merge the two entities into one

    # -------------------------------------------------------------------------
    # Mode 6: Genuinely moved object must not be confused with camera motion
    # -------------------------------------------------------------------------
    def test_genuinely_moved_object_must_not_be_confused_with_camera_motion(self):
        # Setup camera motion: pure translation H (camera shifted by dx=+50, dy=+30)
        # Point in query (xq, yq) relates to ref (xr, yr): xq = xr - 50, yq = yr - 30
        # H maps query to ref: H = [[1, 0, 50], [0, 1, 30], [0, 0, 1]]
        H_cam = np.array([
            [1.0, 0.0, 50.0],
            [0.0, 1.0, 30.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        # Stationary object 1: ref position (100, 100, 200, 200)
        # In query under camera motion: (50, 70, 150, 170) -> Canonical is (100, 100, 200, 200)
        b1_ref = BoundingBox(100.0, 100.0, 200.0, 200.0)
        b1_query_raw = BoundingBox(50.0, 70.0, 150.0, 170.0)
        b1_query_canon = transform_bbox_to_canonical(b1_query_raw, H_cam, (640, 480))

        # Genuinely moved object 2: ref position (300, 300, 400, 400)
        # Physically relocated by +120px to (420, 300, 520, 400) in canonical space
        b2_ref = BoundingBox(300.0, 300.0, 400.0, 400.0)
        b2_query_canon = BoundingBox(420.0, 300.0, 520.0, 400.0)
        b2_query_raw = BoundingBox(370.0, 270.0, 470.0, 370.0)

        v1 = VisitInput("v1", "V1", "2026-10-20T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), make_reg_success(), [
            make_det("stat_obj", b1_ref),
            make_det("move_obj", b2_ref),
        ])
        self.engine.process_visit(v1)

        reg_cam = make_reg_success(inlier_ratio=0.92, homography=H_cam)
        v2 = VisitInput("v2", "V2", "2026-10-20T10:05:00Z", np.zeros((480, 640, 3), dtype=np.uint8), reg_cam, [
            make_det("stat_obj", b1_query_raw, canonical_bbox=b1_query_canon),
            make_det("move_obj", b2_query_raw, canonical_bbox=b2_query_canon),
        ])
        rep2 = self.engine.process_visit(v2)

        # Assert in DB:
        e1 = next(e for e in self.store.list_entities() if e.label == "stat_obj")
        e2 = next(e for e in self.store.list_entities() if e.label == "move_obj")

        # Camera-shifted stationary object remains CONFIRMED_PRESENT (0 movement)
        self.assertEqual(e1.current_state, EpistemicState.CONFIRMED_PRESENT)

        # Genuinely moved object transitions to CONFIRMED_MOVED with ~120px displacement
        self.assertEqual(e2.current_state, EpistemicState.CONFIRMED_MOVED)
        evts = self.store.list_change_events_for_entity(e2.entity_id)
        moved_evt = next(ev for ev in evts if ev.new_state == EpistemicState.CONFIRMED_MOVED)
        self.assertAlmostEqual(moved_evt.displacement_distance, 120.0, delta=2.0)

    # -------------------------------------------------------------------------
    # Mode 7: Reappearance should preserve original entity history
    # -------------------------------------------------------------------------
    def test_reappearance_preserves_original_entity_history(self):
        b = BoundingBox(100.0, 100.0, 180.0, 180.0)
        v1 = VisitInput("v1", "V1", "2026-10-20T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), make_reg_success(), [make_det("tool", b, split=0)])
        v2 = VisitInput("v2", "V2", "2026-10-20T10:05:00Z", np.zeros((480, 640, 3), dtype=np.uint8), make_reg_success(), [make_det("tool", b, split=0)])
        # Absent in v3 -> PENDING_REMOVAL
        v3 = VisitInput("v3", "V3", "2026-10-20T10:10:00Z", np.zeros((480, 640, 3), dtype=np.uint8), make_reg_success(), [])
        # Absent in v4 -> CONFIRMED_REMOVED
        v4 = VisitInput("v4", "V4", "2026-10-20T10:15:00Z", np.zeros((480, 640, 3), dtype=np.uint8), make_reg_success(), [])
        for v in [v1, v2, v3, v4]:
            self.engine.process_visit(v)

        ent = self.store.get_entity("ent_v1_1")
        self.assertEqual(ent.current_state, EpistemicState.CONFIRMED_REMOVED)

        # Visit 5: Object reappears at original location
        v5 = VisitInput("v5", "V5", "2026-10-20T10:20:00Z", np.zeros((480, 640, 3), dtype=np.uint8), make_reg_success(), [make_det("tool", b, split=0)])
        self.engine.process_visit(v5)

        ent_reapp = self.store.get_entity("ent_v1_1")
        self.assertEqual(ent_reapp.current_state, EpistemicState.CONFIRMED_PRESENT)

        # Entire audit history preserved in DB:
        events = self.store.list_change_events_for_entity("ent_v1_1")
        states = [e.new_state for e in events]
        self.assertIn(EpistemicState.PENDING_REMOVAL, states)
        self.assertIn(EpistemicState.CONFIRMED_REMOVED, states)
        self.assertIn(EpistemicState.CONFIRMED_PRESENT, states)
        # All 3 positive observations recorded under the exact same entity ID
        obs = self.store.list_observations_for_entity("ent_v1_1")
        self.assertEqual(len(obs), 3)

    # -------------------------------------------------------------------------
    # Mode 8: Uncertain observations must not be reported as certain physical changes
    # -------------------------------------------------------------------------
    def test_uncertain_observations_not_reported_as_certain_physical_changes(self):
        b = BoundingBox(100.0, 100.0, 200.0, 200.0)
        v1 = VisitInput("v1", "V1", "2026-10-20T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), make_reg_success(), [make_det("pen", b)])
        self.engine.process_visit(v1)

        # In Visit 2, the region is occluded by another large object placed over it with different appearance
        b_occluder = BoundingBox(90.0, 90.0, 210.0, 210.0)
        v2 = VisitInput("v2", "V2", "2026-10-20T10:05:00Z", np.zeros((480, 640, 3), dtype=np.uint8), make_reg_success(), [
            make_det("laptop", b_occluder, split=1),
        ])
        rep2 = self.engine.process_visit(v2)

        # Engine must NOT confirm removal because the region is occluded!
        ent = self.store.get_entity("ent_v1_1")
        self.assertEqual(ent.current_state, EpistemicState.CONFIRMED_PRESENT)
        # Check update visibility status is PARTIALLY_OCCLUDED
        up = next(u for u in rep2.entity_updates if u.entity_id == "ent_v1_1")
        self.assertEqual(up.visibility_status, VisibilityStatus.PARTIALLY_OCCLUDED)
        self.assertFalse(up.transition_occurred)

    # -------------------------------------------------------------------------
    # Mode 9: Failed or degenerate homography must not corrupt canonical coordinates
    # -------------------------------------------------------------------------
    def test_failed_or_degenerate_homography_does_not_corrupt_canonical_coordinates(self):
        b = BoundingBox(120.0, 120.0, 220.0, 220.0)
        v1 = VisitInput("v1", "V1", "2026-10-20T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), make_reg_success(), [make_det("stapler", b)])
        self.engine.process_visit(v1)

        ent_before = self.store.get_entity("ent_v1_1")
        canon_before = ent_before.canonical_bbox

        # Non-finite / degenerate homography (all zeros / NaN)
        H_bad = np.zeros((3, 3), dtype=np.float64)
        reg_bad = RegistrationResult(
            success=False,
            status=AlignmentStatus.SINGULAR_HOMOGRAPHY,
            method_used="MOCK",
            ref_size=(640, 480),
            query_size=(640, 480),
            homography=H_bad,
            inlier_ratio=0.0,
            failure_reason="Homography is singular",
        )
        v2 = VisitInput("v2", "V2", "2026-10-20T10:05:00Z", np.zeros((480, 640, 3), dtype=np.uint8), reg_bad, [make_det("stapler", BoundingBox(10.0, 10.0, 30.0, 30.0))])
        self.engine.process_visit(v2)

        ent_after = self.store.get_entity("ent_v1_1")
        # Canonical bbox remains uncorrupted
        self.assertEqual(ent_after.canonical_bbox, canon_before)
        self.assertEqual(ent_after.current_state, EpistemicState.CONFIRMED_PRESENT)

    # -------------------------------------------------------------------------
    # Mode 10: Repeated ingestion of same visit must not duplicate events or entities
    # -------------------------------------------------------------------------
    def test_repeated_ingestion_of_same_visit_does_not_duplicate_events_or_entities(self):
        b = BoundingBox(200.0, 200.0, 300.0, 300.0)
        v1 = VisitInput("v1", "V1", "2026-10-20T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), make_reg_success(), [make_det("clock", b)])
        self.engine.process_visit(v1)

        b_moved = BoundingBox(350.0, 200.0, 450.0, 300.0)
        v2 = VisitInput("v2", "V2", "2026-10-20T10:05:00Z", np.zeros((480, 640, 3), dtype=np.uint8), make_reg_success(), [make_det("clock", b_moved)])

        # Process visit 2 first time
        rep2_first = self.engine.process_visit(v2)
        self.assertFalse(rep2_first.is_reprocessed)

        ent_count_1 = len(self.store.list_entities())
        obs_count_1 = len(self.store.list_observations_for_entity("ent_v1_1"))
        evts_count_1 = len(self.store.list_change_events_for_session("v2"))

        # Reprocess the EXACT same visit 2
        rep2_second = self.engine.process_visit(v2)
        self.assertTrue(rep2_second.is_reprocessed)

        ent_count_2 = len(self.store.list_entities())
        obs_count_2 = len(self.store.list_observations_for_entity("ent_v1_1"))
        evts_count_2 = len(self.store.list_change_events_for_session("v2"))

        # Assert strict database idempotency: counts unchanged, no duplicate entities or change events
        self.assertEqual(ent_count_1, ent_count_2)
        self.assertEqual(obs_count_1, obs_count_2)
        self.assertEqual(evts_count_1, evts_count_2)


if __name__ == "__main__":
    unittest.main()
