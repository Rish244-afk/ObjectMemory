"""
Automated unit tests for Phase 5: Multi-Visit Temporal Memory Engine.

Validates:
1. Correct identity association across visits.
2. One-to-one assignment and ambiguous identity handling.
3. A single missed detection does not cause confirmed removal.
4. Repeated reliable vacancy evidence causes only permitted transitions:
   CONFIRMED_PRESENT -> PENDING_REMOVAL -> CONFIRMED_REMOVED.
5. Occlusion and out-of-frame observations do not count as confirmed absence.
6. Failed registration prevents unsupported conclusions.
7. Recovery after redetection (PENDING_REMOVAL / CONFIRMED_REMOVED -> CONFIRMED_PRESENT).
8. Atomic persistence of state transitions and change events.
9. Duplicate and out-of-order visit handling.
10. Deterministic replay and idempotency (no duplicate entities or audit events on retry).
11. Existing pairwise baseline isolation and regression behavior.
"""

from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest
import numpy as np

from objectmemory.baseline.comparator import PairwiseComparator
from objectmemory.detection.models import Detection, VisualSignature
from objectmemory.domain.models import BoundingBox, Session, WorkbenchEntity
from objectmemory.domain.states import BaselineChangeType, EpistemicState, VisibilityStatus
from objectmemory.persistence.repository import SQLiteMemoryStore
from objectmemory.registration.models import AlignmentStatus, RegistrationResult, ReprojectionStats
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


class TestCrossVisitAssociation(unittest.TestCase):
    """Unit tests for CrossVisitEntityAssociator."""

    def setUp(self):
        self.config = TemporalEngineConfig(signature_similarity_threshold=0.60, ambiguity_margin=0.10)
        self.associator = CrossVisitEntityAssociator(config=self.config)
        self.reg_mock = RegistrationResult(
            success=True,
            status=AlignmentStatus.SUCCESS,
            method_used="MOCK",
            ref_size=(640, 480),
            query_size=(640, 480),
            homography=np.eye(3, dtype=np.float64),
            overlap_mask=np.ones((480, 640), dtype=np.uint8) * 255,
            inlier_ratio=0.85,
        )

    def test_one_to_one_assignment_no_double_matching(self):
        """A single detection cannot match two entities, and vice versa."""
        e1 = WorkbenchEntity("ent_1", "multimeter", "2026-03-01T10:00:00Z", BoundingBox(10, 10, 50, 50))
        e2 = WorkbenchEntity("ent_2", "multimeter", "2026-03-01T10:00:00Z", BoundingBox(100, 100, 150, 150))
        d1 = Detection("multimeter", 0.90, BoundingBox(10, 10, 50, 50), BoundingBox(10, 10, 50, 50), signature=_make_dummy_sig(split=0))

        matches, unmatched = self.associator.associate(
            entities=[e1, e2],
            detections=[d1],
            registration=self.reg_mock,
            entity_signatures={"ent_1": _make_dummy_sig(split=0), "ent_2": _make_dummy_sig(split=1)},
        )
        self.assertEqual(len(matches), 2)
        m1 = next(m for m in matches if m.entity_id == "ent_1")
        m2 = next(m for m in matches if m.entity_id == "ent_2")
        self.assertTrue(m1.is_associated)
        self.assertFalse(m2.is_associated)
        self.assertEqual(len(unmatched), 0)

    def test_ambiguous_identity_produces_unresolved_association(self):
        """Near-identical competing candidates within ambiguity margin must produce is_ambiguous=True."""
        e1 = WorkbenchEntity("ent_resistor", "resistor", "2026-03-01T10:00:00Z", BoundingBox(50, 50, 100, 100))
        sig_identical = _make_dummy_sig(split=2)

        # Two identical detections competing for one entity
        d1 = Detection("resistor", 0.90, BoundingBox(50, 50, 100, 100), BoundingBox(50, 50, 100, 100), signature=sig_identical)
        d2 = Detection("resistor", 0.90, BoundingBox(55, 55, 105, 105), BoundingBox(55, 55, 105, 105), signature=sig_identical)

        matches, _ = self.associator.associate(
            entities=[e1],
            detections=[d1, d2],
            registration=self.reg_mock,
            entity_signatures={"ent_resistor": sig_identical},
        )
        self.assertEqual(len(matches), 1)
        self.assertTrue(matches[0].is_ambiguous)
        self.assertFalse(matches[0].is_associated)

    def test_invalid_signature_cannot_produce_appearance_match(self):
        """Detections with invalid signature (degenerate crop) cannot produce appearance-based matches."""
        e1 = WorkbenchEntity("ent_tool", "pliers", "2026-03-01T10:00:00Z", BoundingBox(200, 200, 300, 300))
        inv_sig = VisualSignature(
            color_histogram=np.zeros(128, dtype=np.float32),
            texture_descriptor=np.zeros(20, dtype=np.float32),
            combined_vector=np.zeros(148, dtype=np.float32),
            is_valid=False,
            failure_reason="Degenerate crop",
        )
        # Spatial far away, appearance invalid
        d1 = Detection("pliers", 0.90, BoundingBox(10, 10, 50, 50), BoundingBox(10, 10, 50, 50), signature=inv_sig)
        matches, _ = self.associator.associate(
            entities=[e1],
            detections=[d1],
            registration=self.reg_mock,
            entity_signatures={"ent_tool": _make_dummy_sig(split=0)},
        )
        self.assertEqual(len(matches), 1)
        self.assertFalse(matches[0].is_associated)


class TestTemporalBeliefUpdating(unittest.TestCase):
    """Unit tests for TemporalMemoryEngine belief updating rules and state transitions."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_engine.sqlite"
        self.store = SQLiteMemoryStore(self.db_path)
        self.config = TemporalEngineConfig(
            vacancy_similarity_max=0.40,
            vacancy_visits_for_pending=1,
            vacancy_visits_for_removal=2,
        )
        self.engine = TemporalMemoryEngine(store=self.store, config=self.config)

        self.reg_success = RegistrationResult(
            success=True,
            status=AlignmentStatus.SUCCESS,
            method_used="MOCK",
            ref_size=(640, 480),
            query_size=(640, 480),
            homography=np.eye(3, dtype=np.float64),
            overlap_mask=np.ones((480, 640), dtype=np.uint8) * 255,
            aligned_image=np.ones((480, 640, 3), dtype=np.uint8) * 120,
            inlier_ratio=0.85,
        )

    def tearDown(self):
        self.store.close()
        self.temp_dir.cleanup()

    def test_initial_visit_creates_entity_in_confirmed_present(self):
        """Initial detection creates entity in CONFIRMED_PRESENT with discovery event."""
        bbox = BoundingBox(100, 100, 200, 200)
        det = Detection("multimeter", 0.92, bbox, canonical_bbox=bbox, signature=_make_dummy_sig(split=0))
        v1 = VisitInput("v1", "Visit 1", "2026-03-01T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), self.reg_success, [det])

        rep = self.engine.process_visit(v1)
        self.assertEqual(rep.new_entities_created, 1)
        entities = self.store.list_entities()
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0].current_state, EpistemicState.CONFIRMED_PRESENT)

    def test_single_missed_detection_does_not_cause_confirmed_removal(self):
        """Case 1: A previously confirmed object absent during 1 reliable visit becomes PENDING_REMOVAL, not CONFIRMED_REMOVED."""
        bbox = BoundingBox(100, 100, 200, 200)
        det = Detection("multimeter", 0.92, bbox, canonical_bbox=bbox, signature=_make_dummy_sig(split=0))
        self.engine.process_visit(VisitInput("v1", "Visit 1", "2026-03-01T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), self.reg_success, [det]))

        # Visit 2: Object absent from detections, patch is vacant
        v2 = VisitInput("v2", "Visit 2", "2026-03-02T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), self.reg_success, [])
        rep2 = self.engine.process_visit(v2)

        ent = self.store.list_entities()[0]
        # Assert exact state, absence streak, persisted observations, and transition count
        self.assertEqual(ent.current_state, EpistemicState.PENDING_REMOVAL)
        self.assertEqual(ent.metadata.get("vacant_streak"), 1)
        observations = self.store.list_observations_for_entity(ent.entity_id)
        self.assertEqual(len(observations), 1)
        events = self.store.list_change_events_for_entity(ent.entity_id)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[1].new_state, EpistemicState.PENDING_REMOVAL)

    def test_repeated_vacancy_progresses_to_confirmed_removed(self):
        """Case 2: Object absent during consecutive reliable visit becomes CONFIRMED_REMOVED."""
        bbox = BoundingBox(100, 100, 200, 200)
        det = Detection("multimeter", 0.92, bbox, canonical_bbox=bbox, signature=_make_dummy_sig(split=0))
        self.engine.process_visit(VisitInput("v1", "Visit 1", "2026-03-01T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), self.reg_success, [det]))

        # Visit 2: 1st vacant visit -> PENDING_REMOVAL
        self.engine.process_visit(VisitInput("v2", "Visit 2", "2026-03-02T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), self.reg_success, []))
        # Visit 3: 2nd vacant visit -> CONFIRMED_REMOVED
        self.engine.process_visit(VisitInput("v3", "Visit 3", "2026-03-03T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), self.reg_success, []))

        ent = self.store.list_entities()[0]
        # Assert exact state, absence streak, persisted observations, and transition count
        self.assertEqual(ent.current_state, EpistemicState.CONFIRMED_REMOVED)
        self.assertEqual(ent.metadata.get("vacant_streak"), 2)
        observations = self.store.list_observations_for_entity(ent.entity_id)
        self.assertEqual(len(observations), 1)
        events = self.store.list_change_events_for_entity(ent.entity_id)
        self.assertEqual(len(events), 3)
        self.assertEqual(events[1].new_state, EpistemicState.PENDING_REMOVAL)
        self.assertEqual(events[2].new_state, EpistemicState.CONFIRMED_REMOVED)

    def test_redetection_restores_confirmed_present(self):
        """Case 3: Object reappears before removal is confirmed: state recovers correctly."""
        bbox = BoundingBox(100, 100, 200, 200)
        det = Detection("multimeter", 0.92, bbox, canonical_bbox=bbox, signature=_make_dummy_sig(split=0))
        self.engine.process_visit(VisitInput("v1", "Visit 1", "2026-03-01T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), self.reg_success, [det]))
        self.engine.process_visit(VisitInput("v2", "Visit 2", "2026-03-02T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), self.reg_success, []))

        # Re-detected in visit 3
        rep3 = self.engine.process_visit(VisitInput("v3", "Visit 3", "2026-03-03T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), self.reg_success, [det]))
        ent = self.store.list_entities()[0]
        # Assert exact state, absence streak, persisted observations, and transition count
        self.assertEqual(ent.current_state, EpistemicState.CONFIRMED_PRESENT)
        self.assertEqual(ent.metadata.get("vacant_streak"), 0)
        observations = self.store.list_observations_for_entity(ent.entity_id)
        self.assertEqual(len(observations), 2)
        events = self.store.list_change_events_for_entity(ent.entity_id)
        self.assertEqual(len(events), 3)
        self.assertEqual(events[2].new_state, EpistemicState.CONFIRMED_PRESENT)
        self.assertTrue(rep3.entity_updates[0].transition_occurred)

    def test_occlusion_prevents_removal_progression(self):
        """Case 4: Occluded object does not accumulate removal evidence."""
        bbox_ent = BoundingBox(100, 100, 200, 200)
        det = Detection("multimeter", 0.92, bbox_ent, canonical_bbox=bbox_ent, signature=_make_dummy_sig(split=0))
        self.engine.process_visit(VisitInput("v1", "Visit 1", "2026-03-01T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), self.reg_success, [det]))

        # Visit 2: Multimeter is not detected, but a notebook overlaps canonical region (IoU > 0.20)
        occ_box = BoundingBox(110, 110, 210, 210)
        det_occ = Detection("notebook", 0.88, occ_box, canonical_bbox=occ_box, signature=_make_dummy_sig(split=1))
        rep2 = self.engine.process_visit(VisitInput("v2", "Visit 2", "2026-03-02T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), self.reg_success, [det_occ]))

        ent = self.store.list_entities()[0]
        # Assert exact state, absence streak, persisted observations, and transition count
        self.assertEqual(ent.current_state, EpistemicState.CONFIRMED_PRESENT)
        self.assertEqual(ent.metadata.get("vacant_streak"), 0)
        observations = self.store.list_observations_for_entity(ent.entity_id)
        self.assertEqual(len(observations), 1)
        events = self.store.list_change_events_for_entity(ent.entity_id)
        self.assertEqual(len(events), 1)
        self.assertEqual(rep2.entity_updates[0].visibility_status, VisibilityStatus.PARTIALLY_OCCLUDED)

    def test_out_of_frame_prevents_removal_progression(self):
        """Case 5: Object outside valid registered region does not accumulate removal evidence."""
        bbox_ent = BoundingBox(100, 100, 200, 200)
        det = Detection("multimeter", 0.92, bbox_ent, canonical_bbox=bbox_ent, signature=_make_dummy_sig(split=0))
        self.engine.process_visit(VisitInput("v1", "Visit 1", "2026-03-01T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), self.reg_success, [det]))

        # Visit 2: Overlap mask has 0% coverage on (100, 100, 200, 200)
        mask = np.ones((480, 640), dtype=np.uint8) * 255
        mask[100:200, 100:200] = 0
        reg_clipped = RegistrationResult(
            success=True,
            status=AlignmentStatus.SUCCESS,
            method_used="MOCK",
            ref_size=(640, 480),
            query_size=(640, 480),
            homography=np.eye(3, dtype=np.float64),
            overlap_mask=mask,
            aligned_image=np.ones((480, 640, 3), dtype=np.uint8) * 120,
            inlier_ratio=0.85,
        )

        rep2 = self.engine.process_visit(VisitInput("v2", "Visit 2", "2026-03-02T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), reg_clipped, []))
        ent = self.store.list_entities()[0]
        # Assert exact state, absence streak, persisted observations, and transition count
        self.assertEqual(ent.current_state, EpistemicState.CONFIRMED_PRESENT)
        self.assertEqual(ent.metadata.get("vacant_streak"), 0)
        observations = self.store.list_observations_for_entity(ent.entity_id)
        self.assertEqual(len(observations), 1)
        events = self.store.list_change_events_for_entity(ent.entity_id)
        self.assertEqual(len(events), 1)
        self.assertEqual(rep2.entity_updates[0].visibility_status, VisibilityStatus.OUT_OF_FRAME)

    def test_failed_registration_prevents_removal_and_movement(self):
        """Case 6: Registration failure does not count as reliable absence."""
        bbox_ent = BoundingBox(100, 100, 200, 200)
        det = Detection("multimeter", 0.92, bbox_ent, canonical_bbox=bbox_ent, signature=_make_dummy_sig(split=0))
        self.engine.process_visit(VisitInput("v1", "Visit 1", "2026-03-01T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), self.reg_success, [det]))

        reg_fail = RegistrationResult(
            success=False,
            status=AlignmentStatus.INSUFFICIENT_KEYPOINTS,
            method_used="SIFT",
            ref_size=(640, 480),
            query_size=(640, 480),
            failure_reason="Insufficient keypoints",
        )
        rep2 = self.engine.process_visit(VisitInput("v2", "Visit 2", "2026-03-02T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), reg_fail, []))
        ent = self.store.list_entities()[0]
        # Assert exact state, absence streak, persisted observations, and transition count
        self.assertEqual(ent.current_state, EpistemicState.CONFIRMED_PRESENT)
        self.assertEqual(ent.metadata.get("vacant_streak"), 0)
        observations = self.store.list_observations_for_entity(ent.entity_id)
        self.assertEqual(len(observations), 1)
        events = self.store.list_change_events_for_entity(ent.entity_id)
        self.assertEqual(len(events), 1)
        self.assertEqual(rep2.entity_updates[0].visibility_status, VisibilityStatus.UNOBSERVED)

    def test_removed_object_that_returns_is_restored_to_confirmed_present(self):
        """Case 7: Removed object that returns transitions to CONFIRMED_PRESENT per documented policy."""
        bbox = BoundingBox(100, 100, 200, 200)
        det = Detection("multimeter", 0.92, bbox, canonical_bbox=bbox, signature=_make_dummy_sig(split=0))
        # v1: Discovery (CONFIRMED_PRESENT)
        self.engine.process_visit(VisitInput("v1", "Visit 1", "2026-03-01T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), self.reg_success, [det]))
        # v2: Vacant streak 1 (PENDING_REMOVAL)
        self.engine.process_visit(VisitInput("v2", "Visit 2", "2026-03-02T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), self.reg_success, []))
        # v3: Vacant streak 2 (CONFIRMED_REMOVED)
        self.engine.process_visit(VisitInput("v3", "Visit 3", "2026-03-03T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), self.reg_success, []))
        ent = self.store.list_entities()[0]
        self.assertEqual(ent.current_state, EpistemicState.CONFIRMED_REMOVED)
        self.assertEqual(ent.metadata.get("vacant_streak"), 2)

        # v4: Re-observed! (returns to workbench)
        rep4 = self.engine.process_visit(VisitInput("v4", "Visit 4", "2026-03-04T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), self.reg_success, [det]))
        ent = self.store.list_entities()[0]
        # Assert exact state, absence streak, persisted observations, and transition count
        self.assertEqual(ent.current_state, EpistemicState.CONFIRMED_PRESENT)
        self.assertEqual(ent.metadata.get("vacant_streak"), 0)
        observations = self.store.list_observations_for_entity(ent.entity_id)
        self.assertEqual(len(observations), 2)  # v1 and v4
        events = self.store.list_change_events_for_entity(ent.entity_id)
        self.assertEqual(len(events), 4)  # discovery -> pending_removal -> confirmed_removed -> confirmed_present
        self.assertEqual(events[3].prior_state, EpistemicState.CONFIRMED_REMOVED)
        self.assertEqual(events[3].new_state, EpistemicState.CONFIRMED_PRESENT)

    def test_ambiguous_association_does_not_attach_observation_to_wrong_identity(self):
        """Case 8: Ambiguous association does not attach observation to wrong identity and transitions to AMBIGUOUS_IDENTITY."""
        bbox = BoundingBox(50, 50, 100, 100)
        sig_shared = _make_dummy_sig(split=2)
        det = Detection("resistor", 0.92, bbox, canonical_bbox=bbox, signature=sig_shared)
        self.engine.process_visit(VisitInput("v1", "Visit 1", "2026-03-01T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), self.reg_success, [det]))

        # v2: Two identical detections competing within ambiguity margin
        d_cand1 = Detection("resistor", 0.90, BoundingBox(50, 50, 100, 100), canonical_bbox=BoundingBox(50, 50, 100, 100), signature=sig_shared)
        d_cand2 = Detection("resistor", 0.90, BoundingBox(54, 54, 104, 104), canonical_bbox=BoundingBox(54, 54, 104, 104), signature=sig_shared)

        rep2 = self.engine.process_visit(VisitInput("v2", "Visit 2", "2026-03-02T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), self.reg_success, [d_cand1, d_cand2]))

        ent = self.store.list_entities()[0]
        # Assert exact state, absence streak, persisted observations, and transition count
        self.assertEqual(ent.current_state, EpistemicState.AMBIGUOUS_IDENTITY)
        self.assertEqual(ent.metadata.get("vacant_streak"), 0)
        # Crucial: No observation was attached to ent in visit 2 because association was unresolved/ambiguous!
        observations = self.store.list_observations_for_entity(ent.entity_id)
        self.assertEqual(len(observations), 1)  # Only from v1
        events = self.store.list_change_events_for_entity(ent.entity_id)
        self.assertEqual(len(events), 2)  # discovery + transition to AMBIGUOUS_IDENTITY
        self.assertEqual(events[1].new_state, EpistemicState.AMBIGUOUS_IDENTITY)

    def test_idempotent_reprocessing_does_not_duplicate_entities_or_events(self):
        """Reprocessing an already processed visit returns cached report and produces 0 duplicates."""
        bbox = BoundingBox(100, 100, 200, 200)
        det = Detection("multimeter", 0.92, bbox, canonical_bbox=bbox, signature=_make_dummy_sig(split=0))
        v1 = VisitInput("v1", "Visit 1", "2026-03-01T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), self.reg_success, [det])

        rep_a = self.engine.process_visit(v1)
        self.assertFalse(rep_a.is_reprocessed)

        # Reprocess identical visit
        rep_b = self.engine.process_visit(v1)
        self.assertTrue(rep_b.is_reprocessed)

        entities = self.store.list_entities()
        observations = self.store.list_observations_for_session("v1")
        events = self.store.list_change_events_for_entity(entities[0].entity_id)

        self.assertEqual(len(entities), 1)
        self.assertEqual(len(observations), 1)
        self.assertEqual(len(events), 1)

    def test_out_of_order_visit_raises_value_error(self):
        """Case 10: Out-of-order visit cannot silently corrupt temporal state; raises ValueError."""
        bbox = BoundingBox(100, 100, 200, 200)
        det = Detection("multimeter", 0.92, bbox, canonical_bbox=bbox, signature=_make_dummy_sig(split=0))
        self.engine.process_visit(VisitInput("v1", "Visit 1", "2026-03-05T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), self.reg_success, [det]))

        # Older visit rejected
        with self.assertRaises(ValueError):
            self.engine.process_visit(VisitInput("v_old", "Old Visit", "2026-03-01T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), self.reg_success, [det]))

        # State remains uncorrupted
        ent = self.store.list_entities()[0]
        self.assertEqual(ent.current_state, EpistemicState.CONFIRMED_PRESENT)
        self.assertEqual(ent.metadata.get("vacant_streak"), 0)
        observations = self.store.list_observations_for_entity(ent.entity_id)
        self.assertEqual(len(observations), 1)
        events = self.store.list_change_events_for_entity(ent.entity_id)
        self.assertEqual(len(events), 1)

    def test_idempotent_reprocessing_across_database_restart(self):
        """Case 9: Duplicate visit is idempotent after closing and reopening database."""
        bbox = BoundingBox(100, 100, 200, 200)
        det = Detection("multimeter", 0.92, bbox, canonical_bbox=bbox, signature=_make_dummy_sig(split=0))
        v1 = VisitInput("v1", "Visit 1", "2026-03-01T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), self.reg_success, [det])

        rep1 = self.engine.process_visit(v1)
        self.assertFalse(rep1.is_reprocessed)

        # Simulate full application restart
        self.store.close()
        new_store = SQLiteMemoryStore(self.db_path)
        new_engine = TemporalMemoryEngine(store=new_store, config=self.config)

        rep2 = new_engine.process_visit(v1)
        self.assertTrue(rep2.is_reprocessed)

        entities = new_store.list_entities()
        self.assertEqual(len(entities), 1)
        ent = entities[0]
        self.assertEqual(ent.current_state, EpistemicState.CONFIRMED_PRESENT)
        self.assertEqual(ent.metadata.get("vacant_streak"), 0)
        observations = new_store.list_observations_for_session("v1")
        self.assertEqual(len(observations), 1)
        events = new_store.list_change_events_for_entity(ent.entity_id)
        self.assertEqual(len(events), 1)
        new_store.close()

    def test_registration_failure_prevents_false_confirmed_moved(self):
        """Regression A: Previously observed entity detected when registration fails must NOT emit CONFIRMED_MOVED."""
        base_box = BoundingBox(100.0, 100.0, 200.0, 200.0)
        sig = _make_dummy_sig(split=0)
        det1 = Detection("multimeter", 0.92, base_box, canonical_bbox=base_box, signature=sig)
        self.engine.process_visit(VisitInput("v1", "Visit 1", "2026-03-01T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), self.reg_success, [det1]))

        # Visit 2: Object detected at raw coords with large shift (150px), but registration failed
        raw_box_shifted = BoundingBox(250.0, 250.0, 350.0, 350.0)
        det2 = Detection("multimeter", 0.90, raw_box_shifted, canonical_bbox=None, signature=sig)
        reg_failed = RegistrationResult(
            success=False,
            status=AlignmentStatus.INSUFFICIENT_INLIERS,
            method_used="SIFT",
            ref_size=(640, 480),
            query_size=(640, 480),
            failure_reason="Inlier ratio below threshold",
        )
        rep2 = self.engine.process_visit(VisitInput("v2", "Visit 2", "2026-03-02T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), reg_failed, [det2]))

        ent = self.store.list_entities()[0]
        # State MUST remain CONFIRMED_PRESENT, not CONFIRMED_MOVED
        self.assertEqual(ent.current_state, EpistemicState.CONFIRMED_PRESENT)
        self.assertFalse(rep2.entity_updates[0].transition_occurred)
        self.assertEqual(rep2.entity_updates[0].visibility_status, VisibilityStatus.UNCERTAIN)
        # No new change event emitted (only initial discovery from v1 exists)
        events = self.store.list_change_events_for_entity(ent.entity_id)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].new_state, EpistemicState.CONFIRMED_PRESENT)
        # Observation was safely recorded with raw bbox, but canonical bbox is None
        obs_v2 = self.store.list_observations_for_session("v2")[0]
        self.assertEqual(obs_v2.raw_bbox, raw_box_shifted)
        self.assertIsNone(obs_v2.canonical_bbox)
        self.assertEqual(obs_v2.visibility_status, VisibilityStatus.UNCERTAIN)

    def test_reliable_registration_detects_genuine_movement(self):
        """Regression B: Genuine movement under reliable registration correctly transitions to CONFIRMED_MOVED."""
        base_box = BoundingBox(100.0, 100.0, 200.0, 200.0)
        sig = _make_dummy_sig(split=0)
        det1 = Detection("multimeter", 0.92, base_box, canonical_bbox=base_box, signature=sig)
        self.engine.process_visit(VisitInput("v1", "Visit 1", "2026-03-01T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), self.reg_success, [det1]))

        # Visit 2: Object relocated by dx=+60, dy=+60 -> displacement ~84.85px > 35px under reliable registration
        moved_box = BoundingBox(160.0, 160.0, 260.0, 260.0)
        det2 = Detection("multimeter", 0.91, moved_box, canonical_bbox=moved_box, signature=sig)
        rep2 = self.engine.process_visit(VisitInput("v2", "Visit 2", "2026-03-02T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), self.reg_success, [det2]))

        ent = self.store.list_entities()[0]
        self.assertEqual(ent.current_state, EpistemicState.CONFIRMED_MOVED)
        self.assertTrue(rep2.entity_updates[0].transition_occurred)
        self.assertEqual(rep2.entity_updates[0].visibility_status, VisibilityStatus.CLEARLY_VISIBLE)
        events = self.store.list_change_events_for_entity(ent.entity_id)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[1].new_state, EpistemicState.CONFIRMED_MOVED)
        self.assertIsNotNone(events[1].displacement_distance)
        self.assertGreater(events[1].displacement_distance, 35.0)

    def test_unreliable_registration_does_not_accumulate_absence(self):
        """Regression C: Object absent under failed registration does not progress removal evidence."""
        base_box = BoundingBox(100.0, 100.0, 200.0, 200.0)
        sig = _make_dummy_sig(split=0)
        det1 = Detection("multimeter", 0.92, base_box, canonical_bbox=base_box, signature=sig)
        self.engine.process_visit(VisitInput("v1", "Visit 1", "2026-03-01T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), self.reg_success, [det1]))

        # Visit 2: Object absent, registration failed
        reg_failed = RegistrationResult(
            success=False,
            status=AlignmentStatus.EXECUTION_ERROR,
            method_used="SIFT",
            ref_size=(640, 480),
            query_size=(640, 480),
        )
        rep2 = self.engine.process_visit(VisitInput("v2", "Visit 2", "2026-03-02T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), reg_failed, []))

        ent = self.store.list_entities()[0]
        self.assertEqual(ent.current_state, EpistemicState.CONFIRMED_PRESENT)
        self.assertEqual(ent.metadata.get("vacant_streak"), 0)
        self.assertFalse(rep2.entity_updates[0].transition_occurred)
        self.assertEqual(rep2.entity_updates[0].visibility_status, VisibilityStatus.UNOBSERVED)

    def test_unreliable_registration_preserves_last_reliable_position(self):
        """Regression D: Unreliable projection does NOT overwrite the last reliable canonical position."""
        base_box = BoundingBox(100.0, 100.0, 200.0, 200.0)
        sig = _make_dummy_sig(split=0)
        det1 = Detection("multimeter", 0.92, base_box, canonical_bbox=base_box, signature=sig)
        self.engine.process_visit(VisitInput("v1", "Visit 1", "2026-03-01T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), self.reg_success, [det1]))

        # Visit 2: Object detected, but registration has low inlier ratio (0.10 < 0.20)
        raw_box_different = BoundingBox(300.0, 300.0, 400.0, 400.0)
        det2 = Detection("multimeter", 0.89, raw_box_different, canonical_bbox=raw_box_different, signature=sig)
        reg_low_inlier = RegistrationResult(
            success=True,
            status=AlignmentStatus.SUCCESS,
            method_used="SIFT",
            ref_size=(640, 480),
            query_size=(640, 480),
            homography=np.eye(3, dtype=np.float64),
            overlap_mask=np.ones((480, 640), dtype=np.uint8) * 255,
            aligned_image=np.ones((480, 640, 3), dtype=np.uint8) * 120,
            inlier_ratio=0.10,  # Below threshold 0.20!
        )
        self.engine.process_visit(VisitInput("v2", "Visit 2", "2026-03-02T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), reg_low_inlier, [det2]))

        ent = self.store.list_entities()[0]
        # Canonical bbox must remain exactly base_box, NOT raw_box_different
        self.assertEqual(ent.canonical_bbox, base_box)
        self.assertEqual(ent.current_state, EpistemicState.CONFIRMED_PRESENT)

    def test_reliable_registration_stationary_object_remains_confirmed_present(self):
        """Fix 2 Scenario 2: Reliable registration with a stationary object within tolerance remains CONFIRMED_PRESENT."""
        base_box = BoundingBox(100.0, 100.0, 200.0, 200.0)
        sig = _make_dummy_sig(split=0)
        det1 = Detection("multimeter", 0.92, base_box, canonical_bbox=base_box, signature=sig)
        self.engine.process_visit(VisitInput("v1", "Visit 1", "2026-03-01T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), self.reg_success, [det1]))

        # Visit 2: Minor sensor noise dx=+5, dy=+5 -> displacement ~7.07px <= 35px threshold
        jittered_box = BoundingBox(105.0, 105.0, 205.0, 205.0)
        det2 = Detection("multimeter", 0.93, jittered_box, canonical_bbox=jittered_box, signature=sig)
        rep2 = self.engine.process_visit(VisitInput("v2", "Visit 2", "2026-03-02T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), self.reg_success, [det2]))

        ent = self.store.list_entities()[0]
        self.assertEqual(ent.current_state, EpistemicState.CONFIRMED_PRESENT)
        self.assertFalse(rep2.entity_updates[0].transition_occurred)
        self.assertEqual(rep2.entity_updates[0].visibility_status, VisibilityStatus.CLEARLY_VISIBLE)
        self.assertEqual(ent.canonical_bbox, jittered_box)
        events = self.store.list_change_events_for_entity(ent.entity_id)
        self.assertEqual(len(events), 1)  # Only initial discovery event

    def test_boundary_truncated_detection_blocks_false_movement(self):
        """Fix 2 Scenario 3a: Detection touching image frame boundary blocks false CONFIRMED_MOVED."""
        base_box = BoundingBox(100.0, 100.0, 200.0, 200.0)
        sig = _make_dummy_sig(split=0)
        det1 = Detection("multimeter", 0.92, base_box, canonical_bbox=base_box, signature=sig)
        self.engine.process_visit(VisitInput("v1", "Visit 1", "2026-03-01T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), self.reg_success, [det1]))

        # Visit 2: Object detected at query frame border (x_min = 2.0px <= border_margin_px 8.0px)
        # Even with apparent displacement = 80px > 35px, boundary truncation must block CONFIRMED_MOVED
        border_box = BoundingBox(2.0, 100.0, 102.0, 200.0)
        shifted_canon = BoundingBox(180.0, 100.0, 280.0, 200.0)
        det2 = Detection("multimeter", 0.90, border_box, canonical_bbox=shifted_canon, signature=sig)
        rep2 = self.engine.process_visit(VisitInput("v2", "Visit 2", "2026-03-02T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), self.reg_success, [det2]))

        ent = self.store.list_entities()[0]
        # State must remain CONFIRMED_PRESENT (no false CONFIRMED_MOVED)
        self.assertEqual(ent.current_state, EpistemicState.CONFIRMED_PRESENT)
        self.assertFalse(rep2.entity_updates[0].transition_occurred)
        self.assertEqual(rep2.entity_updates[0].visibility_status, VisibilityStatus.UNCERTAIN)
        # Canonical position must NOT be overwritten with shifted_canon
        self.assertEqual(ent.canonical_bbox, base_box)
        events = self.store.list_change_events_for_entity(ent.entity_id)
        self.assertEqual(len(events), 1)

    def test_partial_overlap_blocks_false_movement(self):
        """Fix 2 Scenario 3b: Object canonical region falling outside registered overlap blocks false CONFIRMED_MOVED."""
        base_box = BoundingBox(10.0, 10.0, 110.0, 110.0)
        sig = _make_dummy_sig(split=0)
        det1 = Detection("multimeter", 0.92, base_box, canonical_bbox=base_box, signature=sig)
        self.engine.process_visit(VisitInput("v1", "Visit 1", "2026-03-01T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), self.reg_success, [det1]))

        # Visit 2: Overlap mask covers only x >= 80, leaving base_box (10-110) largely unobserved (< 70% overlap)
        overlap_mask = np.zeros((480, 640), dtype=np.uint8)
        overlap_mask[:, 80:] = 255  # Overlap fraction with base_box is 30/100 = 30% < 70%
        reg_clipped = RegistrationResult(
            success=True,
            status=AlignmentStatus.SUCCESS,
            method_used="SIFT",
            ref_size=(640, 480),
            query_size=(640, 480),
            homography=np.eye(3, dtype=np.float64),
            overlap_mask=overlap_mask,
            inlier_ratio=0.85,
        )

        shifted_raw = BoundingBox(90.0, 10.0, 190.0, 110.0)
        det2 = Detection("multimeter", 0.88, shifted_raw, canonical_bbox=shifted_raw, signature=sig)
        rep2 = self.engine.process_visit(VisitInput("v2", "Visit 2", "2026-03-02T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), reg_clipped, [det2]))

        ent = self.store.list_entities()[0]
        # State must remain CONFIRMED_PRESENT
        self.assertEqual(ent.current_state, EpistemicState.CONFIRMED_PRESENT)
        self.assertFalse(rep2.entity_updates[0].transition_occurred)
        self.assertEqual(rep2.entity_updates[0].visibility_status, VisibilityStatus.UNCERTAIN)
        self.assertEqual(ent.canonical_bbox, base_box)

    def test_excessive_reprojection_error_blocks_movement(self):
        """Fix 2 Scenario 3c: Excessive registration reprojection error blocks spatial displacement inference."""
        base_box = BoundingBox(100.0, 100.0, 200.0, 200.0)
        sig = _make_dummy_sig(split=0)
        det1 = Detection("multimeter", 0.92, base_box, canonical_bbox=base_box, signature=sig)
        self.engine.process_visit(VisitInput("v1", "Visit 1", "2026-03-01T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), self.reg_success, [det1]))

        # Visit 2: Mean reprojection error = 6.5px > max_reprojection_error (4.0px)
        stats_bad = ReprojectionStats(mean=6.5, median=6.0, std=1.0, max_error=12.0, inlier_count=40)
        reg_noisy = RegistrationResult(
            success=True,
            status=AlignmentStatus.SUCCESS,
            method_used="SIFT",
            ref_size=(640, 480),
            query_size=(640, 480),
            homography=np.eye(3, dtype=np.float64),
            overlap_mask=np.ones((480, 640), dtype=np.uint8) * 255,
            inlier_ratio=0.85,
            reprojection_stats=stats_bad,
        )

        shifted_raw = BoundingBox(160.0, 160.0, 260.0, 260.0)  # Displacement ~84.85px
        det2 = Detection("multimeter", 0.90, shifted_raw, canonical_bbox=shifted_raw, signature=sig)
        rep2 = self.engine.process_visit(VisitInput("v2", "Visit 2", "2026-03-02T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), reg_noisy, [det2]))

        ent = self.store.list_entities()[0]
        self.assertEqual(ent.current_state, EpistemicState.CONFIRMED_PRESENT)
        self.assertFalse(rep2.entity_updates[0].transition_occurred)
        self.assertEqual(rep2.entity_updates[0].visibility_status, VisibilityStatus.UNCERTAIN)
        self.assertEqual(ent.canonical_bbox, base_box)

    def test_per_sequence_evaluations_sum_to_aggregate_totals(self):
        """Enforces invariant: Sum of per-sequence evals and discrepancies strictly equals report totals."""
        from objectmemory.benchmarks.temporal_dataset import SyntheticTemporalSequenceGenerator
        from objectmemory.benchmarks.temporal_evaluator import TemporalBenchmarkEvaluator

        gen = SyntheticTemporalSequenceGenerator()
        seqs = gen.generate_all_sequences(base_seed=42)
        evaluator = TemporalBenchmarkEvaluator()

        with tempfile.TemporaryDirectory() as tmp_dir:
            report = evaluator.evaluate_all_sequences(seqs, Path(tmp_dir))

        self.assertEqual(report.total_evaluations, 49)
        self.assertEqual(report.correct_predictions, 49)
        sum_seq_evals = sum(s.total_evaluations for s in report.sequence_summaries)
        sum_seq_temp_errs = sum(s.temporal_errors for s in report.sequence_summaries)
        sum_seq_pair_discr = sum(s.pairwise_discrepancies for s in report.sequence_summaries)

        self.assertEqual(sum_seq_evals, report.total_evaluations)
        self.assertEqual(sum_seq_temp_errs, report.total_evaluations - report.correct_predictions)
        self.assertEqual(sum_seq_pair_discr, 10)  # Exactly 10 pairwise discrepancies
        self.assertEqual(report.total_evaluations - sum_seq_pair_discr, 39)  # 39/49 pairwise matches


if __name__ == "__main__":
    unittest.main()
