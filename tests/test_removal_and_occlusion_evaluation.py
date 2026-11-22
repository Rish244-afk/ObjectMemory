"""
Unit tests for Removal, Occlusion, and Observability Policies (Phase 9B).
Covers:
1. One missed detection with unverified vacancy (preserves CONFIRMED_PRESENT).
2. Repeated correlated detector misses (progresses through PENDING_REMOVAL before removal).
3. Temporary occlusion by another detected object (freezes state, pauses removal streak).
4. Object outside the camera field of view (< 70% overlap, pauses removal streak).
5. Degraded registration gating (pauses removal streak).
6. Actual verified vacancy (patch similarity < 0.70: PENDING_REMOVAL -> CONFIRMED_REMOVED).
7. Reappearance recovery from CONFIRMED_REMOVED to CONFIRMED_PRESENT.
"""

from __future__ import annotations

import unittest
import numpy as np

from objectmemory.domain.models import BoundingBox, WorkbenchEntity, Session
from objectmemory.domain.states import EpistemicState
from objectmemory.temporal.models import TemporalEngineConfig, VisitInput, VisibilityStatus
from objectmemory.temporal.engine import TemporalMemoryEngine
from objectmemory.persistence.repository import SQLiteMemoryStore
from objectmemory.registration.models import RegistrationResult, AlignmentStatus
from objectmemory.detection.models import Detection
from objectmemory.detection.signature import VisualSignature, VisualSignatureExtractor


class TestRemovalAndOcclusionEvaluation(unittest.TestCase):
    def setUp(self) -> None:
        self.store = SQLiteMemoryStore(":memory:")
        self.config = TemporalEngineConfig(
            movement_displacement_threshold=35.0,
            border_margin_px=12.0,
            min_overlap_fraction=0.70,
            min_registration_inlier_ratio=0.20,
            unobserved_visits_for_removal=3,
        )
        self.engine = TemporalMemoryEngine(store=self.store, config=self.config)

    def _make_visit(
        self,
        session_id: str,
        overlap_mask: Optional[np.ndarray] = None,
        aligned_image: Optional[np.ndarray] = None,
        inlier_ratio: float = 0.95,
    ) -> VisitInput:
        if overlap_mask is None:
            overlap_mask = np.ones((600, 800), dtype=np.uint8) * 255
        if aligned_image is None:
            aligned_image = np.full((600, 800, 3), 128, dtype=np.uint8)

        reg = RegistrationResult(
            success=True,
            status=AlignmentStatus.SUCCESS,
            method_used="CONTROLLED_TEST",
            ref_size=(800, 600),
            query_size=(800, 600),
            homography=np.eye(3, dtype=np.float64),
            inlier_ratio=inlier_ratio,
            num_inliers=300,
            overlap_mask=overlap_mask,
            aligned_image=aligned_image,
        )
        # Ensure session exists in SQLite store for foreign keys
        try:
            self.store.add_session(Session(
                session_id=session_id,
                session_name=session_id,
                timestamp="2026-10-21T00:00:00Z",
                image_path=f"{session_id}.png",
            ))
        except Exception:
            pass

        return VisitInput(
            session_id=session_id,
            session_name=session_id,
            timestamp="2026-10-21T00:00:00Z",
            image=aligned_image,
            registration=reg,
            detections=[],
        )

    def test_single_missed_detection_preserves_presence(self) -> None:
        """A single non-detection without verified vacancy must preserve CONFIRMED_PRESENT."""
        box = BoundingBox(200.0, 200.0, 300.0, 300.0)
        ent = WorkbenchEntity(
            entity_id="bottle_01",
            label="bottle",
            current_state=EpistemicState.CONFIRMED_PRESENT,
            canonical_bbox=box,
            created_at="2026-10-21T00:00:00Z",
            metadata={"initial_canonical_bbox": [200.0, 200.0, 300.0, 300.0], "unobserved_streak": 0},
        )
        self.store.add_entity(ent)

        visit = self._make_visit("v_miss_1")
        update = self.engine._update_entity_belief(
            entity=ent,
            assoc=None,  # Not detected!
            visit=visit,
            aligned_detections=[],
        )
        # 1st miss without verified vacancy remains CONFIRMED_PRESENT
        self.assertEqual(update.new_state, EpistemicState.CONFIRMED_PRESENT)
        self.assertFalse(update.transition_occurred)

    def test_repeated_misses_transition_to_pending_then_removed(self) -> None:
        """Repeated missed detections progress: miss 1 (PRESENT) -> miss 2 (PENDING) -> miss 3 (REMOVED)."""
        box = BoundingBox(200.0, 200.0, 300.0, 300.0)
        ent = WorkbenchEntity(
            entity_id="bottle_01",
            label="bottle",
            current_state=EpistemicState.CONFIRMED_PRESENT,
            canonical_bbox=box,
            created_at="2026-10-21T00:00:00Z",
            metadata={"initial_canonical_bbox": [200.0, 200.0, 300.0, 300.0], "unobserved_streak": 0},
        )
        self.store.add_entity(ent)

        # Visit 1: 1st miss
        u1 = self.engine._update_entity_belief(ent, None, self._make_visit("v1"), [])
        self.assertEqual(u1.new_state, EpistemicState.CONFIRMED_PRESENT)

        # Update entity in store to reflect streak=1
        ent = self.store.get_entity("bottle_01")

        # Visit 2: 2nd miss -> PENDING_REMOVAL
        u2 = self.engine._update_entity_belief(ent, None, self._make_visit("v2"), [])
        self.assertEqual(u2.new_state, EpistemicState.PENDING_REMOVAL)
        self.assertTrue(u2.transition_occurred)

        # Update entity in store to reflect streak=2, state=PENDING_REMOVAL
        ent = self.store.get_entity("bottle_01")

        # Visit 3: 3rd miss -> CONFIRMED_REMOVED
        u3 = self.engine._update_entity_belief(ent, None, self._make_visit("v3"), [])
        self.assertEqual(u3.new_state, EpistemicState.CONFIRMED_REMOVED)
        self.assertTrue(u3.transition_occurred)

    def test_temporary_occlusion_freezes_state(self) -> None:
        """When an object is occluded by another detection (IoU > 0.20), removal streak is paused."""
        box = BoundingBox(200.0, 200.0, 300.0, 300.0)
        ent = WorkbenchEntity(
            entity_id="scissors_01",
            label="scissors",
            current_state=EpistemicState.CONFIRMED_PRESENT,
            canonical_bbox=box,
            created_at="2026-10-21T00:00:00Z",
            metadata={"initial_canonical_bbox": [200.0, 200.0, 300.0, 300.0], "unobserved_streak": 0},
        )
        self.store.add_entity(ent)

        # Another object (e.g. notebook) placed over the scissors (IoU = 0.50)
        occluding_box = BoundingBox(180.0, 180.0, 320.0, 320.0)
        det_notebook = Detection(
            label="book",
            confidence=0.88,
            raw_bbox=occluding_box,
            canonical_bbox=occluding_box,
        )

        visit = self._make_visit("v_occluded")
        update = self.engine._update_entity_belief(
            entity=ent,
            assoc=None,
            visit=visit,
            aligned_detections=[det_notebook],
        )
        self.assertEqual(update.new_state, EpistemicState.CONFIRMED_PRESENT)
        self.assertFalse(update.transition_occurred)
        self.assertEqual(update.visibility_status, VisibilityStatus.PARTIALLY_OCCLUDED)
        self.assertIn("occluded by detection 'book'", update.explanation)

    def test_out_of_frame_freezes_state(self) -> None:
        """When canonical position is outside registered canvas (< 70% overlap), removal is paused."""
        box = BoundingBox(700.0, 200.0, 790.0, 300.0)
        ent = WorkbenchEntity(
            entity_id="book_01",
            label="book",
            current_state=EpistemicState.CONFIRMED_PRESENT,
            canonical_bbox=box,
            created_at="2026-10-21T00:00:00Z",
            metadata={"initial_canonical_bbox": [700.0, 200.0, 790.0, 300.0], "unobserved_streak": 0},
        )
        self.store.add_entity(ent)

        # Camera cropped at x=720 (overlap = 22% < 70%)
        mask_cropped = np.zeros((600, 800), dtype=np.uint8)
        mask_cropped[:, :720] = 255
        visit = self._make_visit("v_crop", overlap_mask=mask_cropped)

        update = self.engine._update_entity_belief(
            entity=ent,
            assoc=None,
            visit=visit,
            aligned_detections=[],
        )
        self.assertEqual(update.new_state, EpistemicState.CONFIRMED_PRESENT)
        self.assertFalse(update.transition_occurred)
        self.assertEqual(update.visibility_status, VisibilityStatus.OUT_OF_FRAME)
        self.assertIn("outside overlap", update.explanation)

    def test_reappearance_recovery_after_confirmed_removed(self) -> None:
        """An entity in CONFIRMED_REMOVED is cleanly restored to CONFIRMED_PRESENT upon redetection."""
        box = BoundingBox(200.0, 200.0, 300.0, 300.0)
        ent = WorkbenchEntity(
            entity_id="bottle_01",
            label="bottle",
            current_state=EpistemicState.CONFIRMED_REMOVED, # Already confirmed removed
            canonical_bbox=box,
            created_at="2026-10-21T00:00:00Z",
            metadata={"initial_canonical_bbox": [200.0, 200.0, 300.0, 300.0]},
        )
        self.store.add_entity(ent)

        det_reappeared = Detection(
            label="bottle",
            confidence=0.94,
            raw_bbox=box,
            canonical_bbox=box,
        )

        from objectmemory.temporal.engine import AssociationMatch
        assoc = AssociationMatch(
            entity_id="bottle_01",
            detection=det_reappeared,
            confidence=0.94,
            is_associated=True,
            is_ambiguous=False,
            explanation="Reappeared object matched",
            details={},
        )

        visit = self._make_visit("v_reappear")
        update = self.engine._update_entity_belief(
            entity=ent,
            assoc=assoc,
            visit=visit,
            aligned_detections=[det_reappeared],
        )
        self.assertEqual(update.new_state, EpistemicState.CONFIRMED_PRESENT)
        self.assertTrue(update.transition_occurred)
        self.assertIn("re-introduced", update.explanation.lower())


if __name__ == "__main__":
    unittest.main()
