"""
Unit tests for Boundary Clipping, Registration Degradation, and Movement Decisions (Phase 9B).
Controlled experiments:
- Genuine 40px, 60px, 90px movement near an image edge.
- Camera crop without object movement.
- Poor registration quality (gated on inliers < 20%).
- Partial object visibility and canonical overlap gating.
- Return-to-baseline near a boundary.
"""

from __future__ import annotations

import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from objectmemory.domain.models import BoundingBox, WorkbenchEntity
from objectmemory.domain.states import EpistemicState
from objectmemory.temporal.models import TemporalEngineConfig, VisitInput
from objectmemory.temporal.engine import TemporalMemoryEngine
from objectmemory.persistence.repository import SQLiteMemoryStore
from objectmemory.registration.models import RegistrationResult, AlignmentStatus
from objectmemory.detection.models import Detection


class TestBoundaryAndRegistration(unittest.TestCase):
    def setUp(self) -> None:
        self.store = SQLiteMemoryStore(":memory:")
        self.config = TemporalEngineConfig(
            movement_displacement_threshold=35.0,
            border_margin_px=12.0,
            clipped_movement_displacement_threshold=100.0,
            min_overlap_fraction=0.70,
            min_registration_inlier_ratio=0.20,
        )
        self.engine = TemporalMemoryEngine(store=self.store, config=self.config)

    def _make_visit(
        self,
        session_id: str,
        inlier_ratio: float = 0.95,
        num_inliers: int = 300,
        homography: Optional[np.ndarray] = None,
        overlap_mask: Optional[np.ndarray] = None,
        image_size: tuple[int, int] = (800, 600),
    ) -> VisitInput:
        w, h = image_size
        if homography is None:
            homography = np.eye(3, dtype=np.float64)
        if overlap_mask is None:
            overlap_mask = np.ones((h, w), dtype=np.uint8) * 255

        reg = RegistrationResult(
            success=True,
            status=AlignmentStatus.SUCCESS,
            method_used="CONTROLLED_TEST",
            ref_size=(800, 600),
            query_size=(w, h),
            homography=homography,
            inlier_ratio=inlier_ratio,
            num_inliers=num_inliers,
            overlap_mask=overlap_mask,
        )
        return VisitInput(
            session_id=session_id,
            session_name=session_id,
            timestamp="2026-10-21T00:00:00Z",
            image=np.zeros((h, w, 3), dtype=np.uint8),
            registration=reg,
            detections=[],
        )

    def test_poor_registration_suppresses_movement_and_removal(self) -> None:
        """Degraded registration (inlier ratio < 20%) must reject spatial comparison."""
        box_init = BoundingBox(100.0, 100.0, 200.0, 200.0)
        ent = WorkbenchEntity(
            entity_id="test_cup",
            label="cup",
            current_state=EpistemicState.CONFIRMED_PRESENT,
            canonical_bbox=box_init,
            created_at="2026-10-21T00:00:00Z",
            metadata={"initial_canonical_bbox": [100.0, 100.0, 200.0, 200.0]},
        )
        self.store.add_entity(ent)

        # Move 150px with degraded registration (inlier ratio = 0.12 < 0.20)
        box_moved = BoundingBox(250.0, 100.0, 350.0, 200.0)
        det = Detection(label="cup", confidence=0.9, raw_bbox=box_moved, canonical_bbox=box_moved)
        visit_poor = self._make_visit("v_poor", inlier_ratio=0.12, num_inliers=25)

        res = self.engine._evaluate_displacement(
            entity=ent,
            det=det,
            visit=visit_poor,
            initial_bbox=box_init,
            last_reliable_bbox=box_init,
            det_canon=box_moved,
        )
        # Spatial comparison must be marked unreliable and state preserved
        self.assertFalse(res[4])  # is_spatial_reliable
        self.assertEqual(res[5], EpistemicState.CONFIRMED_PRESENT)
        self.assertIn("Registration unverified or failed", res[6])

    def test_camera_crop_suppressed_by_canonical_overlap(self) -> None:
        """Camera crop that cuts off an object's footprint (< 70% overlap) is flagged unverified."""
        # Object near right edge: x from 700 to 790 (width 90)
        box_init = BoundingBox(700.0, 200.0, 790.0, 300.0)
        ent = WorkbenchEntity(
            entity_id="book_edge",
            label="book",
            current_state=EpistemicState.CONFIRMED_PRESENT,
            canonical_bbox=box_init,
            created_at="2026-10-21T00:00:00Z",
            metadata={"initial_canonical_bbox": [700.0, 200.0, 790.0, 300.0]},
        )
        self.store.add_entity(ent)

        # In query frame, camera cropped canvas at x=730 (only 30px of 90px width visible = 33% overlap)
        mask_cropped = np.zeros((600, 800), dtype=np.uint8)
        mask_cropped[:, :730] = 255
        visit_cropped = self._make_visit("v_crop", overlap_mask=mask_cropped, image_size=(730, 600))

        # Detector only sees truncated portion [700, 200, 725, 300], shifting centroid
        box_trunc = BoundingBox(700.0, 200.0, 725.0, 300.0)
        det_trunc = Detection(label="book", confidence=0.85, raw_bbox=box_trunc, canonical_bbox=box_trunc)

        res = self.engine._evaluate_displacement(
            entity=ent,
            det=det_trunc,
            visit=visit_cropped,
            initial_bbox=box_init,
            last_reliable_bbox=box_init,
            det_canon=box_trunc,
        )
        # Must be flagged unverified due to low canonical overlap
        self.assertFalse(res[4])  # is_spatial_reliable
        self.assertEqual(res[5], EpistemicState.CONFIRMED_PRESENT)
        self.assertIn("outside registered overlap", res[6])

    def test_return_to_baseline_near_boundary(self) -> None:
        """An object returning to its baseline position near an image border is correctly identified."""
        box_base = BoundingBox(15.0, 100.0, 95.0, 200.0)  # Near left border
        box_moved = BoundingBox(300.0, 100.0, 380.0, 200.0)  # Moved to interior
        box_returned = BoundingBox(16.0, 101.0, 96.0, 201.0)  # Returned to baseline

        ent = WorkbenchEntity(
            entity_id="cup_boundary",
            label="cup",
            current_state=EpistemicState.CONFIRMED_MOVED,  # Previously moved!
            canonical_bbox=box_moved,
            created_at="2026-10-21T00:00:00Z",
            metadata={
                "initial_canonical_bbox": [15.0, 100.0, 95.0, 200.0],
                "last_reliable_canonical_bbox": [300.0, 100.0, 380.0, 200.0],
            },
        )
        self.store.add_entity(ent)

        visit = self._make_visit("v_return")
        det_ret = Detection(label="cup", confidence=0.92, raw_bbox=box_returned, canonical_bbox=box_returned)

        c_disp, b_disp, is_c_mov, is_ret, is_rel, target_st, expl = self.engine._evaluate_displacement(
            entity=ent,
            det=det_ret,
            visit=visit,
            initial_bbox=box_base,
            last_reliable_bbox=box_moved,
            det_canon=box_returned,
        )
        self.assertTrue(is_c_mov)  # Consecutive move of ~284px
        self.assertTrue(is_ret)    # Return to baseline flag is TRUE
        self.assertEqual(target_st, EpistemicState.CONFIRMED_PRESENT)
        self.assertLess(b_disp, 5.0)  # Baseline displacement < 5px


    def test_genuine_movement_from_boundary_to_interior(self) -> None:
        """Genuine 40px, 60px, 90px movements from boundary into interior must be detected as RELOCATED."""
        # Baseline near border: [700, 200, 790, 300] (within 10px of 800)
        box_init = BoundingBox(700.0, 200.0, 790.0, 300.0)
        ent = WorkbenchEntity(
            entity_id="cup_moving_inward",
            label="cup",
            current_state=EpistemicState.CONFIRMED_PRESENT,
            canonical_bbox=box_init,
            created_at="2026-10-21T00:00:00Z",
            metadata={"initial_canonical_bbox": [700.0, 200.0, 790.0, 300.0]},
        )
        self.store.add_entity(ent)
        visit = self._make_visit("v_inward")

        for shift in [40.0, 60.0, 90.0]:
            # Move object leftward away from border into interior (no longer clipped)
            box_moved = BoundingBox(700.0 - shift, 200.0, 790.0 - shift, 300.0)
            det = Detection(label="cup", confidence=0.92, raw_bbox=box_moved, canonical_bbox=box_moved)

            c_disp, b_disp, is_c_mov, is_ret, is_rel, target_st, expl = self.engine._evaluate_displacement(
                entity=ent,
                det=det,
                visit=visit,
                initial_bbox=box_init,
                last_reliable_bbox=box_init,
                det_canon=box_moved,
            )
            self.assertTrue(is_c_mov, f"Shift {shift}px should be recognized as consecutive movement")
            self.assertEqual(target_st, EpistemicState.CONFIRMED_MOVED, f"Shift {shift}px should transition to CONFIRMED_MOVED")
            self.assertFalse(is_ret)
            self.assertAlmostEqual(b_disp, shift, delta=0.5)


if __name__ == "__main__":
    unittest.main()
