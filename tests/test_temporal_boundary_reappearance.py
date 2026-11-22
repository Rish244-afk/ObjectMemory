"""
Regression tests for Phase 5 Temporal Memory Engine:
Part A: Boundary-related false movement suppression and edge reliability.
Part B: Reappearance and displacement baselines (initial vs last reliable).
"""

from __future__ import annotations

import tempfile
import unittest
import numpy as np

from objectmemory.detection.models import Detection, VisualSignature
from objectmemory.domain.models import BoundingBox
from objectmemory.domain.states import EpistemicState, VisibilityStatus
from objectmemory.persistence.repository import SQLiteMemoryStore
from objectmemory.registration.models import AlignmentStatus, RegistrationResult
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


class TestTemporalBoundaryAndReappearance(unittest.TestCase):
    """
    Tests covering boundary-related false movement suppression,
    baseline displacement references, and reappearance state logic.
    """

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.store = SQLiteMemoryStore(self.tmp.name)
        self.store.__enter__()
        self.config = TemporalEngineConfig(
            min_registration_inlier_ratio=0.20,
            border_margin_px=8.0,
            movement_displacement_threshold=35.0,
            clipped_movement_displacement_threshold=100.0,
            max_drift_step_displacement=20.0,
            drift_iou_threshold=0.35,
            excluded_background_classes=["dining table"],
        )
        self.engine = TemporalMemoryEngine(store=self.store, config=self.config)
        self.img_h, self.img_w = 700, 800
        self.dummy_img = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        self.reg_success = RegistrationResult(
            success=True,
            status=AlignmentStatus.SUCCESS,
            method_used="SIFT",
            ref_size=(self.img_w, self.img_h),
            query_size=(self.img_w, self.img_h),
            homography=np.eye(3, dtype=np.float64),
            overlap_mask=np.ones((self.img_h, self.img_w), dtype=np.uint8) * 255,
            aligned_image=np.ones((self.img_h, self.img_w, 3), dtype=np.uint8) * 120,
            inlier_ratio=0.85,
        )

    def tearDown(self):
        self.store.__exit__(None, None, None)
        try:
            import os
            os.unlink(self.tmp.name)
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # PART A — Boundary-Related Movement Tests
    # -------------------------------------------------------------------------

    def test_part_a_stationary_edge_clipped_object_crop_shift(self):
        """Stationary edge-clipped object with a crop shift does not trigger CONFIRMED_MOVED."""
        # Visit 1: Edge-clipped object touching left border (x_min = 2.0 <= 8.0)
        base_box = BoundingBox(2.0, 100.0, 102.0, 200.0)
        sig = _make_dummy_sig(split=0)
        det1 = Detection("cup", 0.90, base_box, canonical_bbox=base_box, signature=sig)
        self.engine.process_visit(VisitInput(
            "v1", "Visit 1", "2026-03-01T10:00:00Z",
            self.dummy_img, self.reg_success, [det1],
        ))

        # Visit 2: Camera crop shift results in apparent 50px displacement (< 100px)
        shifted_raw = BoundingBox(4.0, 100.0, 104.0, 200.0)
        shifted_canon = BoundingBox(52.0, 100.0, 152.0, 200.0)
        det2 = Detection("cup", 0.88, shifted_raw, canonical_bbox=shifted_canon, signature=sig)
        rep2 = self.engine.process_visit(VisitInput(
            "v2", "Visit 2", "2026-03-02T10:00:00Z",
            self.dummy_img, self.reg_success, [det2],
        ))

        ent = self.store.list_entities()[0]
        self.assertEqual(ent.current_state, EpistemicState.CONFIRMED_PRESENT)
        self.assertFalse(rep2.entity_updates[0].transition_occurred)
        self.assertEqual(rep2.entity_updates[0].visibility_status, VisibilityStatus.UNCERTAIN)
        # Canonical position must be preserved, not overwritten with unreliable shifted coordinates
        self.assertEqual(ent.canonical_bbox, base_box)
        events = self.store.list_change_events_for_entity(ent.entity_id)
        self.assertEqual(len(events), 1)  # Only initial discovery

    def test_part_a_stationary_interior_object(self):
        """Stationary interior object remains CONFIRMED_PRESENT with CLEARLY_VISIBLE status."""
        box1 = BoundingBox(200.0, 200.0, 300.0, 300.0)
        sig = _make_dummy_sig(split=0)
        det1 = Detection("mug", 0.95, box1, canonical_bbox=box1, signature=sig)
        self.engine.process_visit(VisitInput(
            "v1", "Visit 1", "2026-03-01T10:00:00Z",
            self.dummy_img, self.reg_success, [det1],
        ))

        # Visit 2: Stationary (< 35px shift)
        box2 = BoundingBox(205.0, 202.0, 305.0, 302.0)
        det2 = Detection("mug", 0.94, box2, canonical_bbox=box2, signature=sig)
        rep2 = self.engine.process_visit(VisitInput(
            "v2", "Visit 2", "2026-03-02T10:00:00Z",
            self.dummy_img, self.reg_success, [det2],
        ))

        ent = self.store.list_entities()[0]
        self.assertEqual(ent.current_state, EpistemicState.CONFIRMED_PRESENT)
        self.assertFalse(rep2.entity_updates[0].transition_occurred)
        self.assertEqual(rep2.entity_updates[0].visibility_status, VisibilityStatus.CLEARLY_VISIBLE)

    def test_part_a_genuine_movement_unclipped_object(self):
        """Genuine movement of an unclipped object correctly transitions to CONFIRMED_MOVED."""
        box1 = BoundingBox(100.0, 100.0, 200.0, 200.0)
        sig = _make_dummy_sig(split=0)
        det1 = Detection("cup", 0.92, box1, canonical_bbox=box1, signature=sig)
        self.engine.process_visit(VisitInput(
            "v1", "Visit 1", "2026-03-01T10:00:00Z",
            self.dummy_img, self.reg_success, [det1],
        ))

        # Visit 2: Relocated by 300px across the workbench
        box2 = BoundingBox(400.0, 300.0, 500.0, 400.0)
        det2 = Detection("cup", 0.91, box2, canonical_bbox=box2, signature=sig)
        rep2 = self.engine.process_visit(VisitInput(
            "v2", "Visit 2", "2026-03-02T10:00:00Z",
            self.dummy_img, self.reg_success, [det2],
        ))

        ent = self.store.list_entities()[0]
        self.assertEqual(ent.current_state, EpistemicState.CONFIRMED_MOVED)
        self.assertTrue(rep2.entity_updates[0].transition_occurred)
        self.assertEqual(rep2.entity_updates[0].visibility_status, VisibilityStatus.CLEARLY_VISIBLE)
        self.assertEqual(ent.canonical_bbox, box2)

    def test_part_a_clipped_object_insufficient_evidence(self):
        """A clipped object with displacement under threshold preserves uncertainty and position."""
        box1 = BoundingBox(150.0, 150.0, 250.0, 250.0)
        sig = _make_dummy_sig(split=0)
        det1 = Detection("laptop", 0.90, box1, canonical_bbox=box1, signature=sig)
        self.engine.process_visit(VisitInput(
            "v1", "Visit 1", "2026-03-01T10:00:00Z",
            self.dummy_img, self.reg_success, [det1],
        ))

        # Visit 2: Raw box clipped at right edge (x_max = 796 >= 800 - 8) with apparent 60px shift (< 100px)
        raw2 = BoundingBox(695.0, 150.0, 796.0, 250.0)
        canon2 = BoundingBox(210.0, 150.0, 310.0, 250.0)
        det2 = Detection("laptop", 0.89, raw2, canonical_bbox=canon2, signature=sig)
        rep2 = self.engine.process_visit(VisitInput(
            "v2", "Visit 2", "2026-03-02T10:00:00Z",
            self.dummy_img, self.reg_success, [det2],
        ))

        ent = self.store.list_entities()[0]
        self.assertEqual(ent.current_state, EpistemicState.CONFIRMED_PRESENT)
        self.assertEqual(rep2.entity_updates[0].visibility_status, VisibilityStatus.UNCERTAIN)
        self.assertEqual(ent.canonical_bbox, box1)

    def test_part_a_registration_failure_preserving_last_reliable_position(self):
        """Failed registration preserves the last reliable position and records UNCERTAIN visibility."""
        box1 = BoundingBox(150.0, 150.0, 250.0, 250.0)
        sig = _make_dummy_sig(split=0)
        det1 = Detection("tool", 0.91, box1, canonical_bbox=box1, signature=sig)
        self.engine.process_visit(VisitInput(
            "v1", "Visit 1", "2026-03-01T10:00:00Z",
            self.dummy_img, self.reg_success, [det1],
        ))

        # Visit 2: Registration failure
        reg_fail = RegistrationResult(
            success=False,
            status=AlignmentStatus.INSUFFICIENT_INLIERS,
            method_used="SIFT",
            ref_size=(self.img_w, self.img_h),
            query_size=(self.img_w, self.img_h),
            homography=None,
            inlier_ratio=0.10,
        )
        det2 = Detection("tool", 0.88, BoundingBox(180.0, 180.0, 280.0, 280.0), signature=sig)
        rep2 = self.engine.process_visit(VisitInput(
            "v2", "Visit 2", "2026-03-02T10:00:00Z",
            self.dummy_img, reg_fail, [det2],
        ))

        ent = self.store.list_entities()[0]
        self.assertEqual(ent.current_state, EpistemicState.CONFIRMED_PRESENT)
        self.assertEqual(rep2.entity_updates[0].visibility_status, VisibilityStatus.UNCERTAIN)
        self.assertEqual(ent.canonical_bbox, box1)

    def test_part_a_full_frame_dining_table_excluded_from_movement(self):
        """Environmental support class 'dining table' is excluded from movement transitions."""
        table_box1 = BoundingBox(1.0, 200.0, 798.0, 698.0)
        sig = _make_dummy_sig(split=0)
        det1 = Detection("dining table", 0.85, table_box1, canonical_bbox=table_box1, signature=sig)
        self.engine.process_visit(VisitInput(
            "v1", "Visit 1", "2026-03-01T10:00:00Z",
            self.dummy_img, self.reg_success, [det1],
        ))

        # Visit 2: Framing shift causes 55px centroid shift
        table_box2 = BoundingBox(1.0, 240.0, 798.0, 698.0)
        table_canon2 = BoundingBox(15.0, 240.0, 790.0, 698.0)
        det2 = Detection("dining table", 0.82, table_box2, canonical_bbox=table_canon2, signature=sig)
        rep2 = self.engine.process_visit(VisitInput(
            "v2", "Visit 2", "2026-03-02T10:00:00Z",
            self.dummy_img, self.reg_success, [det2],
        ))

        ent = self.store.list_entities()[0]
        self.assertEqual(ent.current_state, EpistemicState.CONFIRMED_PRESENT)
        self.assertFalse(rep2.entity_updates[0].transition_occurred)

    # -------------------------------------------------------------------------
    # PART B — Reappearance and Displacement Baseline Tests
    # -------------------------------------------------------------------------

    def test_part_b_scissors_like_reappearance_near_last_reliable_position(self):
        """
        Reappearance within 15-20px of the last reliable position restores CONFIRMED_PRESENT,
        even if accumulated drift from Day 1 initial position is > 35px.
        """
        sig = _make_dummy_sig(split=0)
        # Visit 1: Initial position at center (350.5, 539.5)
        init_box = BoundingBox(210.0, 489.0, 491.0, 590.0)
        det1 = Detection("scissors", 0.90, init_box, canonical_bbox=init_box, signature=sig)
        self.engine.process_visit(VisitInput(
            "v1", "Visit 1", "2026-03-01T10:00:00Z",
            self.dummy_img, self.reg_success, [det1],
        ))

        # Visit 2: Small reliable shift (center 319.55, 536.45, 31.1px from init <= 35px)
        v2_box = BoundingBox(184.7, 474.8, 454.4, 598.1)
        det2 = Detection("scissors", 0.89, v2_box, canonical_bbox=v2_box, signature=sig)
        self.engine.process_visit(VisitInput(
            "v2", "Visit 2", "2026-03-02T10:00:00Z",
            self.dummy_img, self.reg_success, [det2],
        ))

        # Visit 3: Missing / vacant patch -> transitions to PENDING_REMOVAL
        self.engine.process_visit(VisitInput(
            "v3", "Visit 3", "2026-03-03T10:00:00Z",
            self.dummy_img, self.reg_success, [],  # Zero detections
        ))
        ent = self.store.list_entities()[0]
        self.assertEqual(ent.current_state, EpistemicState.PENDING_REMOVAL)

        # Visit 4: Reappears at center (311.5, 549.6)
        # Distance to V2 (last reliable): 15.4px (<= 35px)
        # Distance to V1 (initial): 40.3px (> 35px)
        v4_box = BoundingBox(183.3, 490.7, 439.7, 608.5)
        det4 = Detection("scissors", 0.91, v4_box, canonical_bbox=v4_box, signature=sig)
        rep4 = self.engine.process_visit(VisitInput(
            "v4", "Visit 4", "2026-03-04T10:00:00Z",
            self.dummy_img, self.reg_success, [det4],
        ))

        ent = self.store.list_entities()[0]
        # Must restore to CONFIRMED_PRESENT, NOT CONFIRMED_MOVED
        self.assertEqual(ent.current_state, EpistemicState.CONFIRMED_PRESENT)
        self.assertTrue(rep4.entity_updates[0].transition_occurred)
        self.assertEqual(rep4.entity_updates[0].new_state, EpistemicState.CONFIRMED_PRESENT)

    def test_part_b_genuine_reappearance_at_clearly_different_position(self):
        """Reappearance at a clearly different position (> 35px from last and initial) transitions to CONFIRMED_MOVED."""
        sig = _make_dummy_sig(split=0)
        box1 = BoundingBox(100.0, 100.0, 200.0, 200.0)
        det1 = Detection("scissors", 0.90, box1, canonical_bbox=box1, signature=sig)
        self.engine.process_visit(VisitInput(
            "v1", "Visit 1", "2026-03-01T10:00:00Z",
            self.dummy_img, self.reg_success, [det1],
        ))

        # Visit 2: Missing -> PENDING_REMOVAL
        self.engine.process_visit(VisitInput(
            "v2", "Visit 2", "2026-03-02T10:00:00Z",
            self.dummy_img, self.reg_success, [],
        ))
        ent = self.store.list_entities()[0]
        self.assertEqual(ent.current_state, EpistemicState.PENDING_REMOVAL)

        # Visit 3: Reappears 200px away
        box3 = BoundingBox(350.0, 300.0, 450.0, 400.0)
        det3 = Detection("scissors", 0.88, box3, canonical_bbox=box3, signature=sig)
        rep3 = self.engine.process_visit(VisitInput(
            "v3", "Visit 3", "2026-03-03T10:00:00Z",
            self.dummy_img, self.reg_success, [det3],
        ))

        ent = self.store.list_entities()[0]
        self.assertEqual(ent.current_state, EpistemicState.CONFIRMED_MOVED)
        self.assertEqual(rep3.entity_updates[0].new_state, EpistemicState.CONFIRMED_MOVED)

    def test_part_b_gradual_coordinate_drift_without_physical_movement(self):
        """Small per-visit drift that accumulates past 35px while retaining IoU >= 0.35 does NOT trigger CONFIRMED_MOVED."""
        sig = _make_dummy_sig(split=0)
        box1 = BoundingBox(100.0, 100.0, 200.0, 200.0)
        det1 = Detection("item", 0.92, box1, canonical_bbox=box1, signature=sig)
        self.engine.process_visit(VisitInput(
            "v1", "Visit 1", "2026-03-01T10:00:00Z",
            self.dummy_img, self.reg_success, [det1],
        ))

        # Small 6-8px shifts over consecutive visits
        # Step shifts: (107, 106), (114, 112), (121, 118), (128, 124)
        shifts = [
            BoundingBox(107.0, 106.0, 207.0, 206.0),
            BoundingBox(114.0, 112.0, 214.0, 212.0),
            BoundingBox(121.0, 118.0, 221.0, 218.0),
            BoundingBox(128.0, 124.0, 228.0, 224.0),  # disp from init: sqrt(28^2 + 24^2) = 36.9px (> 35px)
        ]

        for i, box in enumerate(shifts, start=2):
            det = Detection("item", 0.90, box, canonical_bbox=box, signature=sig)
            rep = self.engine.process_visit(VisitInput(
                f"v{i}", f"Visit {i}", f"2026-03-0{i}T10:00:00Z",
                self.dummy_img, self.reg_success, [det],
            ))
            ent = self.store.list_entities()[0]
            # Must remain CONFIRMED_PRESENT throughout drift
            self.assertEqual(ent.current_state, EpistemicState.CONFIRMED_PRESENT)
            self.assertFalse(rep.entity_updates[0].transition_occurred)

    def test_part_b_genuinely_gradual_relocation_across_multiple_visits(self):
        """A genuinely gradual relocation where object departs its initial footprint transitions to CONFIRMED_MOVED."""
        sig = _make_dummy_sig(split=0)
        box1 = BoundingBox(100.0, 100.0, 150.0, 150.0)  # 50x50 box
        det1 = Detection("cup", 0.90, box1, canonical_bbox=box1, signature=sig)
        self.engine.process_visit(VisitInput(
            "v1", "Visit 1", "2026-03-01T10:00:00Z",
            self.dummy_img, self.reg_success, [det1],
        ))

        # Visit 2: Pushed 25px right (still partially overlaps initial: IoU = 0.50)
        box2 = BoundingBox(125.0, 100.0, 175.0, 150.0)
        det2 = Detection("cup", 0.90, box2, canonical_bbox=box2, signature=sig)
        self.engine.process_visit(VisitInput(
            "v2", "Visit 2", "2026-03-02T10:00:00Z",
            self.dummy_img, self.reg_success, [det2],
        ))
        ent = self.store.list_entities()[0]
        self.assertEqual(ent.current_state, EpistemicState.CONFIRMED_PRESENT)

        # Visit 3: Pushed another 35px right (total disp = 60px > 35px, IoU with init = 0.0 < 0.35)
        box3 = BoundingBox(160.0, 100.0, 210.0, 150.0)
        det3 = Detection("cup", 0.90, box3, canonical_bbox=box3, signature=sig)
        rep3 = self.engine.process_visit(VisitInput(
            "v3", "Visit 3", "2026-03-03T10:00:00Z",
            self.dummy_img, self.reg_success, [det3],
        ))
        ent = self.store.list_entities()[0]
        self.assertEqual(ent.current_state, EpistemicState.CONFIRMED_MOVED)
        self.assertTrue(rep3.entity_updates[0].transition_occurred)

    def test_part_b_reappearance_after_unreliable_registration(self):
        """An entity in PENDING_REMOVAL does not transition on detection when registration is unreliable."""
        sig = _make_dummy_sig(split=0)
        box1 = BoundingBox(100.0, 100.0, 200.0, 200.0)
        det1 = Detection("tool", 0.90, box1, canonical_bbox=box1, signature=sig)
        self.engine.process_visit(VisitInput(
            "v1", "Visit 1", "2026-03-01T10:00:00Z",
            self.dummy_img, self.reg_success, [det1],
        ))

        # Visit 2: Vacant -> PENDING_REMOVAL
        self.engine.process_visit(VisitInput(
            "v2", "Visit 2", "2026-03-02T10:00:00Z",
            self.dummy_img, self.reg_success, [],
        ))
        ent = self.store.list_entities()[0]
        self.assertEqual(ent.current_state, EpistemicState.PENDING_REMOVAL)

        # Visit 3: Reappears, but registration is unreliable
        reg_unreliable = RegistrationResult(
            success=True,
            status=AlignmentStatus.SUCCESS,
            method_used="SIFT",
            ref_size=(self.img_w, self.img_h),
            query_size=(self.img_w, self.img_h),
            homography=np.eye(3, dtype=np.float64),
            inlier_ratio=0.15,  # Below min 0.20
        )
        det3 = Detection("tool", 0.88, box1, canonical_bbox=box1, signature=sig)
        rep3 = self.engine.process_visit(VisitInput(
            "v3", "Visit 3", "2026-03-03T10:00:00Z",
            self.dummy_img, reg_unreliable, [det3],
        ))

        ent = self.store.list_entities()[0]
        # State must remain PENDING_REMOVAL (cannot confirm presence without reliable spatial registration)
        self.assertEqual(ent.current_state, EpistemicState.PENDING_REMOVAL)
        self.assertFalse(rep3.entity_updates[0].transition_occurred)
        self.assertEqual(rep3.entity_updates[0].visibility_status, VisibilityStatus.UNCERTAIN)


if __name__ == "__main__":
    unittest.main()
