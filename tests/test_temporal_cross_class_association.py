"""
Regression tests for Conservative Cross-Class Object Association in ObjectMemory.

Validates:
1. Book classified as laptop in one visit, then book again (single entity preserved).
2. Same-class association remains unchanged.
3. Two different objects with overlapping boxes are not incorrectly merged.
4. Ambiguous cross-class candidates are rejected.
5. Unreliable registration blocks spatial fallback.
6. Multiple candidates do not produce duplicate or conflicting assignments.
7. Existing entity history remains intact.
8. Missing appearance descriptors do not cause invented similarity scores.
"""

from __future__ import annotations

import tempfile
import unittest
import numpy as np

from objectmemory.detection.models import Detection, VisualSignature
from objectmemory.domain.models import BoundingBox, WorkbenchEntity
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
    elif split == 2:
        vec[:37] = 1.0
        vec[74:111] = 1.0
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


class TestCrossClassAssociation(unittest.TestCase):
    """Unit and integration tests for conservative cross-class association."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.store = SQLiteMemoryStore(self.tmp.name)
        self.store.__enter__()
        self.config = TemporalEngineConfig(
            min_registration_inlier_ratio=0.20,
            border_margin_px=8.0,
            movement_displacement_threshold=35.0,
            enable_cross_class_association=True,
            cross_class_iou_threshold=0.35,
            cross_class_signature_similarity_threshold=0.70,
            cross_class_fallback_iou_threshold=0.75,
            cross_class_ambiguity_margin=0.15,
        )
        self.associator = CrossVisitEntityAssociator(config=self.config)
        self.engine = TemporalMemoryEngine(store=self.store, config=self.config)
        self.img_h, self.img_w = 600, 800
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

    def test_1_book_classified_as_laptop_then_book(self):
        """1. Book classified as laptop in Visit 2, then book in Visit 3: single entity preserved."""
        box = BoundingBox(300.0, 300.0, 600.0, 500.0)
        sig_book = _make_dummy_sig(split=0)

        # Visit 1: Discovered as 'book'
        det1 = Detection("book", 0.90, box, canonical_bbox=box, signature=sig_book)
        v1 = VisitInput("v1", "Visit 1", "2026-03-01T10:00:00Z", self.dummy_img, self.reg_success, [det1])
        rep1 = self.engine.process_visit(v1)
        self.assertEqual(rep1.new_entities_created, 1)
        ent1 = self.store.list_entities()[0]
        self.assertEqual(ent1.label, "book")

        # Visit 2: Same physical book detected as 'laptop' (conf 0.55), same appearance and location
        det2 = Detection("laptop", 0.55, box, canonical_bbox=box, signature=sig_book)
        v2 = VisitInput("v2", "Visit 2", "2026-03-02T10:00:00Z", self.dummy_img, self.reg_success, [det2])
        rep2 = self.engine.process_visit(v2)
        # Spurious laptop entity must NOT be created
        self.assertEqual(rep2.new_entities_created, 0)
        entities = self.store.list_entities()
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0].entity_id, ent1.entity_id)
        self.assertEqual(entities[0].label, "book")  # Established label preserved

        # Check observation recorded classification mismatch
        obs = self.store.list_observations_for_session("v2")[0]
        self.assertTrue(obs.raw_evidence.get("is_cross_class"))
        self.assertEqual(obs.raw_evidence.get("detected_label"), "laptop")
        self.assertEqual(obs.raw_evidence.get("established_label"), "book")

        # Visit 3: Detected as 'book' again
        det3 = Detection("book", 0.88, box, canonical_bbox=box, signature=sig_book)
        v3 = VisitInput("v3", "Visit 3", "2026-03-03T10:00:00Z", self.dummy_img, self.reg_success, [det3])
        rep3 = self.engine.process_visit(v3)
        self.assertEqual(rep3.new_entities_created, 0)
        entities = self.store.list_entities()
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0].label, "book")

    def test_2_same_class_association_remains_unchanged(self):
        """2. Same-class association proceeds through primary pass without cross-class flags."""
        box = BoundingBox(100.0, 100.0, 200.0, 200.0)
        sig = _make_dummy_sig(split=0)
        ent = WorkbenchEntity("ent_cup", "cup", "2026-03-01T10:00:00Z", box)
        det = Detection("cup", 0.92, box, canonical_bbox=box, signature=sig)

        matches, unmatched = self.associator.associate(
            entities=[ent],
            detections=[det],
            registration=self.reg_success,
            entity_signatures={"ent_cup": sig},
        )
        self.assertEqual(len(matches), 1)
        self.assertTrue(matches[0].is_associated)
        self.assertFalse(matches[0].is_cross_class)
        self.assertEqual(len(unmatched), 0)

    def test_3_two_different_objects_overlapping_not_merged(self):
        """3. Two different objects with overlapping boxes but different appearances are not merged."""
        box_ent = BoundingBox(100.0, 100.0, 200.0, 200.0)
        box_det = BoundingBox(120.0, 100.0, 220.0, 200.0)  # IoU ~ 0.67
        sig_tool = _make_dummy_sig(split=0)
        sig_cable = _make_dummy_sig(split=1)  # Orthogonal appearance (cosine sim = 0.0)

        ent = WorkbenchEntity("ent_tool", "tool", "2026-03-01T10:00:00Z", box_ent)
        det = Detection("cable", 0.85, box_det, canonical_bbox=box_det, signature=sig_cable)

        matches, unmatched = self.associator.associate(
            entities=[ent],
            detections=[det],
            registration=self.reg_success,
            entity_signatures={"ent_tool": sig_tool},
        )
        self.assertEqual(len(matches), 1)
        self.assertFalse(matches[0].is_associated)
        self.assertEqual(len(unmatched), 1)  # Cable remains unassociated (eligible for new entity)

    def test_4_ambiguous_cross_class_candidates_rejected(self):
        """4. Ambiguous competing cross-class candidates are rejected (no accidental merger)."""
        box_ent = BoundingBox(200.0, 200.0, 300.0, 300.0)
        sig_common = _make_dummy_sig(split=0)

        ent = WorkbenchEntity("ent_book", "book", "2026-03-01T10:00:00Z", box_ent)
        # Two identical laptop detections equidistant to book
        det1 = Detection("laptop", 0.60, BoundingBox(200.0, 200.0, 300.0, 300.0), canonical_bbox=BoundingBox(200.0, 200.0, 300.0, 300.0), signature=sig_common)
        det2 = Detection("laptop", 0.60, BoundingBox(205.0, 205.0, 305.0, 305.0), canonical_bbox=BoundingBox(205.0, 205.0, 305.0, 305.0), signature=sig_common)

        matches, unmatched = self.associator.associate(
            entities=[ent],
            detections=[det1, det2],
            registration=self.reg_success,
            entity_signatures={"ent_book": sig_common},
        )
        self.assertEqual(len(matches), 1)
        self.assertFalse(matches[0].is_associated)
        self.assertTrue(matches[0].is_ambiguous)

    def test_5_unreliable_registration_blocks_spatial_fallback(self):
        """5. Unreliable registration blocks spatial cross-class fallback."""
        box = BoundingBox(100.0, 100.0, 200.0, 200.0)
        sig = _make_dummy_sig(split=0)

        ent = WorkbenchEntity("ent_book", "book", "2026-03-01T10:00:00Z", box)
        det = Detection("laptop", 0.60, box, canonical_bbox=box, signature=sig)

        reg_fail = RegistrationResult(
            success=False,
            status=AlignmentStatus.INSUFFICIENT_INLIERS,
            method_used="SIFT",
            ref_size=(self.img_w, self.img_h),
            query_size=(self.img_w, self.img_h),
            inlier_ratio=0.10,
        )

        matches, unmatched = self.associator.associate(
            entities=[ent],
            detections=[det],
            registration=reg_fail,
            entity_signatures={"ent_book": sig},
        )
        self.assertEqual(len(matches), 1)
        self.assertFalse(matches[0].is_associated)
        self.assertEqual(len(unmatched), 1)

    def test_6_multiple_candidates_no_conflicting_assignments(self):
        """6. Multiple unmatched candidates produce globally optimal 1:1 assignment without collisions."""
        box_a = BoundingBox(100.0, 100.0, 200.0, 200.0)
        box_b = BoundingBox(400.0, 400.0, 500.0, 500.0)
        sig_a = _make_dummy_sig(split=0)
        sig_b = _make_dummy_sig(split=1)

        e1 = WorkbenchEntity("ent_book", "book", "2026-03-01T10:00:00Z", box_a)
        e2 = WorkbenchEntity("ent_box", "box", "2026-03-01T10:00:00Z", box_b)

        d1 = Detection("laptop", 0.65, box_a, canonical_bbox=box_a, signature=sig_a)  # matches e1
        d2 = Detection("crate", 0.65, box_b, canonical_bbox=box_b, signature=sig_b)   # matches e2

        matches, unmatched = self.associator.associate(
            entities=[e1, e2],
            detections=[d1, d2],
            registration=self.reg_success,
            entity_signatures={"ent_book": sig_a, "ent_box": sig_b},
        )
        self.assertEqual(len(matches), 2)
        m_book = next(m for m in matches if m.entity_id == "ent_book")
        m_box = next(m for m in matches if m.entity_id == "ent_box")
        self.assertTrue(m_book.is_associated)
        self.assertTrue(m_book.is_cross_class)
        self.assertEqual(m_book.detection.label, "laptop")

        self.assertTrue(m_box.is_associated)
        self.assertTrue(m_box.is_cross_class)
        self.assertEqual(m_box.detection.label, "crate")
        self.assertEqual(len(unmatched), 0)

    def test_7_existing_entity_history_remains_intact(self):
        """7. Existing entity history and previous observations remain intact after cross-class match."""
        box = BoundingBox(100.0, 100.0, 200.0, 200.0)
        sig = _make_dummy_sig(split=0)

        # Visit 1
        det1 = Detection("book", 0.90, box, canonical_bbox=box, signature=sig)
        self.engine.process_visit(VisitInput("v1", "Visit 1", "2026-03-01T10:00:00Z", self.dummy_img, self.reg_success, [det1]))

        # Visit 2 (Cross-class match as 'laptop')
        det2 = Detection("laptop", 0.58, box, canonical_bbox=box, signature=sig)
        self.engine.process_visit(VisitInput("v2", "Visit 2", "2026-03-02T10:00:00Z", self.dummy_img, self.reg_success, [det2]))

        ent = self.store.list_entities()[0]
        obs = self.store.list_observations_for_entity(ent.entity_id)
        self.assertEqual(len(obs), 2)
        self.assertEqual(obs[0].session_id, "v1")
        self.assertEqual(obs[1].session_id, "v2")
        # Established label has not been overwritten
        self.assertEqual(ent.label, "book")

    def test_8_missing_appearance_descriptors_no_invented_similarity(self):
        """8. Missing appearance descriptors do not invent similarity scores."""
        box = BoundingBox(100.0, 100.0, 200.0, 200.0)
        ent = WorkbenchEntity("ent_book", "book", "2026-03-01T10:00:00Z", box)

        # Detection has NO signature (None)
        # Case A: Low IoU (0.40 < fallback 0.75) -> rejected
        det_low = Detection("laptop", 0.60, BoundingBox(140.0, 100.0, 240.0, 200.0), canonical_bbox=BoundingBox(140.0, 100.0, 240.0, 200.0), signature=None)
        matches, unmatched = self.associator.associate(
            entities=[ent],
            detections=[det_low],
            registration=self.reg_success,
            entity_signatures={},
        )
        self.assertFalse(matches[0].is_associated)
        self.assertEqual(len(unmatched), 1)

        # Case B: High spatial overlap (IoU >= 0.75) -> accepted with signature_similarity = None (not fabricated!)
        det_high = Detection("laptop", 0.60, BoundingBox(102.0, 100.0, 202.0, 200.0), canonical_bbox=BoundingBox(102.0, 100.0, 202.0, 200.0), signature=None)
        matches_high, unmatched_high = self.associator.associate(
            entities=[ent],
            detections=[det_high],
            registration=self.reg_success,
            entity_signatures={},
        )
        self.assertTrue(matches_high[0].is_associated)
        self.assertTrue(matches_high[0].is_cross_class)
        self.assertIsNone(matches_high[0].signature_similarity)  # Must be None, NOT fabricated float
        self.assertEqual(len(unmatched_high), 0)


if __name__ == "__main__":
    unittest.main()
