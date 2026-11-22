"""
Independent robustness tests for ObjectMemory Phase 5 Temporal Memory Engine.

Evaluates how the temporal engine handles deliberate real-world perturbations:
1. Localization jitter (small shifts within tolerance vs shifts exceeding threshold).
2. Single-visit detection drop (detector miss / transient absence).
3. Transient false-positive detection (hallucinated detection lifecycle).
4. Visually similar objects (confusable objects within ambiguity margin).
5. Photometric perturbation (brightness and contrast variations).
6. Inaccurate or failed registration (spatial evidence gating).
7. Ground-truth isolation (guaranteeing predictions do not inspect ground truth).

Guiding principle:
Does not alter ground truth to match predictions; documents architecture limitations honestly.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np

from objectmemory.benchmarks.synthetic_dataset import (
    _draw_multimeter,
    _draw_workbench_surface,
)
from objectmemory.detection.models import Detection, VisualSignature
from objectmemory.detection.signature import VisualSignatureExtractor
from objectmemory.domain.models import BoundingBox, Session, WorkbenchEntity
from objectmemory.domain.states import BaselineChangeType, EpistemicState, VisibilityStatus
from objectmemory.persistence.repository import SQLiteMemoryStore
from objectmemory.registration.models import AlignmentStatus, RegistrationResult
from objectmemory.temporal.associator import CrossVisitEntityAssociator
from objectmemory.temporal.engine import TemporalMemoryEngine
from objectmemory.temporal.models import TemporalEngineConfig, VisitInput


def _create_synthetic_test_signature(seed: int = 1) -> VisualSignature:
    rng = np.random.default_rng(seed)
    hist = rng.uniform(0.0, 1.0, 128).astype(np.float32)
    hist /= np.linalg.norm(hist)
    grid = rng.uniform(0.0, 1.0, 20).astype(np.float32)
    grid /= np.linalg.norm(grid)
    comb = np.concatenate([hist, grid])
    comb /= np.linalg.norm(comb)
    return VisualSignature(color_histogram=hist, texture_descriptor=grid, combined_vector=comb)


class TestTemporalRobustness(unittest.TestCase):
    """Independent robustness unit tests perturbing detections, coordinates, and images."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "robustness.sqlite"
        self.store = SQLiteMemoryStore(self.db_path)
        self.config = TemporalEngineConfig(
            movement_displacement_threshold=35.0,
            signature_similarity_threshold=0.60,
            ambiguity_margin=0.10,
            vacancy_similarity_max=0.40,
            vacancy_visits_for_pending=1,
            vacancy_visits_for_removal=2,
            min_registration_inlier_ratio=0.35,
        )
        self.engine = TemporalMemoryEngine(store=self.store, config=self.config)
        self.reg_mock_valid = RegistrationResult(
            success=True,
            status=AlignmentStatus.SUCCESS,
            method_used="MOCK",
            ref_size=(640, 480),
            query_size=(640, 480),
            homography=np.eye(3, dtype=np.float64),
            overlap_mask=np.ones((480, 640), dtype=np.uint8) * 255,
            aligned_image=np.ones((480, 640, 3), dtype=np.uint8) * 120,
            inlier_ratio=0.88,
        )

    def tearDown(self):
        self.store.close()
        self.temp_dir.cleanup()

    def test_localization_jitter_small(self):
        """1A. Bounding box jitter within movement tolerance does NOT cause false CONFIRMED_MOVED."""
        base_box = BoundingBox(100.0, 100.0, 200.0, 200.0)
        sig = _create_synthetic_test_signature(seed=10)
        det1 = Detection("multimeter", 0.92, base_box, canonical_bbox=base_box, signature=sig)
        self.engine.process_visit(VisitInput("v1", "Visit 1", "2026-03-01T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), self.reg_mock_valid, [det1]))

        # Visit 2: Jitter by (dx=+8, dy=+6) -> Centroid shift = 10px <= 35px threshold
        jitter_box = BoundingBox(108.0, 106.0, 208.0, 206.0)
        det2 = Detection("multimeter", 0.90, jitter_box, canonical_bbox=jitter_box, signature=sig)
        rep2 = self.engine.process_visit(VisitInput("v2", "Visit 2", "2026-03-02T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), self.reg_mock_valid, [det2]))

        ent = self.store.list_entities()[0]
        self.assertEqual(ent.current_state, EpistemicState.CONFIRMED_PRESENT)
        self.assertFalse(rep2.entity_updates[0].transition_occurred)
        self.assertEqual(len(self.store.list_observations_for_entity(ent.entity_id)), 2)

    def test_localization_jitter_exceeding_threshold_documents_limitation(self):
        """1B. Extreme localization jitter (> 35px) documents architectural boundary."""
        base_box = BoundingBox(100.0, 100.0, 200.0, 200.0)
        sig = _create_synthetic_test_signature(seed=11)
        det1 = Detection("multimeter", 0.92, base_box, canonical_bbox=base_box, signature=sig)
        self.engine.process_visit(VisitInput("v1", "Visit 1", "2026-03-01T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), self.reg_mock_valid, [det1]))

        # Visit 2: Jitter by 40px (> 35px threshold). Architecture treats as displacement.
        jitter_box_large = BoundingBox(140.0, 100.0, 240.0, 200.0)
        det2 = Detection("multimeter", 0.90, jitter_box_large, canonical_bbox=jitter_box_large, signature=sig)
        rep2 = self.engine.process_visit(VisitInput("v2", "Visit 2", "2026-03-02T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), self.reg_mock_valid, [det2]))

        ent = self.store.list_entities()[0]
        # Documented limitation: single-frame noise > 35px triggers CONFIRMED_MOVED
        self.assertEqual(ent.current_state, EpistemicState.CONFIRMED_MOVED)
        self.assertTrue(rep2.entity_updates[0].transition_occurred)

    def test_single_visit_detection_drop_resilience(self):
        """2. Dropping one detection for a single visit does not confirm removal and recovers upon redetection."""
        base_box = BoundingBox(100.0, 100.0, 200.0, 200.0)
        sig = _create_synthetic_test_signature(seed=12)
        det = Detection("multimeter", 0.92, base_box, canonical_bbox=base_box, signature=sig)

        # Visit 1: Confirmed present
        self.engine.process_visit(VisitInput("v1", "Visit 1", "2026-03-01T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), self.reg_mock_valid, [det]))
        # Visit 2: Detection dropped
        rep2 = self.engine.process_visit(VisitInput("v2", "Visit 2", "2026-03-02T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), self.reg_mock_valid, []))
        ent = self.store.list_entities()[0]
        self.assertEqual(ent.current_state, EpistemicState.PENDING_REMOVAL)
        self.assertEqual(ent.metadata.get("vacant_streak"), 1)

        # Visit 3: Redetected
        rep3 = self.engine.process_visit(VisitInput("v3", "Visit 3", "2026-03-03T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), self.reg_mock_valid, [det]))
        ent = self.store.list_entities()[0]
        self.assertEqual(ent.current_state, EpistemicState.CONFIRMED_PRESENT)
        self.assertEqual(ent.metadata.get("vacant_streak"), 0)
        self.assertEqual(len(self.store.list_observations_for_entity(ent.entity_id)), 2)

    def test_transient_false_positive_detection_lifecycle(self):
        """3. An ephemeral false-positive detection creates a candidate and cleanly expires via pending/confirmed removal."""
        box_real = BoundingBox(100.0, 100.0, 200.0, 200.0)
        sig_real = _create_synthetic_test_signature(seed=13)
        det_real = Detection("multimeter", 0.92, box_real, canonical_bbox=box_real, signature=sig_real)
        self.engine.process_visit(VisitInput("v1", "Visit 1", "2026-03-01T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), self.reg_mock_valid, [det_real]))

        # Visit 2: Real object + spurious false positive detection in empty region
        box_fp = BoundingBox(450.0, 300.0, 550.0, 400.0)
        sig_fp = _create_synthetic_test_signature(seed=99)
        det_fp = Detection("multimeter", 0.82, box_fp, canonical_bbox=box_fp, signature=sig_fp)
        rep2 = self.engine.process_visit(VisitInput("v2", "Visit 2", "2026-03-02T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), self.reg_mock_valid, [det_real, det_fp]))
        self.assertEqual(rep2.new_entities_created, 1)

        entities = self.store.list_entities()
        self.assertEqual(len(entities), 2)
        real_ent = next(e for e in entities if e.entity_id != "ent_v2_1")
        fp_ent = next(e for e in entities if e.entity_id == "ent_v2_1")
        self.assertEqual(real_ent.current_state, EpistemicState.CONFIRMED_PRESENT)
        self.assertEqual(fp_ent.current_state, EpistemicState.CONFIRMED_PRESENT)

        # Visit 3: FP disappears -> FP enters PENDING_REMOVAL (streak=1)
        self.engine.process_visit(VisitInput("v3", "Visit 3", "2026-03-03T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), self.reg_mock_valid, [det_real]))
        fp_ent = self.store.get_entity(fp_ent.entity_id)
        self.assertEqual(fp_ent.current_state, EpistemicState.PENDING_REMOVAL)
        self.assertEqual(fp_ent.metadata.get("vacant_streak"), 1)

        # Visit 4: FP still absent -> FP transitions to CONFIRMED_REMOVED (streak=2)
        self.engine.process_visit(VisitInput("v4", "Visit 4", "2026-03-04T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), self.reg_mock_valid, [det_real]))
        fp_ent = self.store.get_entity(fp_ent.entity_id)
        self.assertEqual(fp_ent.current_state, EpistemicState.CONFIRMED_REMOVED)
        self.assertEqual(fp_ent.metadata.get("vacant_streak"), 2)

        # Real entity remained completely undisturbed
        real_ent = self.store.get_entity(real_ent.entity_id)
        self.assertEqual(real_ent.current_state, EpistemicState.CONFIRMED_PRESENT)
        self.assertEqual(len(self.store.list_observations_for_entity(real_ent.entity_id)), 4)

    def test_visually_similar_confusable_objects_ambiguity(self):
        """4. Near-identical objects competing within ambiguity margin flag AMBIGUOUS_IDENTITY without wrong assignment."""
        box1 = BoundingBox(50.0, 50.0, 100.0, 100.0)
        sig_shared = _create_synthetic_test_signature(seed=77)
        det1 = Detection("resistor", 0.92, box1, canonical_bbox=box1, signature=sig_shared)
        self.engine.process_visit(VisitInput("v1", "Visit 1", "2026-03-01T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), self.reg_mock_valid, [det1]))

        # Visit 2: Two detections with identical signature and similar coordinates
        d_cand1 = Detection("resistor", 0.90, BoundingBox(50.0, 50.0, 100.0, 100.0), canonical_bbox=BoundingBox(50.0, 50.0, 100.0, 100.0), signature=sig_shared)
        d_cand2 = Detection("resistor", 0.90, BoundingBox(54.0, 54.0, 104.0, 104.0), canonical_bbox=BoundingBox(54.0, 54.0, 104.0, 104.0), signature=sig_shared)
        rep2 = self.engine.process_visit(VisitInput("v2", "Visit 2", "2026-03-02T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), self.reg_mock_valid, [d_cand1, d_cand2]))

        ent = self.store.list_entities()[0]
        self.assertEqual(ent.current_state, EpistemicState.AMBIGUOUS_IDENTITY)
        # Crucial invariant: No observation attached to ent in visit 2
        self.assertEqual(len(self.store.list_observations_for_entity(ent.entity_id)), 1)

    def test_moderate_photometric_perturbation_preserves_association(self):
        """5A. Moderate illumination variation (brightness +/-15%) maintains HSV similarity and persistent state."""
        extractor = VisualSignatureExtractor()
        engine = TemporalMemoryEngine(store=self.store, config=self.config, signature_extractor=extractor)

        m_box = (250, 120, 360, 300)
        bbox = BoundingBox(*m_box)

        # Frame 1: Base image
        img1 = _draw_workbench_surface(640, 480, seed=123)
        _draw_multimeter(img1, *m_box)
        sig1 = extractor.extract(img1, bbox)
        det1 = Detection("multimeter", 0.94, bbox, canonical_bbox=bbox, signature=sig1)
        engine.process_visit(VisitInput("v1", "Visit 1", "2026-03-01T10:00:00Z", img1, self.reg_mock_valid, [det1]))

        # Frame 2: Realistic moderate photometric shift (alpha = 0.88, beta = 10)
        img2 = np.clip(img1.astype(np.float32) * 0.88 + 10.0, 0, 255).astype(np.uint8)
        sig2 = extractor.extract(img2, bbox)
        sim = float(sig1.cosine_similarity(sig2))
        self.assertGreater(sim, 0.60, f"Expected HSV similarity > 0.60 under moderate shift, got {sim:.3f}")

        det2 = Detection("multimeter", 0.91, bbox, canonical_bbox=bbox, signature=sig2)
        reg_v2 = RegistrationResult(
            success=True,
            status=AlignmentStatus.SUCCESS,
            method_used="MOCK",
            ref_size=(640, 480),
            query_size=(640, 480),
            homography=np.eye(3, dtype=np.float64),
            overlap_mask=np.ones((480, 640), dtype=np.uint8) * 255,
            aligned_image=img2,
            inlier_ratio=0.88,
        )
        rep2 = engine.process_visit(VisitInput("v2", "Visit 2", "2026-03-02T10:00:00Z", img2, reg_v2, [det2]))

        ent = self.store.list_entities()[0]
        self.assertEqual(ent.current_state, EpistemicState.CONFIRMED_PRESENT)
        self.assertEqual(len(self.store.list_observations_for_entity(ent.entity_id)), 2)

    def test_severe_photometric_shift_exposes_descriptor_limitation(self):
        """5B. Severe lighting shift (alpha=0.50) degrades HSV histogram similarity, exposing signature limitation."""
        extractor = VisualSignatureExtractor()
        m_box = (250, 120, 360, 300)
        bbox = BoundingBox(*m_box)

        img1 = _draw_workbench_surface(640, 480, seed=123)
        _draw_multimeter(img1, *m_box)
        sig1 = extractor.extract(img1, bbox)

        # Severe 50% lighting drop
        img_severe = np.clip(img1.astype(np.float32) * 0.50, 0, 255).astype(np.uint8)
        sig_severe = extractor.extract(img_severe, bbox)
        sim = float(sig1.cosine_similarity(sig_severe))

        # Explicitly asserts and documents that naive HSV histogram similarity falls below threshold
        self.assertLess(sim, 0.60, f"Severe lighting drop expected to degrade below 0.60, got {sim:.3f}")

    def test_inaccurate_registration_gating(self):
        """6. Low inlier ratio registration triggers spatial gating and prevents unwarranted state assertions."""
        base_box = BoundingBox(100.0, 100.0, 200.0, 200.0)
        sig = _create_synthetic_test_signature(seed=15)
        det1 = Detection("multimeter", 0.92, base_box, canonical_bbox=base_box, signature=sig)
        self.engine.process_visit(VisitInput("v1", "Visit 1", "2026-03-01T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), self.reg_mock_valid, [det1]))

        # Inaccurate registration (inlier ratio = 0.20 < min 0.35)
        reg_inaccurate = RegistrationResult(
            success=True,
            status=AlignmentStatus.SUCCESS,
            method_used="MOCK",
            ref_size=(640, 480),
            query_size=(640, 480),
            homography=np.eye(3, dtype=np.float64),
            overlap_mask=np.ones((480, 640), dtype=np.uint8) * 255,
            aligned_image=np.ones((480, 640, 3), dtype=np.uint8) * 120,
            inlier_ratio=0.20,  # Below threshold!
        )

        # Detections absent
        rep2 = self.engine.process_visit(VisitInput("v2", "Visit 2", "2026-03-02T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), reg_inaccurate, []))
        ent = self.store.list_entities()[0]
        # Absence streak must NOT progress due to untrusted registration
        self.assertEqual(ent.current_state, EpistemicState.CONFIRMED_PRESENT)

    def test_ground_truth_isolation_from_temporal_engine(self):
        """7. Guarantees temporal engine operates purely causally from VisitInput without ground truth."""
        base_box = BoundingBox(100.0, 100.0, 200.0, 200.0)
        sig = _create_synthetic_test_signature(seed=16)
        det = Detection("multimeter", 0.92, base_box, canonical_bbox=base_box, signature=sig)
        visit_input = VisitInput("v1", "Visit 1", "2026-03-01T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), self.reg_mock_valid, [det])

        # Verify VisitInput dataclass does not possess ground truth fields
        self.assertFalse(hasattr(visit_input, "ground_truth_entities"))
        self.assertFalse(hasattr(visit_input, "expected_states"))
        self.assertFalse(hasattr(det, "expected_state"))

        rep = self.engine.process_visit(visit_input)
        ent = self.store.list_entities()[0]
        self.assertEqual(ent.current_state, EpistemicState.CONFIRMED_PRESENT)

    def test_singular_homography_registration_failure_gating(self):
        """8. Singular or failed homography prevents false movement and removal claims."""
        base_box = BoundingBox(100.0, 100.0, 200.0, 200.0)
        sig = _create_synthetic_test_signature(seed=18)
        det1 = Detection("multimeter", 0.95, base_box, canonical_bbox=base_box, signature=sig)
        self.engine.process_visit(VisitInput(
            "v1", "Visit 1", "2026-03-01T10:00:00Z",
            np.zeros((480, 640, 3), dtype=np.uint8),
            self.reg_mock_valid,
            [det1]
        ))

        ent = self.store.list_entities()[0]
        self.assertEqual(ent.current_state, EpistemicState.CONFIRMED_PRESENT)

        # Visit 2: Singular homography (det ~ 0) with shifted raw detection
        singular_H = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        reg_singular = RegistrationResult(
            success=False,
            status=AlignmentStatus.SINGULAR_HOMOGRAPHY,
            method_used="MOCK",
            ref_size=(640, 480),
            query_size=(640, 480),
            homography=singular_H,
            failure_reason="Homography is near-singular (det=0.00e+00).",
            inlier_ratio=0.10,
        )

        shifted_raw = BoundingBox(300.0, 300.0, 400.0, 400.0)
        # Without reliable registration, canonical_bbox cannot be projected (is None)
        det2 = Detection("multimeter", 0.90, shifted_raw, canonical_bbox=None, signature=sig)
        rep2 = self.engine.process_visit(VisitInput(
            "v2", "Visit 2", "2026-03-02T10:00:00Z",
            np.zeros((480, 640, 3), dtype=np.uint8),
            reg_singular,
            [det2]
        ))

        # Must NOT transition to CONFIRMED_MOVED
        ent_v2 = self.store.get_entity(ent.entity_id)
        self.assertEqual(ent_v2.current_state, EpistemicState.CONFIRMED_PRESENT)
        self.assertFalse(rep2.entity_updates[0].transition_occurred)

        # Visit 3: Singular homography with zero detections (unobserved)
        rep3 = self.engine.process_visit(VisitInput(
            "v3", "Visit 3", "2026-03-03T10:00:00Z",
            np.zeros((480, 640, 3), dtype=np.uint8),
            reg_singular,
            []
        ))

        # Must NOT transition to PENDING_REMOVAL or CONFIRMED_REMOVED
        ent_v3 = self.store.get_entity(ent.entity_id)
        self.assertEqual(ent_v3.current_state, EpistemicState.CONFIRMED_PRESENT)
        self.assertFalse(rep3.entity_updates[0].transition_occurred)
        self.assertEqual(rep3.entity_updates[0].visibility_status, VisibilityStatus.UNOBSERVED)
        self.assertIn("registration", rep3.entity_updates[0].explanation.lower())


if __name__ == "__main__":
    unittest.main()
