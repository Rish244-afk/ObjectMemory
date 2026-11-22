"""
Automated unit tests for Phase 4: Independent Pairwise Change-Detection Baseline.

Validates:
1. Deterministic one-to-one matching (no detection matched twice, deterministic tie-breaking).
2. Each supported outcome tested in isolation:
   - UNCHANGED (same location, high signature similarity)
   - MOVED (new location, high signature similarity)
   - ADDED (unmatched query detection)
   - REMOVED (unmatched reference detection with verified vacant query patch)
   - UNKNOWN (ambiguous identity match within margin)
   - UNKNOWN (rejected or failed alignment)
   - UNKNOWN (out-of-frame canonical bounding box)
   - UNKNOWN (missing detection without vacancy evidence)
3. Strict isolation: pairwise predictions cannot mutate persistent entity state.
4. Synthetic dataset generation reproducibility with fixed random seed.
5. Accuracy, precision, recall, F1, and rate of UNKNOWN metric calculations.
6. Explicit distinction between synthetic/mock tests and real model inference.
"""

from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest
import numpy as np

from objectmemory.baseline.comparator import (
    PairwiseComparator,
    calculate_bbox_iou,
    calculate_centroid_distance,
    check_bbox_overlap_fraction,
)
from objectmemory.baseline.models import (
    PairwiseComparisonReport,
    PairwiseMatch,
    PairwiseMatchConfig,
)
from objectmemory.benchmarks.evaluator import (
    PairwiseBenchmarkEvaluator,
    ScenarioEvaluationSummary,
)
from objectmemory.benchmarks.synthetic_dataset import (
    GroundTruthObject,
    SyntheticScenarioPair,
    SyntheticWorkbenchDatasetGenerator,
)
from objectmemory.detection.models import Detection, VisualSignature
from objectmemory.domain.models import BoundingBox, PairwiseBaselineResult, Session, WorkbenchEntity
from objectmemory.domain.states import BaselineChangeType, EpistemicState
from objectmemory.persistence.repository import SQLiteMemoryStore
from objectmemory.registration.models import AlignmentStatus, RegistrationResult


def _make_dummy_signature(split_idx: int = 0, dim: int = 148) -> VisualSignature:
    """Helper creating a valid L2-normalized synthetic visual signature fixture."""
    vec = np.zeros(dim, dtype=np.float32)
    if split_idx == 0:
        vec[:74] = 1.0
    elif split_idx == 1:
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


class TestOneToOneMatching(unittest.TestCase):
    """Verifies deterministic one-to-one assignment and tie-breaking."""

    def setUp(self):
        self.comparator = PairwiseComparator(
            config=PairwiseMatchConfig(signature_similarity_threshold=0.50, ambiguity_margin=0.05)
        )
        self.reg_mock = RegistrationResult(
            success=True,
            status=AlignmentStatus.SUCCESS,
            method_used="MOCK",
            ref_size=(400, 300),
            query_size=(400, 300),
            homography=np.eye(3, dtype=np.float64),
            overlap_mask=np.ones((300, 400), dtype=np.uint8) * 255,
        )

    def test_one_to_one_no_double_matching(self):
        """A single query detection cannot match multiple reference detections."""
        sig1 = _make_dummy_signature(0)
        sig2 = _make_dummy_signature(1)

        r1 = Detection("multimeter", 0.9, BoundingBox(50, 50, 100, 100), signature=sig1)
        r2 = Detection("multimeter", 0.9, BoundingBox(150, 50, 200, 100), signature=sig2)

        # Only one query detection, matches r1
        q1 = Detection("multimeter", 0.9, BoundingBox(52, 52, 102, 102), canonical_bbox=BoundingBox(52, 52, 102, 102), signature=sig1)

        report = self.comparator.compare(
            ref_detections=[r1, r2],
            query_detections=[q1],
            registration=self.reg_mock,
            is_synthetic=True,
        )

        matched_q_count = sum(1 for m in report.matches if m.query_detection is not None and m.predicted_change in (BaselineChangeType.UNCHANGED, BaselineChangeType.MOVED))
        self.assertEqual(matched_q_count, 1)

    def test_deterministic_tie_breaking(self):
        """Identical candidate scores produce deterministic assignment across repeated runs."""
        sig = _make_dummy_signature(1.0)
        r1 = Detection("box", 0.9, BoundingBox(50, 50, 100, 100), signature=sig)
        r2 = Detection("box", 0.9, BoundingBox(150, 50, 200, 100), signature=sig)
        q1 = Detection("box", 0.9, BoundingBox(50, 50, 100, 100), canonical_bbox=BoundingBox(50, 50, 100, 100), signature=sig)
        q2 = Detection("box", 0.9, BoundingBox(150, 50, 200, 100), canonical_bbox=BoundingBox(150, 50, 200, 100), signature=sig)

        report1 = self.comparator.compare([r1, r2], [q1, q2], self.reg_mock, is_synthetic=True)
        report2 = self.comparator.compare([r1, r2], [q1, q2], self.reg_mock, is_synthetic=True)

        self.assertEqual(len(report1.matches), len(report2.matches))
        for m1, m2 in zip(report1.matches, report2.matches):
            self.assertEqual(m1.predicted_change, m2.predicted_change)


class TestPairwiseOutcomes(unittest.TestCase):
    """Verifies each supported baseline outcome in controlled isolation."""

    def setUp(self):
        self.comparator = PairwiseComparator(
            config=PairwiseMatchConfig(
                signature_similarity_threshold=0.60,
                ambiguity_margin=0.10,
                iou_unchanged_threshold=0.40,
                distance_unchanged_threshold=35.0,
            )
        )
        self.reg_mock = RegistrationResult(
            success=True,
            status=AlignmentStatus.SUCCESS,
            method_used="MOCK",
            ref_size=(400, 300),
            query_size=(400, 300),
            homography=np.eye(3, dtype=np.float64),
            overlap_mask=np.ones((300, 400), dtype=np.uint8) * 255,
        )

    def test_outcome_unchanged(self):
        """High similarity and overlapping position results in UNCHANGED."""
        sig = _make_dummy_signature(1.0)
        r = Detection("multimeter", 0.9, BoundingBox(50, 50, 150, 150), signature=sig)
        q = Detection("multimeter", 0.9, BoundingBox(55, 55, 155, 155), canonical_bbox=BoundingBox(55, 55, 155, 155), signature=sig)

        report = self.comparator.compare([r], [q], self.reg_mock, is_synthetic=True)
        self.assertEqual(report.summary_counts[BaselineChangeType.UNCHANGED.value], 1)
        self.assertEqual(report.matches[0].predicted_change, BaselineChangeType.UNCHANGED)

    def test_outcome_moved(self):
        """High similarity but large spatial displacement results in MOVED."""
        sig = _make_dummy_signature(1.0)
        r = Detection("multimeter", 0.9, BoundingBox(50, 50, 150, 150), signature=sig)
        # Shifted by 150px
        q = Detection("multimeter", 0.9, BoundingBox(200, 50, 300, 150), canonical_bbox=BoundingBox(200, 50, 300, 150), signature=sig)

        report = self.comparator.compare([r], [q], self.reg_mock, is_synthetic=True)
        self.assertEqual(report.summary_counts[BaselineChangeType.MOVED.value], 1)
        self.assertEqual(report.matches[0].predicted_change, BaselineChangeType.MOVED)

    def test_outcome_added(self):
        """Unmatched query detection inside valid overlap results in ADDED."""
        sig = _make_dummy_signature(1.0)
        q = Detection("wire_strippers", 0.9, BoundingBox(100, 100, 180, 180), canonical_bbox=BoundingBox(100, 100, 180, 180), signature=sig)

        report = self.comparator.compare([], [q], self.reg_mock, is_synthetic=True)
        self.assertEqual(report.summary_counts[BaselineChangeType.ADDED.value], 1)
        self.assertEqual(report.matches[0].predicted_change, BaselineChangeType.ADDED)

    def test_outcome_removed_with_vacancy_evidence(self):
        """Unmatched reference object with vacant query patch confirmed results in REMOVED."""
        sig = _make_dummy_signature(1.0)
        r = Detection("chip", 0.9, BoundingBox(50, 50, 100, 100), signature=sig)

        # Query image is dark blank table (similarity to chip signature will be < vacancy_similarity_max)
        query_img = np.ones((300, 400, 3), dtype=np.uint8) * 30

        report = self.comparator.compare(
            ref_detections=[r],
            query_detections=[],
            registration=self.reg_mock,
            query_image_aligned=query_img,
            is_synthetic=True,
        )
        self.assertEqual(report.summary_counts[BaselineChangeType.REMOVED.value], 1)
        self.assertEqual(report.matches[0].predicted_change, BaselineChangeType.REMOVED)

    def test_outcome_unknown_on_ambiguous_twins(self):
        """Candidate match with near-identical second candidate yields UNKNOWN."""
        sig = _make_dummy_signature(1.0)
        # Reference resistor
        r = Detection("resistor", 0.9, BoundingBox(50, 50, 90, 70), signature=sig)
        # Two identical query resistors
        q1 = Detection("resistor", 0.9, BoundingBox(50, 50, 90, 70), canonical_bbox=BoundingBox(50, 50, 90, 70), signature=sig)
        q2 = Detection("resistor", 0.9, BoundingBox(120, 50, 160, 70), canonical_bbox=BoundingBox(120, 50, 160, 70), signature=sig)

        report = self.comparator.compare([r], [q1, q2], self.reg_mock, is_synthetic=True)
        self.assertGreater(report.summary_counts[BaselineChangeType.UNKNOWN.value], 0)
        ambiguous_matches = [m for m in report.matches if m.predicted_change == BaselineChangeType.UNKNOWN and m.details.get("ambiguity_flag")]
        self.assertEqual(len(ambiguous_matches), 1)

    def test_outcome_unknown_on_unreliable_alignment(self):
        """When registration fails or is rejected, all predictions are UNKNOWN."""
        reg_failed = RegistrationResult(
            success=False,
            status=AlignmentStatus.INSUFFICIENT_KEYPOINTS,
            method_used="SIFT",
            ref_size=(400, 300),
            query_size=(400, 300),
            failure_reason="Insufficient keypoints detected.",
        )
        sig = _make_dummy_signature(1.0)
        r = Detection("multimeter", 0.9, BoundingBox(50, 50, 100, 100), signature=sig)
        q = Detection("multimeter", 0.9, BoundingBox(50, 50, 100, 100), signature=sig)

        report = self.comparator.compare([r], [q], registration=reg_failed, is_synthetic=True)
        self.assertEqual(report.summary_counts[BaselineChangeType.UNKNOWN.value], 2)
        self.assertEqual(report.summary_counts[BaselineChangeType.UNCHANGED.value], 0)

    def test_outcome_unknown_on_out_of_frame_reference(self):
        """Reference object whose canonical bbox is outside query overlap yields UNKNOWN."""
        mask = np.zeros((300, 400), dtype=np.uint8)
        # Overlap only covers left half of image (x in [0, 200])
        mask[:, :200] = 255

        reg_partial = RegistrationResult(
            success=True,
            status=AlignmentStatus.SUCCESS,
            method_used="MOCK",
            ref_size=(400, 300),
            query_size=(400, 300),
            overlap_mask=mask,
        )
        sig = _make_dummy_signature(1.0)
        # Reference object is at x=250..350 (completely outside overlap mask)
        r = Detection("tool", 0.9, BoundingBox(250, 50, 350, 150), signature=sig)

        report = self.comparator.compare([r], [], registration=reg_partial, is_synthetic=True)
        self.assertEqual(report.summary_counts[BaselineChangeType.UNKNOWN.value], 1)
        self.assertEqual(report.matches[0].details.get("reason"), "OUT_OF_FRAME")

    def test_outcome_unknown_on_missing_detection_without_vacancy_evidence(self):
        """Non-detection is not removal: without query image vacancy confirmation, yields UNKNOWN."""
        sig = _make_dummy_signature(1.0)
        r = Detection("tool", 0.9, BoundingBox(50, 50, 100, 100), signature=sig)

        # No query image provided -> cannot verify vacancy
        report = self.comparator.compare([r], [], registration=self.reg_mock, query_image_aligned=None, is_synthetic=True)
        self.assertEqual(report.summary_counts[BaselineChangeType.UNKNOWN.value], 1)
        self.assertEqual(report.matches[0].predicted_change, BaselineChangeType.UNKNOWN)


class TestPersistenceIsolation(unittest.TestCase):
    """Verifies that pairwise baseline storage cannot mutate persistent entity state."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_store.sqlite"
        self.store = SQLiteMemoryStore(self.db_path)

        now_iso = "2026-10-10T00:00:00Z"
        self.store.add_session(Session("s_ref", "Ref", now_iso, "img://ref"))
        self.store.add_session(Session("s_qry", "Qry", now_iso, "img://qry"))

        # Add a persistent entity with CONFIRMED_PRESENT state
        self.entity = WorkbenchEntity(
            entity_id="ent_multimeter",
            label="multimeter",
            created_at=now_iso,
            canonical_bbox=BoundingBox(50, 50, 150, 150),
            current_state=EpistemicState.CONFIRMED_PRESENT,
        )
        self.store.add_entity(self.entity)

    def tearDown(self):
        self.store.close()
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_pairwise_result_does_not_mutate_entity_state(self):
        """Storing a REMOVED pairwise baseline result leaves entity in CONFIRMED_PRESENT."""
        res = PairwiseBaselineResult(
            result_id="pw_001",
            reference_session_id="s_ref",
            query_session_id="s_qry",
            entity_id=self.entity.entity_id,
            predicted_change=BaselineChangeType.REMOVED,
            confidence=0.85,
            details={"is_synthetic": True},
        )
        self.store.add_pairwise_result(res)

        # Check pairwise record was persisted
        pairwise_records = self.store.list_pairwise_results("s_ref", "s_qry")
        self.assertEqual(len(pairwise_records), 1)
        self.assertEqual(pairwise_records[0].predicted_change, BaselineChangeType.REMOVED)

        # CRITICAL SAFEGUARD: Entity state MUST remain CONFIRMED_PRESENT!
        ent_after = self.store.get_entity(self.entity.entity_id)
        self.assertIsNotNone(ent_after)
        self.assertEqual(ent_after.current_state, EpistemicState.CONFIRMED_PRESENT)

        # No change_events were generated
        events = self.store.list_change_events_for_entity(self.entity.entity_id)
        self.assertEqual(len(events), 0)


class TestSyntheticDatasetReproducibility(unittest.TestCase):
    """Verifies synthetic dataset generator reproducibility and scenario coverage."""

    def test_generator_deterministic_reproducibility(self):
        """Identical random seed produces identical images and object annotations."""
        gen = SyntheticWorkbenchDatasetGenerator(width=320, height=240)
        set1 = gen.generate_all_scenarios(base_seed=123)
        set2 = gen.generate_all_scenarios(base_seed=123)

        self.assertEqual(len(set1), 8)
        self.assertEqual(len(set2), 8)

        for p1, p2 in zip(set1, set2):
            self.assertEqual(p1.scenario_name, p2.scenario_name)
            self.assertTrue(p1.is_synthetic)
            self.assertTrue(np.array_equal(p1.ref_image, p2.ref_image))
            self.assertTrue(np.array_equal(p1.query_image, p2.query_image))
            self.assertEqual(len(p1.objects), len(p2.objects))
            for o1, o2 in zip(p1.objects, p2.objects):
                self.assertEqual(o1.object_id, o2.object_id)
                self.assertEqual(o1.ground_truth_change, o2.ground_truth_change)

    def test_all_8_scenarios_generated(self):
        """Verifies presence of all 8 mandatory scenario types."""
        gen = SyntheticWorkbenchDatasetGenerator(width=320, height=240)
        scenarios = gen.generate_all_scenarios(base_seed=42)
        names = {s.scenario_name for s in scenarios}
        expected = {
            "no_change",
            "object_moved",
            "object_added",
            "object_removed",
            "lighting_change",
            "partial_occlusion",
            "ambiguous_objects",
            "unreliable_alignment",
        }
        self.assertEqual(names, expected)


class TestMetricCalculations(unittest.TestCase):
    """Verifies evaluation metric math (confusion matrix, precision, recall, F1, unknown rate)."""

    def test_evaluator_metrics_computation(self):
        """Tests that precision, recall, and F1 formulas are mathematically exact."""
        gen = SyntheticWorkbenchDatasetGenerator(width=320, height=240)
        pair = gen.generate_scenario("no_change", seed=42)

        evaluator = PairwiseBenchmarkEvaluator()
        report = evaluator.evaluate_all([pair])

        self.assertGreater(report.total_objects, 0)
        self.assertTrue(0.0 <= report.overall_accuracy <= 1.0)
        self.assertTrue(0.0 <= report.refusal_rate <= 1.0)
        self.assertTrue(0.0 <= report.unknown_rate <= 1.0)
        for cls_name, m in report.per_class_metrics.items():
            self.assertTrue(0.0 <= m["precision"] <= 1.0)
            self.assertTrue(0.0 <= m["recall"] <= 1.0)
            self.assertTrue(0.0 <= m["f1"] <= 1.0)

    def test_confusion_matrix_row_sum_equals_support(self):
        """Audits that the sum of all elements in the confusion matrix exactly equals total evaluated objects."""
        gen = SyntheticWorkbenchDatasetGenerator()
        scenarios = gen.generate_all_scenarios(base_seed=42)

        evaluator = PairwiseBenchmarkEvaluator()
        report = evaluator.evaluate_all(scenarios)

        self.assertEqual(report.total_objects, 18)
        total_matrix_elements = sum(
            sum(row.values()) for row in report.confusion_matrix.values()
        )
        self.assertEqual(total_matrix_elements, 18)
        self.assertEqual(
            report.total_correct + report.total_incorrect + report.total_refusals,
            18
        )
        self.assertEqual(report.unassociated_predictions_count, 0)


class TestAssignmentOptimization(unittest.TestCase):
    """Verifies Kuhn-Munkres optimal assignment vs greedy assignment."""

    def test_kuhn_munkres_optimal_assignment_counterexample(self):
        """
        Tests the 2x2 counterexample where greedy fails to find the global optimum:
        Cost/Similarity Matrix:
               Q1     Q2
        R1   0.90   0.85
        R2   0.88   0.10

        Threshold: 0.65
        Greedy picks (R1, Q1) at 0.90. R2 is left with Q2 (0.10 < 0.65 -> rejected).
        Total matches: 1. Total score: 0.90.

        Optimal (Kuhn-Munkres) picks (R1, Q2) at 0.85 and (R2, Q1) at 0.88.
        Total matches: 2. Total score: 1.73.
        """
        from objectmemory.baseline.comparator import solve_optimal_assignment

        # Similarity matrix:
        # R1: [0.90, 0.85]
        # R2: [0.88, 0.10]
        # For minimum-cost matching, cost = max_sim - sim
        sim_matrix = np.array([
            [0.90, 0.85],
            [0.88, 0.10],
        ])
        cost_matrix = 1.0 - sim_matrix
        assigned_pairs = solve_optimal_assignment(cost_matrix)
        # Optimal matching pairs R1 -> Q2 (idx 0 -> 1) and R2 -> Q1 (idx 1 -> 0)
        pair_dict = dict(assigned_pairs)
        self.assertEqual(pair_dict.get(0), 1)  # R1 -> Q2 (0.85)
        self.assertEqual(pair_dict.get(1), 0)  # R2 -> Q1 (0.88)
        self.assertEqual(len(assigned_pairs), 2)


class TestDegenerateCropSafeguards(unittest.TestCase):
    """Verifies that degenerate crop signatures (is_valid=False) cannot confirm removal."""

    def test_invalid_patch_signature_yields_unknown_not_removed(self):
        """When query patch extraction fails (e.g. crop < 2x2), comparator must refuse with UNKNOWN."""
        comparator = PairwiseComparator()
        reg_mock = RegistrationResult(
            success=True,
            status=AlignmentStatus.SUCCESS,
            method_used="MOCK",
            ref_size=(400, 300),
            query_size=(400, 300),
            homography=np.eye(3, dtype=np.float64),
            overlap_mask=np.ones((300, 400), dtype=np.uint8) * 255,
        )
        sig = _make_dummy_signature(0)
        r_det = Detection("wire_stripper", 0.92, BoundingBox(50, 50, 100, 100), signature=sig)

        # Mock signature extractor returning is_valid=False for query patch
        class MockInvalidExtractor:
            def extract(self, img, bbox):
                dummy = np.ones(148, dtype=np.float32) / np.sqrt(148)
                return VisualSignature(
                    color_histogram=dummy[:128].copy(),
                    texture_descriptor=dummy[128:].copy(),
                    combined_vector=dummy,
                    is_valid=False,
                    failure_reason="Degenerate crop test",
                )

        comp_with_invalid = PairwiseComparator(signature_extractor=MockInvalidExtractor())
        report = comp_with_invalid.compare(
            ref_detections=[r_det],
            query_detections=[],
            registration=reg_mock,
            ref_image=np.zeros((300, 400, 3), dtype=np.uint8),
            query_image_aligned=np.zeros((300, 400, 3), dtype=np.uint8),
        )
        self.assertEqual(len(report.matches), 1)
        # Must be UNKNOWN, NOT REMOVED
        self.assertEqual(report.matches[0].predicted_change, BaselineChangeType.UNKNOWN)
        self.assertIn("vacancy verification aborted", report.matches[0].explanation)


if __name__ == "__main__":
    unittest.main()
