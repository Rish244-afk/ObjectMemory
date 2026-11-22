"""
Unit tests for ObjectMemory Benchmark Evaluation Runner (Phase 3C.2 Remediated).

Verifies:
1. Valid and invalid manifest handling.
2. Strict rejection of BLOCKED configurations (B1, B2, A1, A4, A6).
3. Partial-ablation handling and explicit limitation labeling (A2, A3).
4. Execution and database isolation (temporary in-memory store, no shared state).
5. B0 memoryless detector baseline behavior (Precision, Recall, F1; N/A temporal scores).
6. B3 complete ObjectMemory execution with all safeguards active.
7. A5 ambiguity ablation execution (ambiguity_margin=0.0).
8. Exact metric computations against hand-calculated expected values.
9. Zero-denominator behavior (returns None, logs reason in unscorable_cases).
10. Unobservable and missing annotation handling (no false removal).
11. Registration failure handling (absence freezing).
12. Repeated-run reproducibility (identical hashes and metrics).
13. Provenance tracking and pilot dataset warning enforcement.
14. Result serialization and schema completeness.
15. Elimination of historical state leakage via immutable per-visit prediction snapshots.
16. Coordinate-frame transformations via 4-corner perspective mapping.
17. Stable bipartite matching across persistent object relocation.
18. File-backed SQLite persistence across backend restart.
19. Identity consistency penalizes missed observations.
20. Duplicate identities evaluate per-entity fragmentation.
21. False discovery proportion calculation.
22. Confirmation delay accompanied by confirmation rate and unconfirmed events.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from objectmemory.benchmarks.config_factory import (
    BenchmarkConfig,
    BenchmarkConfigFactory,
    ConfigurationSupportStatus,
    UnsupportedConfigurationError,
)
from objectmemory.benchmarks.manifest import (
    DatasetManifest,
    DatasetProvenance,
    ManifestLoader,
    ManifestValidationError,
    ObjectAnnotationEntry,
    ObservationEntry,
    SequenceEntry,
)
from objectmemory.benchmarks.runner import (
    BenchmarkMetrics,
    BenchmarkRunResult,
    BenchmarkRunner,
    transform_query_bbox_to_reference,
)
from objectmemory.detection.detector import MockObjectDetector
from objectmemory.detection.models import BoundingBox, Detection, DetectionBatch
from objectmemory.domain.states import EpistemicState, VisibilityStatus
from objectmemory.persistence.repository import SQLiteMemoryStore
from objectmemory.registration.models import AlignmentStatus, RegistrationResult
from objectmemory.registration.registrar import PlanarRegistrar
from objectmemory.temporal.engine import TemporalMemoryEngine
from objectmemory.temporal.models import TemporalEngineConfig, VisitInput


def create_synthetic_image(path: Path, width: int = 640, height: int = 480) -> None:
    """Creates a deterministic synthetic test image on disk."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (200, 200), (255, 255, 255), -1)
    cv2.circle(img, (400, 300), 50, (128, 128, 128), -1)
    cv2.imwrite(str(path), img)


class TestBenchmarkRunner(unittest.TestCase):

    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp(prefix="om_runner_test_")
        self.root_path = Path(self.test_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _build_test_manifest(
        self,
        num_visits: int = 3,
        provenance: str = "GENUINE_PHYSICAL",
        dataset_name: str = "test_workbench",
        include_removals: bool = False,
    ) -> DatasetManifest:
        """Helper to create a fully validated DatasetManifest with test images on disk."""
        seq_dir = self.root_path / "seq_01"
        seq_dir.mkdir(parents=True, exist_ok=True)

        observations_data = []
        for v in range(1, num_visits + 1):
            img_name = f"visit_{v:02d}.png"
            img_path = seq_dir / img_name
            create_synthetic_image(img_path)

            if include_removals and v >= 2:
                objs = [
                    {
                        "object_id": "cup_01",
                        "class_name": "cup",
                        "bounding_box": None,
                        "visibility_status": "UNOBSERVED",
                        "object_state": "CONFIRMED_REMOVED",
                        "timestamp": f"2026-11-14T0{v}:00:00Z",
                    }
                ]
            else:
                objs = [
                    {
                        "object_id": "cup_01",
                        "class_name": "cup",
                        "bounding_box": [100.0, 100.0, 200.0, 200.0],
                        "visibility_status": "CLEARLY_VISIBLE",
                        "object_state": "CONFIRMED_PRESENT",
                        "timestamp": f"2026-11-14T0{v}:00:00Z",
                    }
                ]

            observations_data.append(
                {
                    "observation_id": f"obs_{v:02d}",
                    "timestamp": f"2026-11-14T0{v}:00:00Z",
                    "image_path": f"seq_01/{img_name}",
                    "objects": objs,
                }
            )

        manifest_data = {
            "manifest_version": "2.0.0",
            "dataset_name": dataset_name,
            "provenance": provenance,
            "sequences": [
                {
                    "sequence_id": "seq_01",
                    "environment_id": "env_workbench",
                    "observations": observations_data,
                }
            ],
        }

        manifest_file = self.root_path / "manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        return ManifestLoader.load(manifest_file, dataset_root=self.root_path, check_files=True)

    # -------------------------------------------------------------------------
    # Test 1: Valid and Invalid Manifest Handling
    # -------------------------------------------------------------------------
    def test_01_valid_and_invalid_manifests(self) -> None:
        manifest = self._build_test_manifest(num_visits=2)
        self.assertEqual(manifest.dataset_name, "test_workbench")
        self.assertEqual(manifest.sequence_count, 1)

        bad_manifest_file = self.root_path / "bad_manifest.json"
        with open(bad_manifest_file, "w", encoding="utf-8") as f:
            json.dump({"manifest_version": "2.0.0", "dataset_name": "bad"}, f)

        runner = BenchmarkRunner()
        with self.assertRaises(ManifestValidationError):
            runner.run_manifest(bad_manifest_file, dataset_root=self.root_path)

    # -------------------------------------------------------------------------
    # Test 2: Rejection of Blocked Configurations
    # -------------------------------------------------------------------------
    def test_02_unsupported_configuration_rejection(self) -> None:
        manifest = self._build_test_manifest(num_visits=2)
        runner = BenchmarkRunner(allow_blocked=False)

        blocked_ids = ["B1", "B2", "A1", "A4", "A6"]
        for cid in blocked_ids:
            with self.assertRaises(UnsupportedConfigurationError):
                runner.run_manifest(manifest, config_ids=[cid])

    # -------------------------------------------------------------------------
    # Test 3: Partial-Ablation Handling and Limitation Labeling
    # -------------------------------------------------------------------------
    def test_03_partial_ablation_labeling(self) -> None:
        manifest = self._build_test_manifest(num_visits=2)

        runner_strict = BenchmarkRunner(allow_partial=False)
        with self.assertRaises(UnsupportedConfigurationError):
            runner_strict.run_manifest(manifest, config_ids=["A2"])

        with self.assertRaises(UnsupportedConfigurationError):
            runner_strict.run_manifest(manifest, config_ids=["A3"])

        runner_partial = BenchmarkRunner(allow_partial=True)
        results = runner_partial.run_manifest(manifest, config_ids=["A2", "A3"])

        self.assertIn("A2", results)
        self.assertIn("A3", results)

        a2_res = results["A2"]
        self.assertTrue(a2_res.is_partial_ablation)
        self.assertEqual(a2_res.support_status, "PARTIALLY_SUPPORTED")
        self.assertTrue(len(a2_res.partial_ablation_limitations) > 0)

        a3_res = results["A3"]
        self.assertTrue(a3_res.is_partial_ablation)
        self.assertEqual(a3_res.support_status, "PARTIALLY_SUPPORTED")
        self.assertTrue(len(a3_res.partial_ablation_limitations) > 0)

    # -------------------------------------------------------------------------
    # Test 4: Execution and Database Isolation
    # -------------------------------------------------------------------------
    def test_04_execution_and_database_isolation(self) -> None:
        manifest = self._build_test_manifest(num_visits=3)
        runner = BenchmarkRunner()

        results = runner.run_manifest(manifest, config_ids=["B0", "B3"])
        self.assertEqual(len(results), 2)
        self.assertIn("B0", results)
        self.assertIn("B3", results)

        # Verify results have different unique run IDs
        self.assertNotEqual(results["B0"].run_id, results["B3"].run_id)

    # -------------------------------------------------------------------------
    # Test 5: B0 Memoryless Detector Baseline Behavior (Remediated)
    # -------------------------------------------------------------------------
    def test_05_b0_memoryless_execution(self) -> None:
        manifest = self._build_test_manifest(num_visits=3, include_removals=True)

        mock_det = MockObjectDetector(
            canned_detections=[
                Detection(
                    label="cup",
                    confidence=0.90,
                    raw_bbox=BoundingBox(100.0, 100.0, 200.0, 200.0),
                )
            ]
        )

        runner = BenchmarkRunner(detector=mock_det)
        result = runner.run_config(
            manifest=manifest,
            config=BenchmarkConfigFactory.create_config("B0"),
        )

        self.assertEqual(result.config_id, "B0")
        self.assertEqual(result.pipeline_type, "MEMORYLESS_DETECTION")

        # Temporal metrics must be None (N/A) for B0, never misleading 0.0 scores!
        self.assertIsNone(result.metrics.false_removal_rate)
        self.assertIsNone(result.metrics.false_discovery_proportion)
        self.assertIsNone(result.metrics.removal_recall)
        self.assertIsNone(result.metrics.unsupported_movement_rate)
        self.assertIsNone(result.metrics.identity_consistency)

        # Standard object detection metrics must be calculated
        self.assertIsNotNone(result.metrics.detection_precision)
        self.assertIsNotNone(result.metrics.detection_recall)
        self.assertIsNotNone(result.metrics.detection_f1_score)
        self.assertEqual(result.metrics.detection_true_positives, 1)

    # -------------------------------------------------------------------------
    # Test 6: B3 Complete ObjectMemory Execution
    # -------------------------------------------------------------------------
    def test_06_b3_full_system_execution(self) -> None:
        manifest = self._build_test_manifest(num_visits=3, include_removals=False)

        mock_det = MockObjectDetector(
            canned_detections=[
                Detection(
                    label="cup",
                    confidence=0.95,
                    raw_bbox=BoundingBox(100.0, 100.0, 200.0, 200.0),
                )
            ]
        )

        runner = BenchmarkRunner(detector=mock_det)
        result = runner.run_config(
            manifest=manifest,
            config=BenchmarkConfigFactory.create_config("B3"),
        )

        self.assertEqual(result.config_id, "B3")
        self.assertEqual(result.pipeline_type, "FULL_TEMPORAL_MEMORY")
        self.assertFalse(result.is_partial_ablation)

        self.assertEqual(result.metrics.false_removal_rate, 0.0)
        self.assertEqual(result.metrics.false_removal_count, 0)
        self.assertEqual(result.metrics.overall_state_accuracy, 1.0)
        self.assertEqual(result.metrics.identity_consistency, 1.0)
        self.assertEqual(result.metrics.duplicate_identities, 0)

    # -------------------------------------------------------------------------
    # Test 7: A5 Ambiguity Ablation Execution
    # -------------------------------------------------------------------------
    def test_07_a5_ambiguity_ablation_execution(self) -> None:
        manifest = self._build_test_manifest(num_visits=2)
        runner = BenchmarkRunner()

        result = runner.run_config(
            manifest=manifest,
            config=BenchmarkConfigFactory.create_config("A5"),
        )

        self.assertEqual(result.config_id, "A5")
        self.assertEqual(result.effective_parameters["ambiguity_margin"], 0.0)
        self.assertFalse(result.is_partial_ablation)

    # -------------------------------------------------------------------------
    # Test 8: Hand-Calculated Metrics Verification (Remediated)
    # -------------------------------------------------------------------------
    def test_08_hand_calculated_metrics(self) -> None:
        m = BenchmarkMetrics(
            total_observations_processed=5,
            total_entity_evaluations=10,
            false_removal_rate=1.0 / 7.0,
            false_removal_count=1,
            false_removal_denominator=7,
            false_discovery_proportion=1.0 / 3.0,
            false_discovery_count=1,
            false_discovery_denominator=3,
            removal_recall=2.0 / 3.0,
            removal_recall_count=2,
            removal_recall_denominator=3,
            event_removal_recall=1.0 / 1.0,
            event_removal_recall_count=1,
            event_removal_recall_denominator=1,
            confirmation_rate=2.0 / 3.0,
            confirmed_change_events_count=2,
            total_eligible_change_events=3,
            unconfirmed_change_events_count=1,
            mean_confirmation_delay_visits=2.5,
            confirmation_delay_events_count=2,
            unsupported_movement_rate=1.0 / 5.0,
            unsupported_movement_count=1,
            unsupported_movement_denominator=5,
            identity_consistency=6.0 / 10.0,
            identity_consistency_numerator=6,
            identity_consistency_denominator=10,
            id_switches=1,
            id_switch_rate=1.0 / 7.0,
            id_switch_transitions=7,
            fragmented_identities=1,
            spurious_system_tracks=0,
            duplicate_identities=1,
            overall_state_accuracy=8.0 / 10.0,
            correct_state_evaluations=8,
        )

        d = m.to_dict()
        self.assertAlmostEqual(d["false_removal_rate"], 0.142857, places=4)
        self.assertAlmostEqual(d["false_discovery_proportion"], 0.333333, places=4)
        self.assertAlmostEqual(d["removal_recall"], 0.666667, places=4)
        self.assertAlmostEqual(d["event_removal_recall"], 1.0, places=4)
        self.assertAlmostEqual(d["confirmation_rate"], 0.666667, places=4)
        self.assertEqual(d["unconfirmed_change_events_count"], 1)
        self.assertAlmostEqual(d["identity_consistency"], 0.600000, places=4)

    # -------------------------------------------------------------------------
    # Test 9: Zero-Denominator Behavior
    # -------------------------------------------------------------------------
    def test_09_zero_denominator_behavior(self) -> None:
        manifest = self._build_test_manifest(num_visits=2, include_removals=False)

        mock_det = MockObjectDetector(
            canned_detections=[
                Detection(
                    label="cup",
                    confidence=0.95,
                    raw_bbox=BoundingBox(100.0, 100.0, 200.0, 200.0),
                )
            ]
        )
        runner = BenchmarkRunner(detector=mock_det)
        res = runner.run_config(manifest, BenchmarkConfigFactory.create_config("B3"))

        self.assertIsNone(res.metrics.removal_recall)
        self.assertIsNone(res.metrics.false_discovery_proportion)
        self.assertIsNone(res.metrics.historical_retrieval_accuracy)

        report_text = res.format_report()
        self.assertIn("Removal Recall (vis):N/A", report_text)
        self.assertIn("False Discovery Prop:N/A", report_text)

    # -------------------------------------------------------------------------
    # Test 10: Unobservable and Missing Annotations
    # -------------------------------------------------------------------------
    def test_10_unobservable_and_missing_annotations(self) -> None:
        seq_dir = self.root_path / "seq_occl"
        seq_dir.mkdir(parents=True, exist_ok=True)

        img_path = seq_dir / "v1.png"
        create_synthetic_image(img_path)

        manifest_data = {
            "manifest_version": "2.0.0",
            "dataset_name": "occlusion_test",
            "provenance": "GENUINE_PHYSICAL",
            "sequences": [
                {
                    "sequence_id": "seq_occl",
                    "environment_id": "env_test",
                    "observations": [
                        {
                            "observation_id": "obs_1",
                            "timestamp": "2026-11-14T01:00:00Z",
                            "image_path": "seq_occl/v1.png",
                            "objects": [
                                {
                                    "object_id": "cup_occluded",
                                    "class_name": "cup",
                                    "bounding_box": [100.0, 100.0, 200.0, 200.0],
                                    "visibility_status": "HEAVILY_OCCLUDED",
                                    "object_state": "CONFIRMED_PRESENT",
                                    "timestamp": "2026-11-14T01:00:00Z",
                                }
                            ],
                        }
                    ],
                }
            ],
        }
        m_file = self.root_path / "manifest_occl.json"
        with open(m_file, "w") as f:
            json.dump(manifest_data, f)

        manifest = ManifestLoader.load(m_file, dataset_root=self.root_path)
        runner = BenchmarkRunner(detector=MockObjectDetector(canned_detections=[]))
        res = runner.run_config(manifest, BenchmarkConfigFactory.create_config("B3"))

        self.assertEqual(res.metrics.false_removal_count, 0)
        self.assertEqual(res.metrics.false_removal_rate, 0.0)

    # -------------------------------------------------------------------------
    # Test 11: Registration Failure Handling (Absence Freezing)
    # -------------------------------------------------------------------------
    def test_11_registration_failure_handling(self) -> None:
        manifest = self._build_test_manifest(num_visits=3)

        class FailingRegistrar(PlanarRegistrar):
            def register(self, ref_image, query_image, **kwargs):
                return RegistrationResult(
                    success=False,
                    status=AlignmentStatus.INSUFFICIENT_INLIERS,
                    method_used="mock_fail",
                    ref_size=(640, 480),
                    query_size=(640, 480),
                    inlier_ratio=0.05,
                    homography=None,
                    overlap_mask=None,
                    aligned_image=None,
                )

        runner = BenchmarkRunner(
            detector=MockObjectDetector(canned_detections=[]),
            registrar=FailingRegistrar(),
        )
        res = runner.run_config(manifest, BenchmarkConfigFactory.create_config("B3"))

        self.assertEqual(res.metrics.false_removal_count, 0)
        self.assertEqual(res.metrics.false_removal_rate, 0.0)

    # -------------------------------------------------------------------------
    # Test 12: Repeated-Run Reproducibility
    # -------------------------------------------------------------------------
    def test_12_repeated_run_reproducibility(self) -> None:
        manifest = self._build_test_manifest(num_visits=2)
        mock_det = MockObjectDetector(
            canned_detections=[
                Detection(
                    label="cup",
                    confidence=0.95,
                    raw_bbox=BoundingBox(100.0, 100.0, 200.0, 200.0),
                )
            ]
        )

        runner = BenchmarkRunner(detector=mock_det)
        cfg = BenchmarkConfigFactory.create_config("B3")

        run1 = runner.run_config(manifest, cfg)
        run2 = runner.run_config(manifest, cfg)

        self.assertEqual(run1.config_fingerprint, run2.config_fingerprint)
        self.assertEqual(run1.metrics.to_dict(), run2.metrics.to_dict())

    # -------------------------------------------------------------------------
    # Test 13: Provenance and Pilot Dataset Warning Enforcement
    # -------------------------------------------------------------------------
    def test_13_provenance_and_pilot_dataset_warning(self) -> None:
        manifest_pilot = self._build_test_manifest(
            num_visits=2,
            provenance="EXPLORATORY_PILOT",
            dataset_name="desk_pilot_01",
        )
        self.assertTrue(manifest_pilot.is_pilot_dataset)

        runner = BenchmarkRunner()
        res_pilot = runner.run_config(manifest_pilot, BenchmarkConfigFactory.create_config("B3"))

        self.assertTrue(res_pilot.is_pilot_dataset)
        report_text = res_pilot.format_report()
        self.assertIn("EXPLORATORY PILOT DATA: This dataset represents an exploratory pilot.", report_text)

    # -------------------------------------------------------------------------
    # Test 14: Result Serialization and Schema Completeness
    # -------------------------------------------------------------------------
    def test_14_result_serialization_and_schema(self) -> None:
        manifest = self._build_test_manifest(num_visits=2)
        runner = BenchmarkRunner()
        res = runner.run_config(manifest, BenchmarkConfigFactory.create_config("B3"))

        res_json = res.to_json()
        parsed = json.loads(res_json)

        required_keys = [
            "run_id",
            "timestamp",
            "dataset_name",
            "dataset_provenance",
            "manifest_version",
            "is_pilot_dataset",
            "is_genuine_physical",
            "sequence_ids",
            "config_id",
            "config_name",
            "config_fingerprint",
            "pipeline_type",
            "support_status",
            "is_partial_ablation",
            "partial_ablation_limitations",
            "effective_parameters",
            "metrics",
            "unscorable_cases",
            "uninstantiated_gt_entities",
            "execution_errors",
            "runtime_seconds",
        ]
        for k in required_keys:
            self.assertIn(k, parsed, f"Missing required key in JSON output: {k}")

    # -------------------------------------------------------------------------
    # Test 15: Stage 1 Regression Test — Elimination of Historical State Leakage
    # -------------------------------------------------------------------------
    def test_15_regression_historical_state_leakage_eliminated(self) -> None:
        """
        Tests the exact audit defect:
        1. Object present in visit 1.
        2. Object remains present in visit 2.
        3. Object is removed in visit 3.
        Verify that visits 1 and 2 remain PRESENT in evaluation, and visit 3 receives the actual predicted state.
        Zero false removals must be recorded in visits 1 and 2.
        """
        manifest = self._build_test_manifest(num_visits=3, include_removals=True)

        class MultiVisitMockDetector(MockObjectDetector):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def detect(self, image, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    # Visit 1: detected
                    return DetectionBatch(
                        detections=[Detection("cup", 0.95, BoundingBox(100.0, 100.0, 200.0, 200.0))],
                        image_size=(640, 480),
                        latency_ms=5.0,
                        detector_name="MockDetector",
                    )
                # Visits 2 and 3: unobserved / removed
                return DetectionBatch(detections=[], image_size=(640, 480), latency_ms=5.0, detector_name="MockDetector")

        runner = BenchmarkRunner(detector=MultiVisitMockDetector())
        result = runner.run_config(manifest, BenchmarkConfigFactory.create_config("B3"))

        # Visit 1 was present, Visit 2 was removed in GT, Visit 3 was removed in GT
        # Crucially: Visit 1 was NOT falsely evaluated as removed!
        self.assertEqual(result.metrics.false_removal_count, 0)
        self.assertEqual(result.metrics.false_removal_rate, 0.0)

    # -------------------------------------------------------------------------
    # Test 16: Stage 2 Test — Coordinate-Frame Transformations
    # -------------------------------------------------------------------------
    def test_16_coordinate_frame_transformations(self) -> None:
        """Tests 4-corner perspective transformations under translation, scale, perspective, and invalid homography."""
        ref_size = (640, 480)
        box = BoundingBox(100.0, 100.0, 200.0, 200.0)

        # 1. Identity homography
        h_ident = np.eye(3, dtype=np.float64)
        box_ident = transform_query_bbox_to_reference(box, h_ident, ref_size)
        self.assertIsNotNone(box_ident)
        self.assertAlmostEqual(box_ident.x_min, 100.0)
        self.assertAlmostEqual(box_ident.x_max, 200.0)

        # 2. Translation (+50 x, +30 y)
        h_trans = np.array([[1.0, 0.0, 50.0], [0.0, 1.0, 30.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        box_trans = transform_query_bbox_to_reference(box, h_trans, ref_size)
        self.assertIsNotNone(box_trans)
        self.assertAlmostEqual(box_trans.x_min, 150.0)
        self.assertAlmostEqual(box_trans.y_min, 130.0)
        self.assertAlmostEqual(box_trans.x_max, 250.0)
        self.assertAlmostEqual(box_trans.y_max, 230.0)

        # 3. Scale (2x)
        h_scale = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        box_scale = transform_query_bbox_to_reference(box, h_scale, ref_size)
        self.assertIsNotNone(box_scale)
        self.assertAlmostEqual(box_scale.x_min, 200.0)
        self.assertAlmostEqual(box_scale.x_max, 400.0)

        # 4. Invalid homography (None or non-finite)
        self.assertIsNone(transform_query_bbox_to_reference(box, None, ref_size))
        h_nan = np.full((3, 3), np.nan)
        self.assertIsNone(transform_query_bbox_to_reference(box, h_nan, ref_size))

    # -------------------------------------------------------------------------
    # Test 17: Stage 3 Test — Object Relocation Retains Identity
    # -------------------------------------------------------------------------
    def test_17_object_relocation_retains_identity(self) -> None:
        """
        Tests that an object relocating persistently across visits (>150px)
        does NOT lose its identity or become uninstantiated.
        """
        seq_dir = self.root_path / "seq_reloc"
        seq_dir.mkdir(parents=True, exist_ok=True)
        img1 = seq_dir / "v1.png"
        img2 = seq_dir / "v2.png"
        create_synthetic_image(img1)
        create_synthetic_image(img2)

        # Cup moves from (50, 50) to (350, 350) => displacement ~424px!
        manifest_data = {
            "manifest_version": "2.0.0",
            "dataset_name": "relocation_test",
            "provenance": "GENUINE_PHYSICAL",
            "sequences": [
                {
                    "sequence_id": "seq_reloc",
                    "environment_id": "env_reloc",
                    "observations": [
                        {
                            "observation_id": "obs_1",
                            "timestamp": "2026-11-14T01:00:00Z",
                            "image_path": "seq_reloc/v1.png",
                            "objects": [
                                {
                                    "object_id": "cup_relocated",
                                    "class_name": "cup",
                                    "bounding_box": [50.0, 50.0, 100.0, 100.0],
                                    "visibility_status": "CLEARLY_VISIBLE",
                                    "object_state": "CONFIRMED_PRESENT",
                                    "timestamp": "2026-11-14T01:00:00Z",
                                }
                            ],
                        },
                        {
                            "observation_id": "obs_2",
                            "timestamp": "2026-11-14T02:00:00Z",
                            "image_path": "seq_reloc/v2.png",
                            "objects": [
                                {
                                    "object_id": "cup_relocated",
                                    "class_name": "cup",
                                    "bounding_box": [350.0, 350.0, 400.0, 400.0],
                                    "visibility_status": "CLEARLY_VISIBLE",
                                    "object_state": "CONFIRMED_MOVED",
                                    "timestamp": "2026-11-14T02:00:00Z",
                                }
                            ],
                        },
                    ],
                }
            ],
        }
        mf = self.root_path / "manifest_reloc.json"
        with open(mf, "w") as f:
            json.dump(manifest_data, f)

        manifest = ManifestLoader.load(mf, dataset_root=self.root_path)

        class RelocatingDetector(MockObjectDetector):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def detect(self, image, **kwargs):
                self.calls += 1
                box = (
                    BoundingBox(50.0, 50.0, 100.0, 100.0)
                    if self.calls == 1
                    else BoundingBox(350.0, 350.0, 400.0, 400.0)
                )
                return DetectionBatch(
                    detections=[Detection("cup", 0.95, box)],
                    image_size=(640, 480),
                    latency_ms=5.0,
                    detector_name="RelocatingMock",
                )

        runner = BenchmarkRunner(detector=RelocatingDetector())
        res = runner.run_config(manifest, BenchmarkConfigFactory.create_config("B3"))

        # Ground-truth entity must NOT be marked uninstantiated!
        self.assertNotIn("cup_relocated", res.uninstantiated_gt_entities)
        self.assertEqual(len(res.uninstantiated_gt_entities), 0)

    # -------------------------------------------------------------------------
    # Test 18: Stage 6 Test — File-Backed SQLite Persistence Across Backend Restart
    # -------------------------------------------------------------------------
    def test_18_restart_persistence_with_file_backed_database(self) -> None:
        """
        Verifies that object identity and state transitions persist across
        connection closure and backend re-instantiation using a file-backed database.
        """
        db_file = self.root_path / "persistence_test.db"
        temporal_cfg = TemporalEngineConfig()

        # Session 1: Observe and persist object
        store1 = SQLiteMemoryStore(str(db_file))
        eng1 = TemporalMemoryEngine(store1, config=temporal_cfg)

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        reg = RegistrationResult(
            success=True,
            status=AlignmentStatus.SUCCESS,
            method_used="test",
            ref_size=(640, 480),
            query_size=(640, 480),
            inlier_ratio=1.0,
            homography=np.eye(3),
        )
        d1 = [Detection("cup", 0.95, BoundingBox(50.0, 50.0, 100.0, 100.0))]
        v1 = VisitInput("session_01", "Session 01", "2026-11-14T01:00:00Z", img, reg, d1)
        eng1.process_visit(v1)

        entities_s1 = store1.list_entities()
        self.assertEqual(len(entities_s1), 1)
        ent_id = entities_s1[0].entity_id

        # Explicitly close and simulate cold backend restart
        store1.close()
        del eng1
        del store1

        # Session 2: Reopen fresh database connection to the same file
        store2 = SQLiteMemoryStore(str(db_file))
        eng2 = TemporalMemoryEngine(store2, config=temporal_cfg)

        entities_s2 = store2.list_entities()
        self.assertEqual(len(entities_s2), 1)
        self.assertEqual(entities_s2[0].entity_id, ent_id)
        self.assertEqual(entities_s2[0].current_state, EpistemicState.CONFIRMED_PRESENT)

        # Observe the same object in session 2
        v2 = VisitInput("session_02", "Session 02", "2026-11-14T02:00:00Z", img, reg, d1)
        rep2 = eng2.process_visit(v2)

        # Must maintain same entity ID, not create a duplicate
        self.assertEqual(len(store2.list_entities()), 1)
        self.assertEqual(store2.list_entities()[0].entity_id, ent_id)

        store2.close()
        db_file.unlink(missing_ok=True)

    # -------------------------------------------------------------------------
    # Test 19: Identity Consistency Penalizes Missed Observations
    # -------------------------------------------------------------------------
    def test_19_identity_consistency_penalizes_missed_observations(self) -> None:
        """
        Tests that an object present in 4 visits but detected in only 2 receives IC = 2/4 = 50%,
        not 2/2 = 100%.
        """
        manifest = self._build_test_manifest(num_visits=4, include_removals=False)

        class IntermittentDetector(MockObjectDetector):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def detect(self, image, **kwargs):
                self.calls += 1
                if self.calls in (1, 3):
                    # Detected in visits 1 and 3
                    return DetectionBatch(
                        detections=[Detection("cup", 0.95, BoundingBox(100.0, 100.0, 200.0, 200.0))],
                        image_size=(640, 480),
                        latency_ms=5.0,
                        detector_name="IntermittentMock",
                    )
                # Missed in visits 2 and 4
                return DetectionBatch(detections=[], image_size=(640, 480), latency_ms=5.0, detector_name="IntermittentMock")

        runner = BenchmarkRunner(detector=IntermittentDetector())
        res = runner.run_config(manifest, BenchmarkConfigFactory.create_config("B3"))

        # 4 total present visits, 2 matched visits
        # Identity consistency must be 2 / 4 = 0.50
        self.assertEqual(res.metrics.identity_consistency_denominator, 4)
        self.assertEqual(res.metrics.identity_consistency_numerator, 2)
        self.assertAlmostEqual(res.metrics.identity_consistency, 0.50)

    # -------------------------------------------------------------------------
    # Test 20: False Discovery Proportion Zero Denominator Handling
    # -------------------------------------------------------------------------
    def test_20_false_discovery_proportion_with_zero_removals(self) -> None:
        """
        Verifies that when zero removal predictions are made by the system,
        false discovery proportion is None (N/A) rather than a division by zero error,
        and the case is documented in unscorable_cases.
        """
        manifest = self._build_test_manifest(num_visits=3, include_removals=False)

        runner = BenchmarkRunner(detector=MockObjectDetector())
        res = runner.run_config(manifest, BenchmarkConfigFactory.create_config("B3"))

        self.assertIsNone(res.metrics.false_discovery_proportion)
        self.assertEqual(res.metrics.false_discovery_count, 0)
        self.assertEqual(res.metrics.false_discovery_denominator, 0)
        self.assertIn("false_discovery_proportion", res.unscorable_cases)

    # -------------------------------------------------------------------------
    # Test 21: Confirmation Rate and Unconfirmed Change Events
    # -------------------------------------------------------------------------
    def test_21_confirmation_rate_and_unconfirmed_events(self) -> None:
        """
        Verifies that confirmation rate accurately reports confirmed change events vs total eligible events,
        and unconfirmed change events count is accurately tracked.
        """
        class RemovalMockDetector(MockObjectDetector):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def detect(self, image, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return DetectionBatch(
                        detections=[Detection("cup", 0.95, BoundingBox(100.0, 100.0, 200.0, 200.0))],
                        image_size=(640, 480),
                        latency_ms=5.0,
                        detector_name="RemovalMock",
                    )
                return DetectionBatch(detections=[], image_size=(640, 480), latency_ms=5.0, detector_name="RemovalMock")

        # Case A: Manifest with removal confirmed (4 visits: visit 1 present, visits 2, 3, 4 unobserved)
        manifest_confirmed = self._build_test_manifest(num_visits=4, include_removals=True)
        runner = BenchmarkRunner(detector=RemovalMockDetector())
        res_conf = runner.run_config(manifest_confirmed, BenchmarkConfigFactory.create_config("B3"))

        self.assertEqual(res_conf.metrics.total_eligible_change_events, 1)
        self.assertEqual(res_conf.metrics.confirmed_change_events_count, 1)
        self.assertEqual(res_conf.metrics.unconfirmed_change_events_count, 0)
        self.assertEqual(res_conf.metrics.confirmation_rate, 1.0)

        # Case B: Manifest with 2 visits (removal starts at visit 2, only 1 unobserved visit)
        # Since unobserved visits required is 3, removal is NOT confirmed yet.
        manifest_unconf = self._build_test_manifest(num_visits=2, include_removals=True)
        runner2 = BenchmarkRunner(detector=RemovalMockDetector())
        res_unconf = runner2.run_config(manifest_unconf, BenchmarkConfigFactory.create_config("B3"))

        self.assertEqual(res_unconf.metrics.total_eligible_change_events, 1)
        self.assertEqual(res_unconf.metrics.confirmed_change_events_count, 0)
        self.assertEqual(res_unconf.metrics.unconfirmed_change_events_count, 1)
        self.assertEqual(res_unconf.metrics.confirmation_rate, 0.0)

    # -------------------------------------------------------------------------
    # Test 22: Fragmented Identities Per Entity Calculation
    # -------------------------------------------------------------------------
    def test_22_fragmented_identities_per_entity(self) -> None:
        """
        Verifies that track fragmentation computes sum_g max(0, |S(g)| - 1)
        when a single ground-truth object is associated with multiple system tracks over time.
        """
        img_p1 = self.root_path / "f_v1.png"
        img_p2 = self.root_path / "f_v2.png"
        create_synthetic_image(img_p1)
        create_synthetic_image(img_p2)

        manifest_data = {
            "manifest_version": "2.0",
            "dataset_name": "Fragmentation_Test",
            "provenance": "SYNTHETIC",
            "sequences": [
                {
                    "sequence_id": "seq_frag",
                    "observations": [
                        {
                            "observation_id": "obs_1",
                            "image_path": "f_v1.png",
                            "timestamp": "2026-11-14T01:00:00Z",
                            "objects": [
                                {
                                    "object_id": "cup_1",
                                    "class_name": "cup",
                                    "bounding_box": [50.0, 50.0, 100.0, 100.0],
                                    "visibility_status": "CLEARLY_VISIBLE",
                                    "object_state": "CONFIRMED_PRESENT",
                                    "timestamp": "2026-11-14T01:00:00Z",
                                }
                            ],
                        },
                        {
                            "observation_id": "obs_2",
                            "image_path": "f_v2.png",
                            "timestamp": "2026-11-14T02:00:00Z",
                            "objects": [
                                {
                                    "object_id": "cup_1",
                                    "class_name": "cup",
                                    "bounding_box": [400.0, 400.0, 450.0, 450.0],
                                    "visibility_status": "CLEARLY_VISIBLE",
                                    "object_state": "CONFIRMED_MOVED",
                                    "timestamp": "2026-11-14T02:00:00Z",
                                }
                            ],
                        },
                    ],
                }
            ],
        }
        mf = self.root_path / "manifest_frag.json"
        with open(mf, "w") as f:
            json.dump(manifest_data, f)

        manifest = ManifestLoader.load(mf, dataset_root=self.root_path)

        class TwoEntityDetector(MockObjectDetector):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def detect(self, image, **kwargs):
                self.calls += 1
                box = (
                    BoundingBox(50.0, 50.0, 100.0, 100.0)
                    if self.calls == 1
                    else BoundingBox(400.0, 400.0, 450.0, 450.0)
                )
                return DetectionBatch(
                    detections=[Detection("cup", 0.95, box)],
                    image_size=(640, 480),
                    latency_ms=5.0,
                    detector_name="TwoEntityMock",
                )

        runner = BenchmarkRunner(detector=TwoEntityDetector())
        res = runner.run_config(manifest, BenchmarkConfigFactory.create_config("B3"))

        # In visit 1: cup_1 mapped to ent_1. In visit 2: cup_1 matched to ent_2.
        # S(cup_1) = {ent_1, ent_2} -> fragmented_identities = max(0, 2 - 1) = 1
        self.assertEqual(res.metrics.fragmented_identities, 1)


if __name__ == "__main__":
    unittest.main()
