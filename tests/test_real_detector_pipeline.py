"""
Deterministic unit and integration tests for Phase 6: Real-Image Validation and Detector Integration.

Verifies:
1. Real OpenCV DNN CPU inference on real ONNX weights (yolov8n.onnx) without mocking.
2. Detector handles missing weights via FileNotFoundError.
3. Detector rejects invalid images via ValueError.
4. Confidence threshold filtering on real detections.
5. RealWorkbenchDataset loading, schema validation, and template generation.
6. RealBenchmarkEvaluator unannotated dataset status reporting (BLOCKED_PENDING_ANNOTATIONS).
7. End-to-end integration: RealImagePipelineRunner execution through registration,
   canonical projection, signature extraction, temporal belief updating, and SQLite persistence.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from objectmemory.benchmarks.real_evaluator import RealBenchmarkEvaluator
from objectmemory.datasets.real_dataset import (
    RealVisitAnnotation,
    RealVisitFrame,
    RealWorkbenchDataset,
)
from objectmemory.detection.detector import COCO_CLASSES_80, OpenCVDNNDetector
from objectmemory.domain.models import BoundingBox
from objectmemory.domain.states import EpistemicState
from objectmemory.persistence.repository import SQLiteMemoryStore
from objectmemory.pipeline.real_pipeline import RealImagePipelineRunner


class TestRealDetectorAndPipeline(unittest.TestCase):
    """Test suite for OpenCV DNN CPU detector and real image pipeline."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.weights_path = Path("models/yolov8n.onnx")
        cls.demo_img_path = Path("demo_alignment_overlay.png")

    def test_real_yolov8_onnx_inference_on_cpu(self) -> None:
        """
        1. Executes real pretrained YOLOv8 nano inference on CPU via OpenCV DNN.
        Verifies valid detections, COCO label taxonomy, bounded confidences, and finite latency.
        """
        self.assertTrue(self.weights_path.exists(), "Pretrained weights models/yolov8n.onnx must exist.")
        self.assertTrue(self.demo_img_path.exists(), "demo_alignment_overlay.png must exist.")

        detector = OpenCVDNNDetector(
            weights_path=str(self.weights_path),
            framework="onnx",
            architecture="yolov8",
        )
        self.assertTrue(detector.is_available())

        img = cv2.imread(str(self.demo_img_path))
        self.assertIsNotNone(img)

        batch = detector.detect(img, confidence_threshold=0.35)
        self.assertFalse(batch.is_synthetic)
        self.assertEqual(batch.detector_name, "OpenCVDNN_YOLOV8")
        self.assertGreater(batch.latency_ms, 0.0)
        self.assertGreater(len(batch.detections), 0)

        for det in batch.detections:
            self.assertIn(det.label, COCO_CLASSES_80)
            self.assertGreaterEqual(det.confidence, 0.35)
            self.assertLessEqual(det.confidence, 1.0)
            self.assertGreater(det.raw_bbox.x_max - det.raw_bbox.x_min, 0)
            self.assertGreater(det.raw_bbox.y_max - det.raw_bbox.y_min, 0)

    def test_detector_missing_weights_raises_filenotfound(self) -> None:
        """
        2. Verifies detector raises FileNotFoundError when weights are missing,
        rather than inventing fake detections.
        """
        detector = OpenCVDNNDetector(weights_path="models/non_existent_weights.onnx")
        self.assertFalse(detector.is_available())

        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        with self.assertRaises(FileNotFoundError):
            detector.detect(dummy_img)

    def test_detector_invalid_image_raises_value_error(self) -> None:
        """
        3. Verifies detector validates input images and rejects empty or invalid arrays.
        """
        detector = OpenCVDNNDetector(weights_path=str(self.weights_path))

        with self.assertRaises(ValueError):
            detector.detect(np.array([]))

        with self.assertRaises(ValueError):
            detector.detect(None)  # type: ignore

    def test_detector_confidence_threshold_filtering(self) -> None:
        """
        4. Verifies increasing confidence threshold monotonically reduces or preserves detections.
        """
        detector = OpenCVDNNDetector(weights_path=str(self.weights_path))
        img = cv2.imread(str(self.demo_img_path))
        self.assertIsNotNone(img)

        low_conf_batch = detector.detect(img, confidence_threshold=0.30)
        high_conf_batch = detector.detect(img, confidence_threshold=0.70)

        self.assertGreaterEqual(len(low_conf_batch.detections), len(high_conf_batch.detections))
        for det in high_conf_batch.detections:
            self.assertGreaterEqual(det.confidence, 0.70)

    def test_real_dataset_loader_and_schema_validation(self) -> None:
        """
        5. Verifies RealWorkbenchDataset loading, template creation, and schema validation.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            ds_dir = Path(tmp_dir) / "workbench_ds"
            template_manifest = RealWorkbenchDataset.create_empty_template(ds_dir)
            self.assertTrue(template_manifest.exists())

            # Load dataset
            dataset = RealWorkbenchDataset.load_from_directory(ds_dir)
            self.assertEqual(dataset.dataset_name, "laboratory_workbench_template")
            self.assertEqual(len(dataset.visits), 4)
            self.assertTrue(dataset.is_annotated)

            ready, reason = dataset.is_evaluation_ready()
            self.assertFalse(ready)
            self.assertTrue("placeholder" in reason.lower() or "template" in reason.lower() or "missing image" in reason.lower())

    def test_real_evaluator_unannotated_status_reporting(self) -> None:
        """
        6. Verifies RealBenchmarkEvaluator emits BLOCKED_PENDING_ANNOTATIONS status
        when manual ground-truth annotations are not provided.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            # Create synthetic frames to test evaluator status reporting
            img1 = np.full((300, 400, 3), 120, dtype=np.uint8)
            cv2.circle(img1, (100, 100), 20, (0, 0, 255), -1)
            cv2.circle(img1, (250, 150), 30, (0, 255, 0), -1)
            img1_path = tmp_path / "v1.png"
            cv2.imwrite(str(img1_path), img1)

            img2 = np.full((300, 400, 3), 120, dtype=np.uint8)
            cv2.circle(img2, (105, 102), 20, (0, 0, 255), -1)
            cv2.circle(img2, (250, 150), 30, (0, 255, 0), -1)
            img2_path = tmp_path / "v2.png"
            cv2.imwrite(str(img2_path), img2)

            visits = [
                RealVisitFrame("v1", "Session 1", "2026-03-01T10:00:00Z", img1_path),
                RealVisitFrame("v2", "Session 2", "2026-03-01T14:00:00Z", img2_path),
            ]
            dataset = RealWorkbenchDataset("unannotated_set", "Test set", tmp_path, visits)

            ready, reason = dataset.is_evaluation_ready()
            self.assertFalse(ready)
            self.assertIn("No manual ground-truth annotations found", reason)

            db_path = tmp_path / "eval.db"
            with SQLiteMemoryStore(db_path) as store:
                detector = OpenCVDNNDetector(weights_path=str(self.weights_path))
                runner = RealImagePipelineRunner(store=store, detector=detector)
                evaluator = RealBenchmarkEvaluator()
                report = evaluator.evaluate(dataset, runner)

                self.assertEqual(report.status, "BLOCKED_PENDING_ANNOTATIONS")
                self.assertIsNotNone(report.blocking_reason)
                self.assertIsNotNone(report.cpu_profile)
                self.assertGreaterEqual(report.cpu_profile.mean_total_pipeline_latency_ms, 0.0)

    def test_end_to_end_real_pipeline_execution(self) -> None:
        """
        7. Full integration test: Ingests real image frames through registration,
        real CPU detection, canonical bounding box projection, appearance signature extraction,
        temporal memory belief updates, and atomic SQLite persistence.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            db_path = tmp_path / "test_pipeline.db"

            # Create two visit images with distinct visual features
            img1 = cv2.imread(str(self.demo_img_path))
            self.assertIsNotNone(img1)
            h, w = img1.shape[:2]

            img1_path = tmp_path / "visit1.png"
            cv2.imwrite(str(img1_path), img1)

            # Visit 2: slight translation (dx=5, dy=3)
            M = np.float32([[1, 0, 5], [0, 1, 3]])
            img2 = cv2.warpAffine(img1, M, (w, h))
            img2_path = tmp_path / "visit2.png"
            cv2.imwrite(str(img2_path), img2)

            visits = [
                RealVisitFrame("v1", "Session 1", "2026-03-01T10:00:00Z", img1_path),
                RealVisitFrame("v2", "Session 2", "2026-03-01T14:00:00Z", img2_path),
            ]
            dataset = RealWorkbenchDataset("e2e_set", "E2E set", tmp_path, visits)

            with SQLiteMemoryStore(db_path) as store:
                detector = OpenCVDNNDetector(weights_path=str(self.weights_path))
                runner = RealImagePipelineRunner(store=store, detector=detector)

                ds_report = runner.process_dataset(dataset, confidence_threshold=0.50)

                self.assertEqual(ds_report.total_visits_processed, 2)
                self.assertEqual(len(ds_report.visit_results), 2)

                # Verify SQLite database persistence
                sessions = store.list_sessions()
                self.assertEqual(len(sessions), 2)
                self.assertEqual(sessions[0].session_id, "v1")
                self.assertEqual(sessions[1].session_id, "v2")

                entities = store.list_entities()
                self.assertGreater(len(entities), 0)

                # Verify each entity has valid epistemic state and observations
                for entity in entities:
                    self.assertIn(
                        entity.current_state,
                        [
                            EpistemicState.CONFIRMED_PRESENT,
                            EpistemicState.CONFIRMED_MOVED,
                            EpistemicState.PENDING_REMOVAL,
                            EpistemicState.INSUFFICIENT_EVIDENCE,
                            EpistemicState.AMBIGUOUS_IDENTITY,
                        ],
                    )
                    obs = store.list_observations_for_entity(entity.entity_id)
                    self.assertGreater(len(obs), 0)
                    for o in obs:
                        self.assertIsNotNone(o.raw_evidence)
                        self.assertIn("signature_vector", o.raw_evidence)
                        self.assertEqual(len(o.raw_evidence["signature_vector"]), 148)
                        self.assertGreaterEqual(o.detector_confidence, 0.50)


if __name__ == "__main__":
    unittest.main()
