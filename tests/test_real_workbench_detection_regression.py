"""
Regression tests for real-image workbench detection, diagnostic logging,
and confidence threshold sensitivity.
"""

import logging
import sys
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from objectmemory.datasets.real_dataset import RealVisitFrame
from objectmemory.detection.detector import OpenCVDNNDetector
from objectmemory.persistence.repository import SQLiteMemoryStore
from objectmemory.pipeline.real_pipeline import RealImagePipelineRunner
from objectmemory.registration.registrar import PlanarRegistrar
from objectmemory.temporal.models import TemporalEngineConfig


class TestRealWorkbenchDetectionRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.weights_path = Path("models/yolov8n.onnx")
        cls.real_img_path = Path("data/workspaces/objectmemory_real_test/images/visit_01.png")
        if not cls.weights_path.exists():
            raise unittest.SkipTest(f"Pretrained weights not found at {cls.weights_path}")
        if not cls.real_img_path.exists():
            raise unittest.SkipTest(f"Real workbench test image not found at {cls.real_img_path}")

    def test_diagnostic_logging_and_threshold_sensitivity(self):
        """
        Verifies:
        1. OpenCVDNNDetector emits structured diagnostic logs detailing model path,
           input dimensions, tensor shapes, total anchors, candidate filtering, and top classes.
        2. At threshold=0.35, reproduces zero detections (peak score in frame is ~0.324).
        3. At threshold=0.25, detects the metal drawing compass as 'scissors' (conf > 0.30).
        """
        detector = OpenCVDNNDetector(
            weights_path=str(self.weights_path),
            framework="onnx",
            architecture="yolov8",
            letterbox=False,
        )
        self.assertTrue(detector.is_available())

        img = cv2.imread(str(self.real_img_path))
        self.assertIsNotNone(img)
        h, w = img.shape[:2]
        self.assertEqual((h, w), (747, 996))

        # Test with threshold=0.35: reproduces 0 detections, logs candidate evaluation
        with self.assertLogs("objectmemory.detection.detector", level="INFO") as log_ctx:
            batch_35 = detector.detect(img, confidence_threshold=0.35)
            self.assertEqual(len(batch_35.detections), 0)

            log_output = "\n".join(log_ctx.output)
            self.assertIn("input_image_dimensions=(w=996, h=747)", log_output)
            self.assertIn("raw_output_shape=(1, 84, 8400)", log_output)
            self.assertIn("passing_confidence_threshold=0", log_output)
            self.assertIn("scissors: 0.324", log_output)
            self.assertIn("remaining_detections=0", log_output)

        # Test with threshold=0.25: successfully detects drawing compass (scissors)
        with self.assertLogs("objectmemory.detection.detector", level="INFO") as log_ctx2:
            batch_25 = detector.detect(img, confidence_threshold=0.25)
            self.assertGreaterEqual(len(batch_25.detections), 1)

            det = batch_25.detections[0]
            self.assertEqual(det.label, "scissors")
            self.assertGreater(det.confidence, 0.30)
            self.assertLess(det.confidence, 0.35)

            # Bounding box is in the lower-center workbench area
            self.assertGreater(det.raw_bbox.x_min, 200.0)
            self.assertLess(det.raw_bbox.x_max, 700.0)
            self.assertGreater(det.raw_bbox.y_min, 300.0)
            self.assertLess(det.raw_bbox.y_max, 700.0)

            log_output2 = "\n".join(log_ctx2.output)
            self.assertIn("passing_confidence_threshold=3", log_output2)
            self.assertIn("remaining_detections=1", log_output2)

    def test_pipeline_end_to_end_observation_at_threshold_25(self):
        """
        Verifies that running RealImagePipelineRunner on the real workbench photograph
        with confidence_threshold=0.25 establishes an active entity belief and records
        the localized observation without fabricating data.
        """
        detector = OpenCVDNNDetector(
            weights_path=str(self.weights_path),
            framework="onnx",
            architecture="yolov8",
            letterbox=False,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test_memory.db"
            store = SQLiteMemoryStore(db_path, check_same_thread=False)
            try:
                runner = RealImagePipelineRunner(
                    store=store,
                    detector=detector,
                    registrar=PlanarRegistrar(),
                    temporal_config=TemporalEngineConfig(),
                )

                visit_frame = RealVisitFrame(
                    visit_id="visit_01",
                    session_name="Real Workbench Baseline",
                    timestamp="2026-10-20T12:00:00Z",
                    image_path=self.real_img_path,
                )

                result = runner.process_visit(visit_frame, confidence_threshold=0.25)

                # Detections should contain the drawing compass
                self.assertEqual(len(result.detection_batch.detections), 1)
                self.assertEqual(result.detection_batch.detections[0].label, "scissors")

                # Entity should be established in SQLite store
                entities = store.list_entities()
                self.assertEqual(len(entities), 1)
                self.assertEqual(entities[0].label, "scissors")
                self.assertEqual(entities[0].current_state.value, "CONFIRMED_PRESENT")

                # Observation should be persisted
                observations = store.list_observations_for_session("visit_01")
                self.assertEqual(len(observations), 1)
                self.assertEqual(observations[0].entity_id, entities[0].entity_id)
                self.assertIsNotNone(observations[0].raw_bbox)
                self.assertIsNotNone(observations[0].canonical_bbox)
            finally:
                store.close()

    def test_letterboxed_real_workbench_detection(self):
        """
        Verifies OpenCVDNNDetector in default letterbox=True mode on the real workbench image.
        Ensures aspect ratio is preserved, inverse coordinates are valid, and letterbox metadata is recorded.
        """
        detector = OpenCVDNNDetector(
            weights_path=str(self.weights_path),
            framework="onnx",
            architecture="yolov8",
            letterbox=True,
        )
        self.assertTrue(detector.is_available())

        img = cv2.imread(str(self.real_img_path))
        self.assertIsNotNone(img)

        batch = detector.detect(img, confidence_threshold=0.20)
        self.assertIsInstance(batch.detections, list)
        for det in batch.detections:
            self.assertIn("letterbox_info", det.metadata)
            info = det.metadata["letterbox_info"]
            self.assertEqual(info["orig_width"], 996)
            self.assertEqual(info["orig_height"], 747)
            self.assertEqual(info["target_width"], 640)
            self.assertEqual(info["target_height"], 640)
            # Ensure coordinates are within image boundaries
            self.assertGreaterEqual(det.raw_bbox.x_min, 0.0)
            self.assertGreaterEqual(det.raw_bbox.y_min, 0.0)
            self.assertLessEqual(det.raw_bbox.x_max, 996.0)
            self.assertLessEqual(det.raw_bbox.y_max, 747.0)


if __name__ == "__main__":
    unittest.main()
