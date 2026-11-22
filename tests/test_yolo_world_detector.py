"""
Unit tests for the YOLO-World open-vocabulary detector.

Safeguards verified:
1. YOLO-World is registered alongside the baseline YOLOv8n without displacing it as default.
2. Supports dynamic text prompts via CLIP text embeddings.
3. Does not fabricate detections when weights are missing; raises DetectorUnavailableError explicitly.
4. Preserves normalization contract (clamped bboxes, valid scores, max_detections).
5. Embedding caching reuses prompt vectors across successive queries.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import numpy as np

from objectmemory.detection.exceptions import DetectorUnavailableError
from objectmemory.detection.models import DetectorConfiguration
from objectmemory.detection.registry import DetectorRegistry
from objectmemory.detection.yolo_world import YOLOWorldDetector


class TestYOLOWorldDetector(unittest.TestCase):
    """Test suite for YOLO-World detector integration and capabilities."""

    def test_yolo_world_registration_and_defaults(self) -> None:
        """Verifies that YOLO-World is registered without displacing YOLOv8n as the default."""
        names = DetectorRegistry.list_names()
        self.assertIn("yolo-world", names)
        self.assertIn("yolov8-world", names)
        self.assertIn("yolov8n", names)

        default_name = DetectorRegistry.get_default_name()
        self.assertEqual(default_name, "yolov8n", "YOLOv8n must remain the default detector.")

        available = {item["name"]: item for item in DetectorRegistry.list_available()}
        self.assertTrue(available["yolo-world"]["is_open_vocabulary"])
        self.assertFalse(available["yolov8n"]["is_open_vocabulary"])

    def test_yolo_world_model_info_and_availability(self) -> None:
        """Checks metadata and availability reporting."""
        detector = YOLOWorldDetector()
        info = detector.model_info()

        self.assertEqual(info["name"], "yolo-world")
        self.assertTrue(info["is_open_vocabulary"])
        self.assertEqual(info["framework"], "onnxruntime-cpu")
        self.assertIn("license", info)
        self.assertEqual(info["license"], "AGPL-3.0")

        if detector.is_available():
            self.assertIsNone(detector.get_availability_error())
        else:
            err = detector.get_availability_error()
            self.assertIsNotNone(err)
            self.assertIsInstance(err, str)

    def test_yolo_world_inference_on_synthetic_frame(self) -> None:
        """Runs inference on a synthetic test canvas and validates output batch structure."""
        detector = YOLOWorldDetector()
        if not detector.is_available():
            self.skipTest(f"YOLO-World dependencies or weights missing: {detector.get_availability_error()}")

        canvas = np.full((480, 640, 3), 128, dtype=np.uint8)
        cfg = DetectorConfiguration(
            confidence_threshold=0.1,
            target_classes=["notebook", "cup", "scissors"],
        )
        batch = detector.detect(canvas, config=cfg)

        self.assertEqual(batch.detector_name, "yolo-world")
        self.assertFalse(batch.is_synthetic)
        self.assertEqual(batch.image_size, (640, 480))
        self.assertGreaterEqual(batch.latency_ms, 0.0)
        self.assertIsInstance(batch.detections, list)

        for det in batch.detections:
            self.assertIn(det.label, ["notebook", "cup", "scissors"])
            self.assertGreaterEqual(det.confidence, 0.0)
            self.assertLessEqual(det.confidence, 1.0)
            self.assertGreaterEqual(det.raw_bbox.x_min, 0.0)
            self.assertGreaterEqual(det.raw_bbox.y_min, 0.0)
            self.assertLessEqual(det.raw_bbox.x_max, 640.0)
            self.assertLessEqual(det.raw_bbox.y_max, 480.0)

    def test_yolo_world_embedding_caching(self) -> None:
        """Confirms text embeddings are cached across calls with identical prompt lists."""
        detector = YOLOWorldDetector()
        if not detector.is_available():
            self.skipTest("YOLO-World not available for caching test.")

        prompts = ["bottle", "book", "pliers"]
        key = tuple(prompts)

        # Clear cache for isolated test
        detector._embedding_cache.clear()

        self.assertNotIn(key, detector._embedding_cache)
        emb1 = detector._get_text_embeddings(prompts)
        self.assertIn(key, detector._embedding_cache)

        emb2 = detector._get_text_embeddings(prompts)
        self.assertIs(emb1, emb2)

    def test_yolo_world_empty_prompts_handling(self) -> None:
        """Verifies that an empty prompt configuration returns an empty batch gracefully."""
        detector = YOLOWorldDetector()
        if not detector.is_available():
            self.skipTest("YOLO-World not available.")

        canvas = np.zeros((300, 300, 3), dtype=np.uint8)
        cfg = DetectorConfiguration(confidence_threshold=0.2, target_classes=[])
        batch = detector.detect(canvas, config=cfg)

        self.assertEqual(len(batch.detections), 0)
        self.assertEqual(batch.detector_name, "yolo-world")

    def test_yolo_world_unavailable_raises_explicit_error(self) -> None:
        """Verifies that missing weights raise DetectorUnavailableError and do not silently fall back."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            non_existent = Path(tmp_dir) / "does_not_exist.onnx"
            detector = YOLOWorldDetector(weights_path=non_existent)

            self.assertFalse(detector.is_available())
            err = detector.get_availability_error()
            self.assertIsNotNone(err)
            self.assertIn("not found", err)

            canvas = np.zeros((100, 100, 3), dtype=np.uint8)
            with self.assertRaises(DetectorUnavailableError) as ctx:
                detector.detect(canvas)

            self.assertIn("Cannot perform detection", str(ctx.exception))

    def test_yolo_world_invalid_image(self) -> None:
        """Verifies input validation on invalid or empty images."""
        detector = YOLOWorldDetector()
        if not detector.is_available():
            self.skipTest("YOLO-World not available.")

        with self.assertRaises(ValueError):
            detector.detect(None)  # type: ignore

        with self.assertRaises(ValueError):
            detector.detect(np.array([]))

    def test_yolo_world_max_detections_limit(self) -> None:
        """Verifies that max_detections is strictly enforced."""
        detector = YOLOWorldDetector()
        if not detector.is_available():
            self.skipTest("YOLO-World not available.")

        canvas = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
        cfg = DetectorConfiguration(
            confidence_threshold=0.001,
            max_detections=2,
            target_classes=["box", "paper"],
        )
        batch = detector.detect(canvas, config=cfg)
        self.assertLessEqual(len(batch.detections), 2)


if __name__ == "__main__":
    unittest.main()
