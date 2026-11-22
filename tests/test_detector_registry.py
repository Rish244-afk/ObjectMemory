"""
Unit tests for Pluggable Detector Architecture (Phase 2).

Validates:
1. DetectorConfiguration validation and boundary constraints.
2. DetectorRegistry registration, discovery, retrieval, and defaults.
3. Output normalization: bounding box clamping, degenerate box removal,
   target classes filtering, max_detections clipping, and metadata preservation.
4. Model availability checking and DetectorUnavailableError semantics.
5. Image input validation and channel normalization.
"""

from __future__ import annotations

import math
from pathlib import Path
import unittest
import numpy as np

from objectmemory.detection.detector import (
    MockObjectDetector,
    ObjectDetector,
    OpenCVDNNDetector,
)
from objectmemory.detection.exceptions import (
    DetectorConfigurationError,
    DetectorError,
    DetectorUnavailableError,
)
from objectmemory.detection.models import (
    Detection,
    DetectionBatch,
    DetectorConfiguration,
    VisualSignature,
)
from objectmemory.detection.registry import DetectorRegistry
from objectmemory.domain.models import BoundingBox


class TestDetectorConfiguration(unittest.TestCase):
    """Tests configuration validation and normalization."""

    def test_default_configuration_valid(self) -> None:
        cfg = DetectorConfiguration()
        self.assertAlmostEqual(cfg.confidence_threshold, 0.25)
        self.assertAlmostEqual(cfg.nms_threshold, 0.45)
        self.assertIsNone(cfg.target_classes)
        self.assertIsNone(cfg.max_detections)

    def test_custom_valid_configuration(self) -> None:
        cfg = DetectorConfiguration(
            confidence_threshold=0.6,
            nms_threshold=0.3,
            target_classes=["  multimeter ", "Screwdriver", "multimeter", ""],
            max_detections=10,
        )
        self.assertAlmostEqual(cfg.confidence_threshold, 0.6)
        self.assertAlmostEqual(cfg.nms_threshold, 0.3)
        # Deduplicated and whitespace stripped
        self.assertEqual(cfg.target_classes, ["multimeter", "Screwdriver"])
        self.assertEqual(cfg.max_detections, 10)

    def test_invalid_confidence_threshold_raises(self) -> None:
        for invalid in [-0.01, 1.01, float("nan"), float("inf")]:
            with self.subTest(invalid=invalid):
                with self.assertRaises(DetectorConfigurationError):
                    DetectorConfiguration(confidence_threshold=invalid)

    def test_invalid_nms_threshold_raises(self) -> None:
        for invalid in [-0.05, 1.5, float("nan"), float("inf")]:
            with self.subTest(invalid=invalid):
                with self.assertRaises(DetectorConfigurationError):
                    DetectorConfiguration(nms_threshold=invalid)

    def test_invalid_max_detections_raises(self) -> None:
        for invalid in [0, -5, "ten"]:
            with self.subTest(invalid=invalid):
                with self.assertRaises(DetectorConfigurationError):
                    DetectorConfiguration(max_detections=invalid)  # type: ignore


class TestDetectorRegistry(unittest.TestCase):
    """Tests DetectorRegistry lifecycle, factories, and discovery."""

    def setUp(self) -> None:
        DetectorRegistry.reset_to_defaults()

    def tearDown(self) -> None:
        DetectorRegistry.reset_to_defaults()

    def test_builtin_detectors_registered(self) -> None:
        names = DetectorRegistry.list_names()
        self.assertIn("yolov8n", names)
        self.assertIn("yolov8", names)
        self.assertIn("mock", names)
        self.assertEqual(DetectorRegistry.get_default_name(), "yolov8n")

    def test_get_default_returns_yolov8n_instance(self) -> None:
        detector = DetectorRegistry.get()
        self.assertIsInstance(detector, OpenCVDNNDetector)

    def test_get_by_name(self) -> None:
        mock_det = DetectorRegistry.get("mock")
        self.assertIsInstance(mock_det, MockObjectDetector)

    def test_get_unregistered_raises_keyerror(self) -> None:
        with self.assertRaises(KeyError) as ctx:
            DetectorRegistry.get("non_existent_detector_xyz")
        self.assertIn("non_existent_detector_xyz", str(ctx.exception))

    def test_custom_registration_and_unregister(self) -> None:
        called = {"count": 0}

        def custom_factory(**kwargs) -> ObjectDetector:
            called["count"] += 1
            return MockObjectDetector()

        DetectorRegistry.register(
            "custom_mock",
            custom_factory,
            metadata={"description": "Test custom detector", "is_open_vocabulary": True},
        )
        self.assertIn("custom_mock", DetectorRegistry.list_names())

        det = DetectorRegistry.get("custom_mock")
        self.assertIsInstance(det, MockObjectDetector)
        self.assertEqual(called["count"], 1)

        DetectorRegistry.unregister("custom_mock")
        self.assertNotIn("custom_mock", DetectorRegistry.list_names())

    def test_list_available_status(self) -> None:
        available_list = DetectorRegistry.list_available()
        self.assertGreaterEqual(len(available_list), 3)

        by_name = {item["name"]: item for item in available_list}
        self.assertTrue(by_name["mock"]["is_available"])
        self.assertTrue(by_name["mock"]["is_synthetic"])
        self.assertFalse(by_name["mock"]["is_open_vocabulary"])

        # yolov8n status matches whether models/yolov8n.onnx is on disk
        weights_exist = Path("models/yolov8n.onnx").exists()
        self.assertEqual(by_name["yolov8n"]["is_available"], weights_exist)
        self.assertTrue(by_name["yolov8n"]["is_default"])


class TestDetectorOutputNormalizationAndSafeguards(unittest.TestCase):
    """Tests detector output normalization, clamping, filtering, and metadata preservation."""

    def setUp(self) -> None:
        dummy_sig = VisualSignature(
            color_histogram=np.zeros(128, dtype=np.float32),
            texture_descriptor=np.zeros(16, dtype=np.float32),
            combined_vector=np.ones(148, dtype=np.float32) / np.sqrt(148),
        )
        self.raw_detections = [
            Detection(
                label="multimeter",
                confidence=0.92,
                raw_bbox=BoundingBox(10.0, 10.0, 100.0, 100.0),
                signature=dummy_sig,
                metadata={"source_sensor": "rgb"},
            ),
            Detection(
                label="oscilloscope",
                confidence=0.85,
                raw_bbox=BoundingBox(120.0, 30.0, 220.0, 150.0),
                canonical_bbox=BoundingBox(125.0, 35.0, 225.0, 155.0),
                metadata={"channel_count": 4},
            ),
            Detection(
                label="wire_stripper",
                confidence=0.70,
                # Out of bounds: image will be (300, 200) -> x_max 350.0 exceeds 300.0
                raw_bbox=BoundingBox(50.0, 50.0, 350.0, 180.0),
            ),
            Detection(
                label="degenerate_box",
                confidence=0.60,
                raw_bbox=BoundingBox(50.0, 50.0, 50.0, 50.0),  # Zero area
            ),
            Detection(
                label="low_conf_probe",
                confidence=0.20,
                raw_bbox=BoundingBox(10.0, 10.0, 40.0, 40.0),
            ),
        ]
        self.detector = MockObjectDetector(canned_detections=self.raw_detections)
        self.test_img = np.zeros((200, 300, 3), dtype=np.uint8)

    def test_preserves_metadata_and_signatures(self) -> None:
        cfg = DetectorConfiguration(confidence_threshold=0.5)
        batch = self.detector.detect(self.test_img, config=cfg)

        labels = {d.label: d for d in batch.detections}
        self.assertIn("multimeter", labels)
        self.assertEqual(labels["multimeter"].metadata.get("source_sensor"), "rgb")
        self.assertIsNotNone(labels["multimeter"].signature)

        self.assertIn("oscilloscope", labels)
        self.assertEqual(labels["oscilloscope"].metadata.get("channel_count"), 4)
        self.assertIsNotNone(labels["oscilloscope"].canonical_bbox)

    def test_clamps_out_of_bounds_bounding_boxes(self) -> None:
        cfg = DetectorConfiguration(confidence_threshold=0.5)
        batch = self.detector.detect(self.test_img, config=cfg)

        labels = {d.label: d for d in batch.detections}
        self.assertIn("wire_stripper", labels)
        ws = labels["wire_stripper"]
        # Raw was x_max=350, image width is 300 -> clamped to 300.0
        self.assertEqual(ws.raw_bbox.x_max, 300.0)
        self.assertTrue(ws.metadata.get("raw_bbox_clamped"))

    def test_removes_degenerate_boxes(self) -> None:
        cfg = DetectorConfiguration(confidence_threshold=0.1)
        batch = self.detector.detect(self.test_img, config=cfg)

        labels = [d.label for d in batch.detections]
        self.assertNotIn("degenerate_box", labels)

    def test_target_classes_filtering(self) -> None:
        cfg = DetectorConfiguration(
            confidence_threshold=0.5,
            target_classes=["MULTIMETER"],  # Case-insensitive
        )
        batch = self.detector.detect(self.test_img, config=cfg)

        self.assertEqual(len(batch.detections), 1)
        self.assertEqual(batch.detections[0].label, "multimeter")

    def test_max_detections_limit_preserves_highest_confidence(self) -> None:
        cfg = DetectorConfiguration(confidence_threshold=0.5, max_detections=2)
        batch = self.detector.detect(self.test_img, config=cfg)

        self.assertEqual(len(batch.detections), 2)
        # 0.92 and 0.85 should be retained, 0.70 discarded
        self.assertEqual(batch.detections[0].confidence, 0.92)
        self.assertEqual(batch.detections[1].confidence, 0.85)


class TestUnavailableDetectorAndErrorHandling(unittest.TestCase):
    """Tests error semantics when models or weights are missing or inputs are invalid."""

    def test_missing_weights_raises_detector_unavailable_error(self) -> None:
        missing_weights = Path("models/definitely_non_existent_weights_12345.onnx")
        detector = OpenCVDNNDetector(weights_path=str(missing_weights))

        self.assertFalse(detector.is_available())
        self.assertIsNotNone(detector.get_availability_error())
        self.assertIn("not found", detector.get_availability_error().lower())

        # Raises DetectorUnavailableError, which is a FileNotFoundError subclass
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        with self.assertRaises(DetectorUnavailableError) as ctx:
            detector.detect(dummy_img)

        # Confirm backward compatibility: also caught as FileNotFoundError and DetectorError
        self.assertIsInstance(ctx.exception, FileNotFoundError)
        self.assertIsInstance(ctx.exception, DetectorError)

    def test_invalid_image_inputs_raise_value_error(self) -> None:
        detector = MockObjectDetector()
        for invalid_img in [None, np.zeros((0, 0), dtype=np.uint8), np.zeros((50,), dtype=np.uint8)]:
            with self.subTest(invalid_img=invalid_img):
                with self.assertRaises(ValueError):
                    detector.detect(invalid_img)  # type: ignore

    def test_grayscale_and_rgba_inputs_handled_without_crash(self) -> None:
        # If weights exist, test that OpenCVDNNDetector processes grayscale and 4-channel images
        weights_path = Path("models/yolov8n.onnx")
        if not weights_path.exists():
            self.skipTest("models/yolov8n.onnx not present for forward pass test.")

        detector = OpenCVDNNDetector(weights_path=str(weights_path))
        gray = np.zeros((200, 200), dtype=np.uint8)
        rgba = np.zeros((200, 200, 4), dtype=np.uint8)

        batch_gray = detector.detect(gray)
        self.assertIsInstance(batch_gray, DetectionBatch)

        batch_rgba = detector.detect(rgba)
        self.assertIsInstance(batch_rgba, DetectionBatch)


if __name__ == "__main__":
    unittest.main()
