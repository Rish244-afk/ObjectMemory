"""
Unit and integration tests for YOLO26mDetector, DetectorRegistry integration,
coordinate decoding, class mapping, and safe model switching.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from objectmemory.api.workspace_manager import WorkspaceManager
from objectmemory.detection.exceptions import DetectorUnavailableError
from objectmemory.detection.letterbox import letterbox_image
from objectmemory.detection.registry import DetectorRegistry
from objectmemory.detection.yolo26 import YOLO26mDetector


class TestYOLO26mDetector(unittest.TestCase):
    def setUp(self) -> None:
        DetectorRegistry.reset_to_defaults()

    def test_registry_preserves_yolov8m_default_and_registers_yolo26m(self) -> None:
        self.assertEqual(DetectorRegistry.get_default_name(), "yolov8m")
        names = DetectorRegistry.list_names()
        self.assertIn("yolov8m", names)
        self.assertIn("yolov8n", names)
        self.assertIn("yolo26m", names)
        avail = {item["name"]: item for item in DetectorRegistry.list_available()}
        self.assertTrue(avail["yolov8m"]["is_default"])
        self.assertFalse(avail["yolo26m"]["is_default"])
        self.assertTrue(avail["yolo26m"]["is_available"])

    def test_missing_weights_raises_unavailable_no_silent_fallback(self) -> None:
        det = YOLO26mDetector(weights_path="models/nonexistent_yolo26m.onnx", lazy_load=True)
        self.assertFalse(det.is_available())
        self.assertIsNotNone(det.get_availability_error())
        with self.assertRaises(DetectorUnavailableError):
            det.detect(np.zeros((480, 640, 3), dtype=np.uint8))

    def test_model_info_and_hash_logging(self) -> None:
        det = YOLO26mDetector(lazy_load=True)
        info = det.model_info()
        self.assertEqual(info["architecture"], "yolo26m")
        self.assertEqual(info["weights_sha256"], YOLO26mDetector.EXPECTED_ONNX_SHA256)
        self.assertEqual(info["classes_count"], 80)
        self.assertFalse(info["is_loaded"])

    def test_decode_predictions_class_mapping_and_coordinates(self) -> None:
        det = YOLO26mDetector(lazy_load=True)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        _, lb_info = letterbox_image(img, (640, 640))

        # Construct synthetic (1, 84, 8400) tensor with 1 high-confidence 'bottle' (class 39)
        raw = np.zeros((1, 84, 8400), dtype=np.float32)
        raw[0, 0, 0] = 320.0  # cx in 640x640 letterbox
        raw[0, 1, 0] = 320.0  # cy in 640x640 letterbox -> center of 640x480 image = (320, 240)
        raw[0, 2, 0] = 100.0  # w
        raw[0, 3, 0] = 200.0  # h
        raw[0, 4 + 39, 0] = 0.88  # class 39 = bottle

        dets = det._decode_predictions(raw, lb_info, conf_thresh=0.25, nms_thresh=0.45)
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0].label, "bottle")
        self.assertAlmostEqual(dets[0].confidence, 0.88, places=4)
        self.assertAlmostEqual(dets[0].raw_bbox.x_min, 270.0, places=1)
        self.assertAlmostEqual(dets[0].raw_bbox.y_min, 140.0, places=1)
        self.assertAlmostEqual(dets[0].raw_bbox.x_max, 370.0, places=1)
        self.assertAlmostEqual(dets[0].raw_bbox.y_max, 340.0, places=1)

    def test_inference_smoke_on_physical_pilot_v01(self) -> None:
        img_path = Path("data/physical_pilot_v2/movement/V01.jpeg")
        if not img_path.exists():
            self.skipTest("Physical pilot image not found")
        img = cv2.imread(str(img_path))
        det = DetectorRegistry.get("yolo26m")
        batch = det.detect(img, confidence_threshold=0.25)
        self.assertEqual(batch.detector_name, "yolo26m")
        self.assertFalse(batch.is_synthetic)
        labels = {d.label for d in batch.detections}
        self.assertIn("bottle", labels)
        self.assertIn("scissors", labels)
        self.assertIn("cup", labels)

    def test_workspace_manager_safe_model_switching(self) -> None:
        tmp = tempfile.mkdtemp()
        try:
            mgr = WorkspaceManager(base_dir=Path(tmp))
            d_default = mgr._get_detector(None)
            self.assertEqual(d_default.model_info()["architecture"], "yolov8")
            d_26 = mgr._get_detector("yolo26m")
            self.assertEqual(d_26.model_info()["architecture"], "yolo26m")
            self.assertIn("yolo26m", mgr._detector_instances)
            # Switching to yolov8n evicts non-default yolo26m
            d_8n = mgr._get_detector("yolov8n")
            self.assertEqual(d_8n.model_info()["architecture"], "yolov8")
            self.assertNotIn("yolo26m", mgr._detector_instances)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
