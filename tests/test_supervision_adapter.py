"""
Unit tests for Roboflow Supervision adapter integration in ObjectMemory.
Verifies conversion, coordinate consistency, label mapping, invalid boxes,
empty batches, and visualization without tracker replacement.
"""

import math
import unittest
import numpy as np
import supervision as sv

from objectmemory.domain.models import BoundingBox
from objectmemory.detection.models import Detection, DetectionBatch
from objectmemory.detection.supervision_adapter import (
    annotate_with_supervision,
    evaluate_detection_map,
    from_supervision_detections,
    to_supervision_detections,
)


class TestSupervisionAdapter(unittest.TestCase):
    def setUp(self):
        self.box1 = BoundingBox(10.0, 20.0, 100.0, 150.0)
        self.box2 = BoundingBox(120.0, 40.0, 200.0, 250.0)
        self.det1 = Detection(label="cup", confidence=0.88, raw_bbox=self.box1)
        self.det2 = Detection(label="bottle", confidence=0.92, raw_bbox=self.box2)
        self.batch = DetectionBatch(
            detections=[self.det1, self.det2],
            image_size=(640, 480),
            latency_ms=12.5,
            detector_name="test_yolo",
        )

    def test_to_supervision_basic_conversion(self):
        sv_dets = to_supervision_detections(self.batch)
        self.assertEqual(len(sv_dets), 2)
        np.testing.assert_allclose(
            sv_dets.xyxy[0], [10.0, 20.0, 100.0, 150.0], rtol=1e-5
        )
        np.testing.assert_allclose(
            sv_dets.xyxy[1], [120.0, 40.0, 200.0, 250.0], rtol=1e-5
        )
        self.assertAlmostEqual(float(sv_dets.confidence[0]), 0.88, places=2)
        self.assertAlmostEqual(float(sv_dets.confidence[1]), 0.92, places=2)
        self.assertIn("class_name", sv_dets.data)
        self.assertEqual(list(sv_dets.data["class_name"]), ["cup", "bottle"])

    def test_coordinate_consistency_roundtrip(self):
        sv_dets = to_supervision_detections(self.batch)
        restored_batch = from_supervision_detections(
            sv_dets,
            detector_name="test_yolo",
            image_size=(640, 480),
            latency_ms=12.5,
        )
        self.assertEqual(len(restored_batch.detections), 2)
        r0 = restored_batch.detections[0]
        self.assertEqual(r0.label, "cup")
        self.assertAlmostEqual(r0.confidence, 0.88, places=2)
        self.assertAlmostEqual(r0.raw_bbox.x_min, 10.0, places=4)
        self.assertAlmostEqual(r0.raw_bbox.y_min, 20.0, places=4)
        self.assertAlmostEqual(r0.raw_bbox.x_max, 100.0, places=4)
        self.assertAlmostEqual(r0.raw_bbox.y_max, 150.0, places=4)

    def test_label_mapping_with_explicit_dict(self):
        mapping = {"bottle": 0, "cup": 1}
        sv_dets = to_supervision_detections(self.batch, class_to_id=mapping)
        self.assertEqual(int(sv_dets.class_id[0]), 1)  # cup -> 1
        self.assertEqual(int(sv_dets.class_id[1]), 0)  # bottle -> 0

        # Test reverse mapping
        id_to_class = {0: "bottle", 1: "cup"}
        restored = from_supervision_detections(sv_dets, id_to_class=id_to_class)
        self.assertEqual(restored.detections[0].label, "cup")
        self.assertEqual(restored.detections[1].label, "bottle")

    def test_empty_detections_handling(self):
        empty_batch = DetectionBatch(
            detections=[],
            image_size=(640, 480),
            latency_ms=0.0,
            detector_name="empty",
        )
        sv_empty = to_supervision_detections(empty_batch)
        self.assertEqual(len(sv_empty), 0)

        restored = from_supervision_detections(sv_empty)
        self.assertEqual(len(restored.detections), 0)

    def test_multiple_same_class_objects(self):
        c1 = Detection(label="cup", confidence=0.85, raw_bbox=BoundingBox(10, 10, 50, 50))
        c2 = Detection(label="cup", confidence=0.91, raw_bbox=BoundingBox(100, 100, 150, 150))
        c3 = Detection(label="cup", confidence=0.74, raw_bbox=BoundingBox(200, 200, 250, 250))
        batch = DetectionBatch(detections=[c1, c2, c3], image_size=(640, 480), latency_ms=5.0, detector_name="multi")

        sv_dets = to_supervision_detections(batch)
        self.assertEqual(len(sv_dets), 3)
        self.assertTrue(np.all(sv_dets.data["class_name"] == "cup"))
        self.assertEqual(len(np.unique(sv_dets.class_id)), 1)
        np.testing.assert_allclose(sv_dets.confidence, [0.85, 0.91, 0.74], rtol=1e-4)

    def test_invalid_boxes_rejection(self):
        # We test that invalid or inverted boxes raise ValueError in strict mode
        # 1. Inverted coordinates
        with self.assertRaises(ValueError):
            BoundingBox(100.0, 50.0, 20.0, 150.0)

        # 2. Inverted coordinates bypassing BoundingBox
        class DummyDet:
            label = "bad"
            confidence = 0.5
            canonical_bbox = None
            class DummyBox:
                x_min, y_min, x_max, y_max = 100.0, 50.0, 20.0, 150.0
            raw_bbox = DummyBox()

        with self.assertRaises(ValueError):
            to_supervision_detections([DummyDet()])

        # 3. Non-finite coordinates
        class NanDet:
            label = "bad"
            confidence = 0.5
            canonical_bbox = None
            class NanBox:
                x_min, y_min, x_max, y_max = 10.0, float("nan"), 50.0, 50.0
            raw_bbox = NanBox()

        with self.assertRaises(ValueError):
            to_supervision_detections([NanDet()])

    def test_annotate_with_supervision(self):
        image = np.full((480, 640, 3), 128, dtype=np.uint8)
        annotated = annotate_with_supervision(image, self.batch)
        self.assertEqual(annotated.shape, (480, 640, 3))
        self.assertEqual(annotated.dtype, np.uint8)
        # Check that pixels changed due to drawing
        self.assertFalse(np.array_equal(image, annotated))

    def test_evaluate_detection_map(self):
        # Perfect detection scenario
        preds = [self.batch]
        targets = [self.batch]
        eval_res = evaluate_detection_map(preds, targets)
        self.assertAlmostEqual(eval_res["map50"], 1.0, places=2)
        self.assertAlmostEqual(eval_res["map50_95"], 1.0, places=2)
        self.assertEqual(eval_res["total_images"], 1)


if __name__ == "__main__":
    unittest.main()
