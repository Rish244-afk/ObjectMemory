"""
Automated unit tests for Phase 3: Object Detection and Visual Signature Extraction.

Validates:
1. Coordinate transformations for bounding boxes across homographies.
2. Visual signature properties (normalization, dimension, identity, contrast).
3. Degenerate and boundary crop handling.
4. Pluggable detector interfaces (MockObjectDetector and OpenCVDNNDetector).
5. Pipeline execution, confidence filtering, and Domain Observation conversion.
"""

import math
import sys
import unittest
from pathlib import Path
import cv2
import numpy as np

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from objectmemory.detection.detector import (
    MockObjectDetector,
    OpenCVDNNDetector,
)
from objectmemory.detection.models import Detection, VisualSignature
from objectmemory.detection.pipeline import (
    DetectionPipeline,
    transform_bbox_to_canonical,
)
from objectmemory.detection.signature import VisualSignatureExtractor
from objectmemory.domain.models import BoundingBox
from objectmemory.domain.states import VisibilityStatus


class TestBoundingBoxTransformation(unittest.TestCase):

    def test_bbox_transformation_identity(self):
        """Under identity homography, canonical bbox equals raw bbox."""
        bbox = BoundingBox(50.0, 60.0, 150.0, 160.0)
        H_eye = np.eye(3, dtype=np.float64)
        canonical = transform_bbox_to_canonical(bbox, H_eye, ref_size=(400, 300))

        self.assertIsNotNone(canonical)
        self.assertAlmostEqual(canonical.x_min, 50.0)
        self.assertAlmostEqual(canonical.y_min, 60.0)
        self.assertAlmostEqual(canonical.x_max, 150.0)
        self.assertAlmostEqual(canonical.y_max, 160.0)

    def test_bbox_transformation_translation(self):
        """Under pure translation H, canonical coordinates shift accordingly."""
        bbox = BoundingBox(50.0, 60.0, 150.0, 160.0)
        # Shift query by (-20, -10) into reference space
        H_trans = np.array([[1.0, 0.0, -20.0], [0.0, 1.0, -10.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        canonical = transform_bbox_to_canonical(bbox, H_trans, ref_size=(400, 300))

        self.assertIsNotNone(canonical)
        self.assertAlmostEqual(canonical.x_min, 30.0)
        self.assertAlmostEqual(canonical.y_min, 50.0)
        self.assertAlmostEqual(canonical.x_max, 130.0)
        self.assertAlmostEqual(canonical.y_max, 150.0)

    def test_bbox_transformation_out_of_bounds_returns_none(self):
        """Bounding box projected entirely outside of reference canvas returns None."""
        bbox = BoundingBox(10.0, 10.0, 50.0, 50.0)
        # Shift by +1000 pixels far outside a (400, 300) canvas
        H_far = np.array([[1.0, 0.0, 1000.0], [0.0, 1.0, 1000.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        canonical = transform_bbox_to_canonical(bbox, H_far, ref_size=(400, 300))
        self.assertIsNone(canonical)

    def test_bbox_transformation_non_finite_returns_none(self):
        """Homography containing NaN or Inf returns None gracefully."""
        bbox = BoundingBox(10.0, 10.0, 50.0, 50.0)
        H_nan = np.eye(3)
        H_nan[0, 2] = float("nan")
        self.assertIsNone(transform_bbox_to_canonical(bbox, H_nan, ref_size=(400, 300)))


class TestVisualSignatureExtraction(unittest.TestCase):

    def setUp(self):
        self.extractor = VisualSignatureExtractor()
        # Create a synthetic image with distinct yellow and green colored patches
        self.img = np.zeros((200, 200, 3), dtype=np.uint8)
        # Yellow region with vertical stripe texture (simulated multimeter)
        self.img[10:90, 10:90] = (20, 200, 230)
        for x in range(15, 85, 10):
            cv2.line(self.img, (x, 10), (x, 90), (0, 0, 0), 2)

        # Green region with horizontal stripe texture (simulated PCB)
        self.img[110:190, 110:190] = (30, 150, 40)
        for y in range(115, 185, 10):
            cv2.line(self.img, (110, y), (190, y), (255, 255, 255), 2)

    def test_signature_normalization_and_dimensions(self):
        """Signature combined vector is L2-normalized and finite."""
        bbox = BoundingBox(10.0, 10.0, 90.0, 90.0)
        sig = self.extractor.extract(self.img, bbox)

        self.assertIsInstance(sig, VisualSignature)
        self.assertTrue(np.all(np.isfinite(sig.combined_vector)))
        self.assertAlmostEqual(float(np.linalg.norm(sig.combined_vector)), 1.0, places=5)
        # Standard dim: 128 (color) + 16 (gradient) + 4 (stats) = 148
        self.assertEqual(len(sig.combined_vector), 148)

    def test_signature_similarity_identical_crops(self):
        """Extracting signatures from identical regions yields similarity ~= 1.0."""
        bbox = BoundingBox(10.0, 10.0, 90.0, 90.0)
        sig1 = self.extractor.extract(self.img, bbox)
        sig2 = self.extractor.extract(self.img, bbox)

        sim = sig1.cosine_similarity(sig2)
        self.assertAlmostEqual(sim, 1.0, places=4)

    def test_signature_contrast_different_colors(self):
        """Signatures from yellow vs. green patches have significantly lower similarity."""
        yellow_box = BoundingBox(10.0, 10.0, 90.0, 90.0)
        green_box = BoundingBox(110.0, 110.0, 190.0, 190.0)

        sig_yellow = self.extractor.extract(self.img, yellow_box)
        sig_green = self.extractor.extract(self.img, green_box)

        sim = sig_yellow.cosine_similarity(sig_green)
        self.assertLess(sim, 0.4)

    def test_signature_degenerate_crop_handling(self):
        """Degenerate 0x0 or out-of-bounds crops return valid fallback without crashing."""
        degenerate_box = BoundingBox(0.0, 0.0, 1.0, 1.0)
        sig = self.extractor.extract(self.img, degenerate_box)

        self.assertIsNotNone(sig)
        self.assertTrue(np.all(np.isfinite(sig.combined_vector)))
        self.assertAlmostEqual(float(np.linalg.norm(sig.combined_vector)), 1.0, places=5)


class TestDetectionPipelineAndInterfaces(unittest.TestCase):

    def setUp(self):
        self.mock_detections = [
            Detection(label="multimeter", confidence=0.88, raw_bbox=BoundingBox(10.0, 10.0, 90.0, 90.0)),
            Detection(label="soldering_iron", confidence=0.75, raw_bbox=BoundingBox(100.0, 20.0, 180.0, 80.0)),
            Detection(label="probe", confidence=0.30, raw_bbox=BoundingBox(50.0, 50.0, 70.0, 70.0)),  # Low conf
        ]
        self.mock_detector = MockObjectDetector(self.mock_detections)
        self.pipeline = DetectionPipeline(self.mock_detector)
        self.test_img = np.ones((200, 200, 3), dtype=np.uint8) * 128

    def test_mock_detector_flags_synthetic_and_filters_confidence(self):
        """Mock detector clearly sets is_synthetic=True and filters below threshold."""
        batch = self.mock_detector.detect(self.test_img, confidence_threshold=0.5)

        self.assertTrue(batch.is_synthetic)
        self.assertEqual(len(batch.detections), 2)  # 0.88 and 0.75 retained, 0.30 filtered
        labels = [d.label for d in batch.detections]
        self.assertIn("multimeter", labels)
        self.assertIn("soldering_iron", labels)
        self.assertNotIn("probe", labels)

    def test_pipeline_with_homography_coordinates(self):
        """Pipeline populates canonical_bbox and appearance signatures."""
        H = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 20.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        batch = self.pipeline.process_frame(
            self.test_img,
            homography=H,
            ref_size=(300, 300),
            confidence_threshold=0.5,
        )

        for det in batch.detections:
            self.assertIsNotNone(det.signature)
            self.assertIsNotNone(det.canonical_bbox)
            # Check translation shift (+10, +20)
            self.assertAlmostEqual(det.canonical_bbox.x_min, det.raw_bbox.x_min + 10.0)
            self.assertAlmostEqual(det.canonical_bbox.y_min, det.raw_bbox.y_min + 20.0)

    def test_pipeline_without_homography_leaves_canonical_none(self):
        """When homography is None (unaligned/failed), canonical_bbox is None."""
        batch = self.pipeline.process_frame(self.test_img, homography=None, confidence_threshold=0.5)

        for det in batch.detections:
            self.assertIsNotNone(det.raw_bbox)
            self.assertIsNone(det.canonical_bbox)

    def test_conversion_to_domain_observations(self):
        """Converts DetectionBatch into Observation records without manufacturing scores."""
        batch = self.pipeline.process_frame(self.test_img, confidence_threshold=0.5)
        observations = self.pipeline.to_observations(
            batch,
            session_id="sess_t1",
            timestamp="2026-10-10T12:00:00Z",
            image_reference="frame_t1.jpg",
            alignment_quality=0.92,
        )

        self.assertEqual(len(observations), 2)
        obs0 = observations[0]
        self.assertEqual(obs0.session_id, "sess_t1")
        self.assertEqual(obs0.detector_confidence, 0.88)
        self.assertIsNone(obs0.identity_match_score)  # Must NOT be manufactured
        self.assertEqual(obs0.alignment_quality, 0.92)
        self.assertEqual(obs0.raw_evidence["label"], "multimeter")
        self.assertTrue(obs0.raw_evidence["is_synthetic"])

    def test_opencvdnn_detector_missing_weights_raises_filenotfound(self):
        """OpenCVDNNDetector raises descriptive FileNotFoundError when model files do not exist."""
        non_existent_weights = "C:/models/non_existent_yolo.onnx"
        detector = OpenCVDNNDetector(weights_path=non_existent_weights)

        self.assertFalse(detector.is_available())
        with self.assertRaises(FileNotFoundError):
            detector.detect(self.test_img)

    def test_opencvdnn_detector_model_info(self):
        """Model info accurately documents backend, license, and known limitations."""
        detector = OpenCVDNNDetector()
        info = detector.model_info()

        self.assertEqual(info["backend"], "OpenCV DNN (CPU)")
        self.assertIn("limitations", info)
        self.assertGreater(len(info["limitations"]), 0)


if __name__ == "__main__":
    unittest.main()
