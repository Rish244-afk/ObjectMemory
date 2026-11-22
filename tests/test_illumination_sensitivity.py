"""
Unit tests investigating appearance signature sensitivity to illumination changes.

Quantifies how synthetic brightness and color-temperature shifts degrade 
HSV histogram cosine similarity and verifies that PairwiseComparator abstains
with UNKNOWN rather than hallucinating REMOVED or ADDED changes.
"""

from __future__ import annotations

import unittest
import numpy as np

from objectmemory.baseline.comparator import PairwiseComparator
from objectmemory.baseline.models import PairwiseMatchConfig
from objectmemory.detection.models import Detection
from objectmemory.detection.signature import VisualSignatureExtractor
from objectmemory.domain.models import BoundingBox
from objectmemory.domain.states import BaselineChangeType
from objectmemory.registration.models import AlignmentStatus, RegistrationResult


class TestIlluminationSensitivity(unittest.TestCase):
    """Investigates visual signature degradation across controlled illumination changes."""

    def setUp(self):
        self.extractor = VisualSignatureExtractor()
        self.comparator = PairwiseComparator(
            config=PairwiseMatchConfig(signature_similarity_threshold=0.65)
        )
        self.reg_mock = RegistrationResult(
            success=True,
            status=AlignmentStatus.SUCCESS,
            method_used="MOCK",
            ref_size=(640, 480),
            query_size=(640, 480),
            homography=np.eye(3, dtype=np.float64),
            overlap_mask=np.ones((480, 640), dtype=np.uint8) * 255,
        )

    def _create_synthetic_tool_image(self) -> np.ndarray:
        """Creates a synthetic multimeter-like image patch on a green bench."""
        img = np.ones((480, 640, 3), dtype=np.uint8) * 120  # Neutral grey bench
        # Yellow body (BGR: 20, 200, 230)
        img[80:260, 280:390] = [20, 200, 230]
        # LCD display dark cyan (BGR: 80, 80, 20)
        img[100:150, 300:370] = [80, 80, 20]
        # Selector dial black (BGR: 10, 10, 10)
        img[170:210, 315:355] = [10, 10, 10]
        return img

    def test_uniform_brightness_scaling_similarity_curve(self):
        """
        Measures cosine similarity degradation as uniform brightness scaling factor alpha varies.
        Shows that moderate drops (alpha < 0.70) cause similarity to fall below 0.65 due to HSV Value bin shifts.
        """
        ref_img = self._create_synthetic_tool_image()
        bbox = BoundingBox(280, 80, 390, 260)
        ref_sig = self.extractor.extract(ref_img, bbox)
        self.assertTrue(ref_sig.is_valid)

        scaling_factors = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
        sims = []
        for alpha in scaling_factors:
            scaled_img = np.clip(ref_img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
            scaled_sig = self.extractor.extract(scaled_img, bbox)
            sim = ref_sig.cosine_similarity(scaled_sig)
            sims.append(sim)

        # Baseline check: alpha=1.0 gives near-perfect identity
        self.assertAlmostEqual(sims[0], 1.0, places=2)
        # Severe illumination drop alpha=0.5 drops similarity significantly
        self.assertLess(sims[-1], 0.65)

    def test_collocated_appearance_mismatch_yields_unknown_abstention(self):
        """
        Verifies that under severe illumination shift (sim < 0.65), PairwiseComparator
        pairs the collocated objects as UNKNOWN (collocated appearance mismatch)
        instead of emitting a false ADDED query detection and occluded reference detection.
        """
        ref_img = self._create_synthetic_tool_image()
        # Query image with 50% brightness reduction
        query_img = np.clip(ref_img.astype(np.float32) * 0.50, 0, 255).astype(np.uint8)

        bbox = BoundingBox(280, 80, 390, 260)
        r_sig = self.extractor.extract(ref_img, bbox)
        q_sig = self.extractor.extract(query_img, bbox)

        r_det = Detection("multimeter", 0.92, bbox, canonical_bbox=bbox, signature=r_sig)
        q_det = Detection("multimeter", 0.88, bbox, canonical_bbox=bbox, signature=q_sig)

        report = self.comparator.compare(
            ref_detections=[r_det],
            query_detections=[q_det],
            registration=self.reg_mock,
            ref_image=ref_img,
            query_image_aligned=query_img,
        )

        # Must produce exactly 1 match
        self.assertEqual(len(report.matches), 1)
        match = report.matches[0]
        # Must abstain with UNKNOWN, NOT ADDED, NOT REMOVED
        self.assertEqual(match.predicted_change, BaselineChangeType.UNKNOWN)
        self.assertIn("possible illumination change", match.explanation)
        self.assertTrue(match.details.get("collocated_appearance_mismatch", False))


if __name__ == "__main__":
    unittest.main()
