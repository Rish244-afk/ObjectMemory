"""
Unit tests for spatial coordinate normalization, displacement vectors, and 2.5D workbench projections.
"""

import math
import unittest
from objectmemory.domain.models import BoundingBox
from objectmemory.spatial.transform import (
    normalize_bbox,
    denormalize_bbox,
    compute_centroid,
    compute_displacement_vector,
    project_to_isometric,
    is_point_in_bbox,
)


class TestSpatialCoordinates(unittest.TestCase):
    def test_normalize_bbox_standard(self):
        bbox = BoundingBox(x_min=100.0, y_min=200.0, x_max=300.0, y_max=400.0)
        norm = normalize_bbox(bbox, image_width=1000, image_height=1000)
        self.assertAlmostEqual(norm[0], 0.1)
        self.assertAlmostEqual(norm[1], 0.2)
        self.assertAlmostEqual(norm[2], 0.3)
        self.assertAlmostEqual(norm[3], 0.4)

    def test_normalize_bbox_clamped(self):
        bbox = BoundingBox(x_min=0.0, y_min=100.0, x_max=1200.0, y_max=800.0)
        norm = normalize_bbox(bbox, image_width=1000, image_height=800, clamp=True)
        self.assertAlmostEqual(norm[0], 0.0)
        self.assertAlmostEqual(norm[1], 0.125)
        self.assertAlmostEqual(norm[2], 1.0)
        self.assertAlmostEqual(norm[3], 1.0)

    def test_denormalize_bbox_roundtrip(self):
        orig = BoundingBox(x_min=120.0, y_min=240.0, x_max=480.0, y_max=720.0)
        w, h = 1920, 1080
        norm = normalize_bbox(orig, w, h)
        restored = denormalize_bbox(norm, w, h)
        self.assertAlmostEqual(restored.x_min, orig.x_min, places=3)
        self.assertAlmostEqual(restored.y_min, orig.y_min, places=3)
        self.assertAlmostEqual(restored.x_max, orig.x_max, places=3)
        self.assertAlmostEqual(restored.y_max, orig.y_max, places=3)

    def test_invalid_dimensions(self):
        bbox = BoundingBox(x_min=10.0, y_min=10.0, x_max=20.0, y_max=20.0)
        with self.assertRaises(ValueError):
            normalize_bbox(bbox, image_width=0, image_height=100)
        with self.assertRaises(ValueError):
            denormalize_bbox((0.1, 0.1, 0.2, 0.2), image_width=100, image_height=-5)

    def test_centroid_and_displacement(self):
        b1 = BoundingBox(x_min=0.0, y_min=0.0, x_max=10.0, y_max=10.0)  # center (5, 5)
        b2 = BoundingBox(x_min=3.0, y_min=4.0, x_max=13.0, y_max=14.0)  # center (8, 9)
        c1 = compute_centroid(b1)
        c2 = compute_centroid(b2)
        self.assertEqual(c1, (5.0, 5.0))
        self.assertEqual(c2, (8.0, 9.0))

        dx, dy, dist = compute_displacement_vector(b1, b2)
        self.assertAlmostEqual(dx, 3.0)
        self.assertAlmostEqual(dy, 4.0)
        self.assertAlmostEqual(dist, 5.0)

    def test_isometric_projection(self):
        # Origin (0,0,0) projects to (0,0)
        iso = project_to_isometric(0.0, 0.0, 0.0)
        self.assertAlmostEqual(iso[0], 0.0)
        self.assertAlmostEqual(iso[1], 0.0)

        # Elevation z moves along y axis negatively
        iso_elevated = project_to_isometric(0.0, 0.0, 50.0)
        self.assertAlmostEqual(iso_elevated[0], 0.0)
        self.assertAlmostEqual(iso_elevated[1], -50.0)

    def test_point_in_bbox(self):
        box = BoundingBox(x_min=10.0, y_min=20.0, x_max=50.0, y_max=60.0)
        self.assertTrue(is_point_in_bbox(30.0, 40.0, box))
        self.assertTrue(is_point_in_bbox(10.0, 20.0, box))
        self.assertFalse(is_point_in_bbox(5.0, 40.0, box))
        self.assertFalse(is_point_in_bbox(30.0, 70.0, box))


if __name__ == "__main__":
    unittest.main()
