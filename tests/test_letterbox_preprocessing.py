"""
Unit and regression tests for letterbox preprocessing and coordinate transformations.

Covers:
- Image aspect ratio preservation and symmetric neutral-grey padding.
- Exact coordinate round-trip across test resolutions:
  (640x640, 1280x720, 720x1280, 1920x1080, 400x600).
- Reversal of scaling and padding within <= 1.0 pixel tolerance.
- Boundary clamping (negative and out-of-range coordinates).
- Degenerate box rejection.
- Color channel convention preservation.
- Detector output schema contracts.
"""

from __future__ import annotations

import sys
from pathlib import Path
import unittest
import numpy as np

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from objectmemory.detection.letterbox import (
    LetterboxInfo,
    inverse_transform_coordinates,
    inverse_transform_cxcywh,
    letterbox_image,
)
from objectmemory.domain.models import BoundingBox


class TestLetterboxPreprocessing(unittest.TestCase):
    def setUp(self) -> None:
        self.resolutions = [
            (640, 640),    # Square baseline
            (1280, 720),   # Widescreen 16:9 landscape
            (720, 1280),   # Portrait
            (1920, 1080),  # Full HD 16:9
            (400, 600),    # Small non-standard aspect ratio
        ]

    def test_aspect_ratio_preservation_and_padding_geometry(self) -> None:
        """
        Test A: Verify that the aspect ratio of the unpadded image region matches
        the original image aspect ratio, and total padded size is strictly 640x640.
        """
        for w, h in self.resolutions:
            with self.subTest(resolution=(w, h)):
                dummy_img = np.zeros((h, w, 3), dtype=np.uint8)
                padded, info = letterbox_image(dummy_img, target_size=(640, 640), pad_value=114)

                self.assertEqual(padded.shape, (640, 640, 3))
                self.assertEqual(info.orig_width, w)
                self.assertEqual(info.orig_height, h)
                self.assertEqual(info.target_width, 640)
                self.assertEqual(info.target_height, 640)

                # Total padding must match
                self.assertEqual(info.pad_left + info.new_unpad_width + info.pad_right, 640)
                self.assertEqual(info.pad_top + info.new_unpad_height + info.pad_bottom, 640)

                # Symmetric padding (difference between left/right and top/bottom is <= 1 px)
                self.assertLessEqual(abs(info.pad_left - info.pad_right), 1)
                self.assertLessEqual(abs(info.pad_top - info.pad_bottom), 1)

                # Aspect ratio preservation: new_unpad_w / new_unpad_h ~= w / h
                orig_aspect = float(w) / float(h)
                unpad_aspect = float(info.new_unpad_width) / float(info.new_unpad_height)
                self.assertAlmostEqual(orig_aspect, unpad_aspect, delta=0.02)

    def test_coordinate_round_trip_accuracy(self) -> None:
        """
        Test B & C: Verify that known bounding boxes in original images transform
        to model coordinates and invert back to original coordinates within <= 1.0 px tolerance.
        """
        for w, h in self.resolutions:
            dummy_img = np.zeros((h, w, 3), dtype=np.uint8)
            _, info = letterbox_image(dummy_img, target_size=(640, 640), pad_value=114)
            r = info.scale

            # Define known test boxes: center, upper-left, lower-right
            test_boxes = [
                (w * 0.25, h * 0.25, w * 0.75, h * 0.75),   # Centered box
                (w * 0.10, h * 0.10, w * 0.40, h * 0.35),   # Upper-left quadrant
                (w * 0.60, h * 0.60, w * 0.90, h * 0.85),   # Lower-right quadrant
                (10.0, 10.0, 50.0, 50.0),                   # Small object
            ]

            for orig_box in test_boxes:
                with self.subTest(resolution=(w, h), box=orig_box):
                    ox1, oy1, ox2, oy2 = orig_box

                    # Forward map to model space
                    mx1 = ox1 * r + info.pad_left
                    my1 = oy1 * r + info.pad_top
                    mx2 = ox2 * r + info.pad_left
                    my2 = oy2 * r + info.pad_top

                    # Model center, width, height
                    cx = (mx1 + mx2) / 2.0
                    cy = (my1 + my2) / 2.0
                    mw = mx2 - mx1
                    mh = my2 - my1

                    # Invert back to original coordinates via inverse_transform_cxcywh
                    recovered = inverse_transform_cxcywh(cx, cy, mw, mh, info)
                    self.assertIsNotNone(recovered)
                    rx1, ry1, rx2, ry2 = recovered

                    self.assertAlmostEqual(rx1, ox1, delta=1.0)
                    self.assertAlmostEqual(ry1, oy1, delta=1.0)
                    self.assertAlmostEqual(rx2, ox2, delta=1.0)
                    self.assertAlmostEqual(ry2, oy2, delta=1.0)

    def test_boundary_clamping_and_out_of_range(self) -> None:
        """
        Test D & E: Verify boxes near or exceeding image boundaries are clamped correctly.
        """
        for w, h in self.resolutions:
            dummy_img = np.zeros((h, w, 3), dtype=np.uint8)
            _, info = letterbox_image(dummy_img, target_size=(640, 640), pad_value=114)

            # A box extending outside the model input into negative coordinates
            recovered = inverse_transform_coordinates(-20.0, -15.0, 300.0, 200.0, info)
            self.assertIsNotNone(recovered)
            rx1, ry1, rx2, ry2 = recovered
            self.assertGreaterEqual(rx1, 0.0)
            self.assertGreaterEqual(ry1, 0.0)
            self.assertLessEqual(rx2, float(w))
            self.assertLessEqual(ry2, float(h))

            # A box extending past the model max (640, 640)
            recovered_max = inverse_transform_coordinates(200.0, 200.0, 700.0, 750.0, info)
            self.assertIsNotNone(recovered_max)
            rx1, ry1, rx2, ry2 = recovered_max
            self.assertLessEqual(rx2, float(w))
            self.assertLessEqual(ry2, float(h))

    def test_degenerate_box_rejection(self) -> None:
        """
        Test F: Verify degenerate boxes (zero width/height, collapsed coordinates, or fully in padding) are rejected.
        """
        dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
        _, info = letterbox_image(dummy_img, target_size=(640, 640), pad_value=114)

        # Collapsed line
        self.assertIsNone(inverse_transform_coordinates(100.0, 100.0, 100.0, 150.0, info))
        self.assertIsNone(inverse_transform_coordinates(100.0, 100.0, 150.0, 100.0, info))

        # Box situated entirely inside the top padding region (pad_top is 140 for 1280x720)
        # Coordinates in model space: y from 10 to 50 (entirely < pad_top)
        self.assertIsNone(inverse_transform_coordinates(200.0, 10.0, 300.0, 50.0, info))

    def test_color_channel_and_pad_value(self) -> None:
        """
        Test G: Verify padding value is strictly 114 (YOLO neutral grey) and channels are intact.
        """
        # BGR image with distinct color in corner
        img_bgr = np.full((100, 200, 3), fill_value=50, dtype=np.uint8)
        padded, info = letterbox_image(img_bgr, target_size=(640, 640), pad_value=114)

        # Padding border pixels should equal pad_value (114) across all channels
        self.assertEqual(padded[0, 0, 0], 114)
        self.assertEqual(padded[0, 0, 1], 114)
        self.assertEqual(padded[0, 0, 2], 114)

        # Center content pixel should equal original fill_value (50)
        center_y = info.pad_top + info.new_unpad_height // 2
        center_x = info.pad_left + info.new_unpad_width // 2
        self.assertEqual(padded[center_y, center_x, 0], 50)
        self.assertEqual(padded[center_y, center_x, 1], 50)
        self.assertEqual(padded[center_y, center_x, 2], 50)

    def test_public_detection_output_schema_preservation(self) -> None:
        """
        Test H: Verify Detection and BoundingBox contracts are preserved.
        """
        dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
        _, info = letterbox_image(dummy_img, target_size=(640, 640), pad_value=114)

        recovered = inverse_transform_cxcywh(320.0, 320.0, 100.0, 100.0, info)
        self.assertIsNotNone(recovered)
        bbox = BoundingBox(x_min=recovered[0], y_min=recovered[1], x_max=recovered[2], y_max=recovered[3])
        self.assertTrue(0.0 <= bbox.x_min <= bbox.x_max <= 1280.0)
        self.assertTrue(0.0 <= bbox.y_min <= bbox.y_max <= 720.0)


if __name__ == "__main__":
    unittest.main()
