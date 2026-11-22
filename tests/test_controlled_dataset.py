"""
Unit tests for the Controlled Temporal Benchmark dataset loader and validator.
"""

import os
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from objectmemory.datasets.controlled_dataset import ControlledDataset


class TestControlledDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset_dir = os.path.join(
            os.path.dirname(__file__), "..", "data", "controlled_temporal_v1"
        )
        if not os.path.exists(cls.dataset_dir):
            raise unittest.SkipTest("Controlled temporal dataset not extracted.")
        cls.dataset = ControlledDataset.load(cls.dataset_dir)

    def test_dataset_structure(self):
        self.assertEqual(len(self.dataset.sequences), 6)
        expected_seq_ids = {
            "S01_lighting_only",
            "S02_cup_move_return",
            "S03_book_occlusion",
            "S04_bottle_removal",
            "S05_phone_move_persistent",
            "S06_camera_view_nuisance",
        }
        self.assertEqual(set(self.dataset.sequences.keys()), expected_seq_ids)

        for seq_id, seq in self.dataset.sequences.items():
            self.assertEqual(len(seq.visits), 5, f"Sequence {seq_id} does not have 5 visits")

    def test_integrity_validation(self):
        report = self.dataset.validate()
        self.assertEqual(report["status"], "PASSED")
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["images_validated"], 30)
        self.assertEqual(report["annotations_validated"], 178)
        self.assertEqual(report["expected_events_validated"], 14)

    def test_image_decoding(self):
        seq = self.dataset.sequences["S01_lighting_only"]
        v1 = seq.get_visit(1)
        img = v1.load_image()
        self.assertIsInstance(img, np.ndarray)
        self.assertEqual(img.shape, (480, 640, 3))
        self.assertEqual(img.dtype, np.uint8)

    def test_expected_events_consistency(self):
        events = self.dataset.expected_events
        self.assertEqual(len(events), 14)
        s02_events = [e for e in events if e.sequence_id == "S02_cup_move_return"]
        self.assertEqual(len(s02_events), 2)
        self.assertEqual(s02_events[0].event, "object_moved")
        self.assertEqual(s02_events[0].visit_index, 3)
        self.assertEqual(s02_events[0].object_id, "cup_01")
        self.assertEqual(s02_events[1].event, "return_to_baseline")
        self.assertEqual(s02_events[1].visit_index, 4)
        self.assertEqual(s02_events[1].object_id, "cup_01")


if __name__ == "__main__":
    unittest.main()
