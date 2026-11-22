"""
Unit tests for the Physical Workbench Dataset Loader and Integrity Validator (Phase 14C).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from objectmemory.datasets.physical_workbench import (
    AbsenceReason,
    DatasetCaptureStatus,
    PhysicalVisibility,
    PhysicalWorkbenchAnnotation,
    PhysicalWorkbenchDataset,
    detect_aruco_corner_markers,
)
from objectmemory.domain.models import BoundingBox
from objectmemory.domain.states import EpistemicState


class TestPhysicalWorkbenchDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset_dir = Path("data/physical_workbench_v1")
        if not cls.dataset_dir.exists():
            raise unittest.SkipTest("Physical workbench v1 directory not found.")
        cls.dataset = PhysicalWorkbenchDataset.load(str(cls.dataset_dir))

    def test_canonical_manifest_structure(self):
        """Verifies all 7 canonical scenarios are defined with 5 visits each."""
        self.assertEqual(len(self.dataset.sequences), 7)
        expected_scenarios = {
            "S01_stationary_baseline",
            "S02_temporary_movement_return",
            "S03_persistent_relocation",
            "S04_permanent_removal",
            "S05_partial_occlusion",
            "S06_illumination_stress",
            "S07_viewpoint_perturbation",
        }
        self.assertEqual(set(self.dataset.sequences.keys()), expected_scenarios)

        total_visits = 0
        total_annotations = 0
        for seq_id, seq in self.dataset.sequences.items():
            self.assertEqual(len(seq.visits), 5, f"Scenario {seq_id} should have exactly 5 visits")
            for v_idx, visit in enumerate(seq.visits, start=1):
                self.assertEqual(visit.visit_index, v_idx)
                total_visits += 1
                total_annotations += len(visit.annotations)

        self.assertEqual(total_visits, 35)
        self.assertEqual(total_annotations, 175)

    def test_awaiting_data_status_honesty(self):
        """Verifies that when images are not on disk, capture status is honestly AWAITING_PHYSICAL_DATA."""
        status = self.dataset.capture_status
        self.assertEqual(status, DatasetCaptureStatus.AWAITING_PHYSICAL_DATA)

    def test_validation_integrity_clean_schema(self):
        """Verifies that the canonical protocol manifest passes all semantic validation checks."""
        report = self.dataset.validate()
        self.assertEqual(report["status"], "PASSED")
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["capture_status"], "AWAITING_PHYSICAL_DATA")
        self.assertEqual(report["total_sequences"], 7)
        self.assertEqual(report["total_visits"], 35)
        self.assertEqual(report["total_annotations"], 175)

    def test_out_of_bounds_bbox_detection(self):
        """Fails when an annotation has bounding box exceeding image dimensions."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest = {
                "dataset_name": "Test OOB",
                "version": "1.0.0",
                "default_image_dimensions": [1920, 1080],
                "sequences": [
                    {
                        "sequence_id": "S01_test",
                        "visits": [
                            {
                                "visit_id": "v01",
                                "visit_index": 1,
                                "image": "images/test.jpg",
                                "annotations": [
                                    {
                                        "object_id": "cup_01",
                                        "class_label": "cup",
                                        "bbox_xyxy": [100, 100, 2000, 500],  # 2000 > 1920
                                        "visibility": "VISIBLE",
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
            m_path = os.path.join(tmp_dir, "manifest.json")
            with open(m_path, "w") as f:
                json.dump(manifest, f)

            ds = PhysicalWorkbenchDataset.load(tmp_dir)
            report = ds.validate()
            self.assertEqual(report["status"], "FAILED")
            self.assertTrue(any("exceeds bounds" in err for err in report["errors"]))

    def test_inverted_bbox_detection(self):
        """Fails when an annotation has min >= max coordinates."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest = {
                "dataset_name": "Test Inverted",
                "version": "1.0.0",
                "default_image_dimensions": [1920, 1080],
                "sequences": [
                    {
                        "sequence_id": "S01_test",
                        "visits": [
                            {
                                "visit_id": "v01",
                                "visit_index": 1,
                                "image": "images/test.jpg",
                                "annotations": [
                                    {
                                        "object_id": "cup_01",
                                        "class_label": "cup",
                                        "bbox_xyxy": [300, 100, 200, 500],  # 300 >= 200
                                        "visibility": "VISIBLE",
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
            m_path = os.path.join(tmp_dir, "manifest.json")
            with open(m_path, "w") as f:
                json.dump(manifest, f)

            ds = PhysicalWorkbenchDataset.load(tmp_dir)
            report = ds.validate()
            self.assertEqual(report["status"], "FAILED")
            self.assertTrue(any("Malformed inverted bbox" in err for err in report["errors"]))

    def test_duplicate_visit_and_non_chronological_detection(self):
        """Fails when visit indices are duplicate or non-chronological."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest = {
                "dataset_name": "Test Order",
                "version": "1.0.0",
                "default_image_dimensions": [1920, 1080],
                "sequences": [
                    {
                        "sequence_id": "S01_test",
                        "visits": [
                            {"visit_id": "v01", "visit_index": 2, "annotations": []},
                            {"visit_id": "v02", "visit_index": 1, "annotations": []},  # non-chronological
                        ],
                    }
                ],
            }
            m_path = os.path.join(tmp_dir, "manifest.json")
            with open(m_path, "w") as f:
                json.dump(manifest, f)

            ds = PhysicalWorkbenchDataset.load(tmp_dir)
            report = ds.validate()
            self.assertEqual(report["status"], "FAILED")
            self.assertTrue(any("Non-chronological visit order" in err for err in report["errors"]))

    def test_contradiction_detection_absent_with_bbox(self):
        """Fails when visibility=ABSENT is paired with a non-null bounding box."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest = {
                "dataset_name": "Test Contradiction",
                "version": "1.0.0",
                "default_image_dimensions": [1920, 1080],
                "sequences": [
                    {
                        "sequence_id": "S01_test",
                        "visits": [
                            {
                                "visit_id": "v01",
                                "visit_index": 1,
                                "annotations": [
                                    {
                                        "object_id": "bottle_01",
                                        "class_label": "bottle",
                                        "bbox_xyxy": [100, 100, 200, 300],
                                        "visibility": "ABSENT",
                                        "absence_reason": "CONFIRMED_REMOVED",
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
            m_path = os.path.join(tmp_dir, "manifest.json")
            with open(m_path, "w") as f:
                json.dump(manifest, f)

            ds = PhysicalWorkbenchDataset.load(tmp_dir)
            report = ds.validate()
            self.assertEqual(report["status"], "FAILED")
            self.assertTrue(any("visibility=ABSENT but non-null bounding box" in err for err in report["errors"]))

    def test_distinguish_absence_categories(self):
        """Verifies distinct modeling of occlusion vs insufficient evidence vs confirmed removal."""
        ann_occluded = PhysicalWorkbenchAnnotation(
            sequence_id="S05",
            visit_id="v02",
            object_id="book_01",
            class_label="book",
            bbox=BoundingBox(10, 10, 50, 50),
            visibility=PhysicalVisibility.PARTIALLY_OCCLUDED,
            absence_reason=AbsenceReason.OCCLUSION,
            expected_state=EpistemicState.CONFIRMED_PRESENT,
        )
        self.assertEqual(ann_occluded.visibility, PhysicalVisibility.PARTIALLY_OCCLUDED)
        self.assertEqual(ann_occluded.absence_reason, AbsenceReason.OCCLUSION)

        ann_removed = PhysicalWorkbenchAnnotation(
            sequence_id="S04",
            visit_id="v04",
            object_id="bottle_01",
            class_label="bottle",
            bbox=None,
            visibility=PhysicalVisibility.ABSENT,
            absence_reason=AbsenceReason.CONFIRMED_REMOVED,
            expected_state=EpistemicState.CONFIRMED_REMOVED,
        )
        self.assertEqual(ann_removed.visibility, PhysicalVisibility.ABSENT)
        self.assertEqual(ann_removed.absence_reason, AbsenceReason.CONFIRMED_REMOVED)
        self.assertEqual(ann_removed.expected_state, EpistemicState.CONFIRMED_REMOVED)

    def test_aruco_detection_on_synthetic_canvas(self):
        """Verifies ArUco corner marker detection on an image canvas."""
        canvas = np.full((500, 500, 3), 200, dtype=np.uint8)
        res = detect_aruco_corner_markers(canvas)
        self.assertTrue(res["supported"])
        self.assertEqual(res["markers_detected"], 0)

    def test_legacy_real_workbench_adapter(self):
        """Verifies the non-mutating adapter loads data/real_workbench without altering files."""
        pilot_dir = Path("data/real_workbench")
        if not pilot_dir.exists():
            self.skipTest(f"Directory not found: {pilot_dir}")

        ds = PhysicalWorkbenchDataset.load(str(pilot_dir))
        self.assertEqual(ds.capture_status, DatasetCaptureStatus.COMPLETE)
        self.assertEqual(len(ds.sequences), 1)
        self.assertIn("desk_pilot_01", ds.sequences)

        seq = ds.sequences["desk_pilot_01"]
        self.assertEqual(len(seq.visits), 7)
        total_anns = sum(len(v.annotations) for v in seq.visits)
        self.assertEqual(total_anns, 42)

        annotated_classes = ds.get_all_annotated_classes()
        self.assertIn("cup", annotated_classes)
        self.assertIn("bottle", annotated_classes)
        self.assertIn("scissors", annotated_classes)
        self.assertIn("book", annotated_classes)
        self.assertIn("potted plant", annotated_classes)

        report = ds.validate()
        self.assertEqual(report["status"], "PASSED")
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["images_found"], 7)
        self.assertEqual(report["images_missing"], 0)


if __name__ == "__main__":
    unittest.main()
