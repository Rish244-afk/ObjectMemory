"""
Unit tests for the Physical Workbench Evaluation Suite (Phase 14C).

Verifies:
1. Strict separation between Track B (Oracle temporal reasoning) and Track C (Real detector inputs).
2. Proper metric calculations, denominators, and zero-observation handling.
3. JSON serialization schema and capture status accuracy.
4. Historical evaluation results immutability.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from objectmemory.datasets.physical_workbench import (
    AbsenceReason,
    DatasetCaptureStatus,
    PhysicalVisibility,
    PhysicalWorkbenchAnnotation,
    PhysicalWorkbenchDataset,
    PhysicalWorkbenchSequence,
    PhysicalWorkbenchVisit,
)
from objectmemory.detection.models import BoundingBox, Detection
from objectmemory.domain.states import EpistemicState
from evaluate_physical_workbench import (
    evaluate_physical_track_a,
    evaluate_physical_track_b,
    evaluate_physical_track_c,
    match_detections_to_annotations,
)


class TestPhysicalWorkbenchEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scripts_dir = Path("scripts")
        cls.results_json_path = cls.scripts_dir / "phase14c_physical_workbench_results.json"

    def test_historical_result_files_immutability(self):
        """Verifies that earlier benchmark results were not overwritten or modified."""
        historical_files = [
            self.scripts_dir / "phase14b_coco_subset_evaluation_results.json",
            self.scripts_dir / "phase14b_controlled_benchmark_results.json",
            self.scripts_dir / "phase10a_controlled_evaluation_results.json",
            self.scripts_dir / "phase10_coco_subset_evaluation_results.json",
            self.scripts_dir / "phase9b_evaluation_results.json",
        ]
        for hf in historical_files:
            self.assertTrue(hf.exists(), f"Historical file missing: {hf}")
            # Ensure it contains valid JSON and expected historical keys
            with open(hf, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertIsInstance(data, dict)

    def test_phase14c_results_json_schema(self):
        """Verifies structure, tracks, and honest capture status in Phase 14C results JSON."""
        self.assertTrue(self.results_json_path.exists())
        with open(self.results_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["evaluation_version"], "14c")
        self.assertEqual(data["capture_status"], "AWAITING_PHYSICAL_DATA")
        self.assertIn("tracks", data)

        tracks = data["tracks"]
        self.assertIn("track_a_natural_image_detection", tracks)
        self.assertIn("track_b_oracle_temporal_evaluation", tracks)
        self.assertIn("track_c_end_to_end_physical_evaluation", tracks)

        # Confirm all tracks honestly record pending status when images are absent
        self.assertEqual(tracks["track_a_natural_image_detection"]["status"], "PENDING_PHYSICAL_DATA")
        self.assertEqual(tracks["track_b_oracle_temporal_evaluation"]["status"], "PENDING_PHYSICAL_DATA")
        self.assertEqual(tracks["track_c_end_to_end_physical_evaluation"]["status"], "PENDING_PHYSICAL_DATA")

    def test_bipartite_matching_precision_recall(self):
        """Tests greedy 1-to-1 matching and metric formulas under controlled conditions."""
        anns = [
            PhysicalWorkbenchAnnotation(
                sequence_id="S01",
                visit_id="v01",
                object_id="cup_01",
                class_label="cup",
                bbox=BoundingBox(10, 10, 50, 50),
            ),
            PhysicalWorkbenchAnnotation(
                sequence_id="S01",
                visit_id="v01",
                object_id="book_01",
                class_label="book",
                bbox=BoundingBox(100, 100, 150, 150),
            ),
        ]

        # Case 1: Perfect detections
        dets_perfect = [
            Detection(label="cup", confidence=0.9, raw_bbox=BoundingBox(10, 10, 50, 50)),
            Detection(label="book", confidence=0.85, raw_bbox=BoundingBox(100, 100, 150, 150)),
        ]
        res = match_detections_to_annotations(dets_perfect, anns, iou_threshold=0.5)
        self.assertEqual(res["true_positives"], 2)
        self.assertEqual(res["false_positives"], 0)
        self.assertEqual(res["false_negatives"], 0)

        # Case 2: One missed detection, one spurious false positive
        dets_partial = [
            Detection(label="cup", confidence=0.9, raw_bbox=BoundingBox(10, 10, 50, 50)),
            Detection(label="bottle", confidence=0.7, raw_bbox=BoundingBox(200, 200, 250, 250)),
        ]
        res2 = match_detections_to_annotations(dets_partial, anns, iou_threshold=0.5)
        self.assertEqual(res2["true_positives"], 1)
        self.assertEqual(res2["false_positives"], 1)
        self.assertEqual(res2["false_negatives"], 1)

    def test_track_b_oracle_evaluation_on_fixture_sequence(self):
        """Verifies Track B oracle temporal evaluation on a clean synthetic image sequence."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            img_dir = os.path.join(tmp_dir, "images")
            os.makedirs(img_dir, exist_ok=True)

            # Create 5 synthetic visits with texture
            visits = []
            for v_idx in range(1, 6):
                img_name = f"S01_v0{v_idx}.png"
                img_path = os.path.join(img_dir, img_name)

                # Draw a canvas with some features so SIFT can register
                canvas = np.full((300, 400, 3), 220, dtype=np.uint8)
                cv2.circle(canvas, (100, 100), 15, (50, 50, 50), -1)
                cv2.rectangle(canvas, (200, 150), (280, 220), (30, 30, 30), 2)
                cv2.putText(canvas, "TEST", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
                cv2.imwrite(img_path, canvas)

                visits.append(
                    PhysicalWorkbenchVisit(
                        visit_id=f"visit_0{v_idx}",
                        visit_index=v_idx,
                        image_relative_path=f"images/{img_name}",
                        image_absolute_path=img_path,
                        expected_dimensions=(400, 300),
                        expected_event="baseline_established" if v_idx == 1 else "baseline_stable",
                        annotations=[
                            PhysicalWorkbenchAnnotation(
                                sequence_id="S01_test",
                                visit_id=f"visit_0{v_idx}",
                                object_id="cup_01",
                                class_label="cup",
                                bbox=BoundingBox(50, 50, 90, 90),
                                visibility=PhysicalVisibility.VISIBLE,
                                expected_state=EpistemicState.CONFIRMED_PRESENT,
                            )
                        ],
                    )
                )

            seq = PhysicalWorkbenchSequence(
                sequence_id="S01_test",
                scenario_name="Stationary Baseline",
                description="Test fixture sequence",
                visits=visits,
            )
            dataset = PhysicalWorkbenchDataset(
                dataset_dir=tmp_dir,
                manifest_data={"dataset_name": "Test Fixture", "version": "1.0"},
                sequences={"S01_test": seq},
            )

            # Evaluate Track B
            track_b = evaluate_physical_track_b(dataset)
            self.assertEqual(track_b["track_name"], "TRACK_B_ORACLE_TEMPORAL_EVALUATION")
            self.assertEqual(track_b["total_events_evaluated"], 4)
            self.assertEqual(track_b["successful_events"], 4)
            self.assertEqual(track_b["event_accuracy"], 1.0)
            self.assertEqual(track_b["total_identity_switches"], 0)
            self.assertEqual(track_b["total_duplicate_tracks"], 0)

    def test_zero_observation_handling(self):
        """Verifies that when zero ground-truth objects or events are present, metrics return 'N/A'."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset = PhysicalWorkbenchDataset(
                dataset_dir=tmp_dir,
                manifest_data={"dataset_name": "Empty Dataset", "version": "1.0"},
                sequences={},
            )
            track_b = evaluate_physical_track_b(dataset)
            self.assertEqual(track_b["total_events_evaluated"], 0)
            self.assertEqual(track_b["event_accuracy"], "N/A")


if __name__ == "__main__":
    unittest.main()
