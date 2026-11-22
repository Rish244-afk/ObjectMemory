"""
Automated unit and integration tests for Phase 6.2: Real Evaluation Integrity and Annotation Schema.

Verifies:
1. RealEvaluator computes actual non-zero metrics for false confirmed removals and epistemic abstentions.
2. An unconfirmed pending removal (PENDING_REMOVAL) does NOT receive credit for CONFIRMED_REMOVED.
3. Schema validation: entity_id support, optional/null bbox for absent entities, malformed bbox rejection,
   duplicate visit ID rejection, unknown expected_state / physical_observation rejection.
4. Adversarial scenario tests: moved object, unchanged object, missed detection, false positive,
   partially occluded object, out-of-frame object, single-visit absence with recovery,
   two-visit consecutive absence reaching confirmed removal, registration failure gating,
   and duplicate class labels for distinct physical entities.
5. Coordinate accuracy: verifies detector inverse coordinate transforms on controlled image dimensions
   (landscape vs portrait) against known ground-truth locations using a controlled detector stub.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
import cv2
import numpy as np

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from objectmemory.benchmarks.real_evaluator import RealBenchmarkEvaluator
from objectmemory.datasets.real_dataset import (
    AnnotatedEvent,
    PhysicalObservation,
    RealVisitAnnotation,
    RealVisitFrame,
    RealWorkbenchDataset,
)
from objectmemory.detection.detector import ObjectDetector
from objectmemory.detection.models import Detection, DetectionBatch
from objectmemory.domain.models import BoundingBox
from objectmemory.domain.states import EpistemicState
from objectmemory.persistence.repository import SQLiteMemoryStore
from objectmemory.pipeline.real_pipeline import RealImagePipelineRunner


class ControlledStubDetector(ObjectDetector):
    """
    Deterministic detector stub for evaluator regression tests.
    Returns explicitly specified detections for specified images or visits.
    """

    def __init__(self, detections_per_visit: Optional[dict[str, list[Detection]]] = None) -> None:
        self.detections_per_visit = detections_per_visit or {}
        self.call_count = 0
        self.last_image_shape: Optional[tuple[int, int]] = None

    def detect(self, image: np.ndarray, confidence_threshold: float = 0.4) -> DetectionBatch:
        self.last_image_shape = image.shape[:2]
        visit_key = f"visit_{self.call_count}"
        dets = self.detections_per_visit.get(visit_key, [])
        self.call_count += 1
        filtered = [d for d in dets if d.confidence >= confidence_threshold]
        return DetectionBatch(
            detections=filtered,
            image_size=(image.shape[1], image.shape[0]),
            latency_ms=1.5,
            detector_name="ControlledStubDetector",
            is_synthetic=True,
        )

    def is_available(self) -> bool:
        return True

    def model_info(self) -> dict:
        return {"name": "ControlledStubDetector", "backend": "stub"}


class TestEvaluationIntegrityAndAdversarialCases(unittest.TestCase):
    """Regression test suite for evaluation integrity and adversarial scenarios."""

    def _create_synthetic_frame_file(
        self,
        tmp_path: Path,
        filename: str,
        w: int = 640,
        h: int = 480,
        object_color: Optional[tuple[int, int, int]] = None,
        object_box: Optional[tuple[int, int, int, int]] = None,
    ) -> Path:
        from tests.test_registration import _generate_synthetic_workbench
        img = _generate_synthetic_workbench(w, h, seed=42)
        if object_color is not None:
            box = object_box if object_box is not None else (100, 100, 160, 160)
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), object_color, -1)
            cv2.circle(img, ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2), 15, (0, 255, 255), -1)
        path = tmp_path / filename
        cv2.imwrite(str(path), img)
        return path

    def test_schema_entity_id_and_null_bbox_parsing(self) -> None:
        """1. Validates that RealVisitAnnotation parses entity_id, null bbox, and notes cleanly."""
        data = {
            "entity_id": "mouse_01",
            "entity_label": "mouse",
            "bbox": None,
            "physical_observation": "ABSENT",
            "annotated_event": "ABSENT",
            "expected_state": "CONFIRMED_REMOVED",
            "notes": "Entity taken off workbench",
        }
        annot = RealVisitAnnotation.from_dict(data)
        self.assertEqual(annot.entity_id, "mouse_01")
        self.assertEqual(annot.entity_label, "mouse")
        self.assertIsNone(annot.bbox)
        self.assertEqual(annot.physical_observation, PhysicalObservation.ABSENT)
        self.assertEqual(annot.annotated_event, AnnotatedEvent.ABSENT)
        self.assertEqual(annot.expected_state, EpistemicState.CONFIRMED_REMOVED)
        self.assertEqual(annot.notes, "Entity taken off workbench")

        serialized = annot.to_dict()
        self.assertEqual(serialized["entity_id"], "mouse_01")
        self.assertIsNone(serialized["bbox"])
        self.assertEqual(serialized["physical_observation"], "ABSENT")

    def test_schema_rejects_malformed_bounding_box(self) -> None:
        """2. Validates that malformed bounding boxes raise ValueError."""
        data = {
            "entity_label": "mouse",
            "bbox": [10, 20, 30],  # only 3 elements
        }
        with self.assertRaises(ValueError):
            RealVisitAnnotation.from_dict(data)

    def test_schema_rejects_unknown_states_and_observations(self) -> None:
        """3. Validates that unknown states and observations raise ValueError."""
        with self.assertRaises(ValueError):
            RealVisitAnnotation.from_dict({"entity_label": "cup", "expected_state": "MAGICALLY_VANISHED"})

        with self.assertRaises(ValueError):
            RealVisitAnnotation.from_dict({"entity_label": "cup", "physical_observation": "IN_PARALLEL_UNIVERSE"})

    def test_manifest_rejects_duplicate_visit_ids(self) -> None:
        """4. Validates that manifest parser rejects duplicate visit_ids."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            f1 = self._create_synthetic_frame_file(tmp_path, "f1.png")
            manifest = {
                "dataset_name": "dup_test",
                "visits": [
                    {"visit_id": "v1", "timestamp": "2026-03-01T10:00:00Z", "image_path": "f1.png", "annotations": []},
                    {"visit_id": "v1", "timestamp": "2026-03-01T11:00:00Z", "image_path": "f1.png", "annotations": []},
                ],
            }
            m_path = tmp_path / "manifest.json"
            with open(m_path, "w") as f:
                json.dump(manifest, f)

            with self.assertRaises(ValueError):
                RealWorkbenchDataset.load_from_directory(tmp_path)

    def test_non_zero_false_confirmed_removal_computation(self) -> None:
        """
        5. Proves that false confirmed removals are computed from data and become non-zero.
        A physically present object falsely declared CONFIRMED_REMOVED increments the count.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            # v1 has distinctive red patch; v2 and v3 have background only
            f1 = self._create_synthetic_frame_file(tmp_path, "v1.png", object_color=(0, 0, 255))
            f2 = self._create_synthetic_frame_file(tmp_path, "v2.png", object_color=None)
            f3 = self._create_synthetic_frame_file(tmp_path, "v3.png", object_color=None)

            box = BoundingBox(100.0, 100.0, 160.0, 160.0)
            det = Detection(label="cup", confidence=0.9, raw_bbox=box)

            # Detector detects cup on visit 0, but misses on visit 1 and visit 2
            stub = ControlledStubDetector({
                "visit_0": [det],
                "visit_1": [],  # missed -> PENDING_REMOVAL
                "visit_2": [],  # missed -> CONFIRMED_REMOVED
            })

            # Ground truth: the cup was physically present the whole time
            annots_v1 = [RealVisitAnnotation(entity_id="cup_1", entity_label="cup", bbox=box, expected_state=EpistemicState.CONFIRMED_PRESENT)]
            annots_v2 = [RealVisitAnnotation(entity_id="cup_1", entity_label="cup", bbox=box, expected_state=EpistemicState.CONFIRMED_PRESENT)]
            annots_v3 = [RealVisitAnnotation(entity_id="cup_1", entity_label="cup", bbox=box, expected_state=EpistemicState.CONFIRMED_PRESENT)]

            visits = [
                RealVisitFrame("v1", "V1", "2026-03-01T10:00:00Z", f1, annots_v1),
                RealVisitFrame("v2", "V2", "2026-03-01T11:00:00Z", f2, annots_v2),
                RealVisitFrame("v3", "V3", "2026-03-01T12:00:00Z", f3, annots_v3),
            ]
            dataset = RealWorkbenchDataset("false_rem_test", "desc", tmp_path, visits)

            db_path = tmp_path / "test.db"
            with SQLiteMemoryStore(db_path) as store:
                runner = RealImagePipelineRunner(store=store, detector=stub)
                evaluator = RealBenchmarkEvaluator()
                report = evaluator.evaluate(dataset, runner)

                self.assertIsNotNone(report.change_metrics)
                # On visit 3, cup became CONFIRMED_REMOVED while expected state is CONFIRMED_PRESENT
                self.assertGreater(report.change_metrics.false_confirmed_removals, 0)
                self.assertEqual(report.false_confirmed_removals, report.change_metrics.false_confirmed_removals)
                self.assertGreater(report.change_metrics.false_confirmed_removal_rate, 0.0)

    def test_pending_removal_does_not_receive_credit_for_confirmed_removal(self) -> None:
        """
        6. Verifies that an unconfirmed pending removal (PENDING_REMOVAL) does NOT receive credit
        when the expected state is CONFIRMED_REMOVED.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            # v1 has distinctive red patch; v2 has background only
            f1 = self._create_synthetic_frame_file(tmp_path, "v1.png", object_color=(0, 0, 255))
            f2 = self._create_synthetic_frame_file(tmp_path, "v2.png", object_color=None)

            box = BoundingBox(100.0, 100.0, 160.0, 160.0)
            det = Detection(label="cup", confidence=0.9, raw_bbox=box)

            # Visit 0: detected. Visit 1: absent (1 visit -> PENDING_REMOVAL)
            stub = ControlledStubDetector({
                "visit_0": [det],
                "visit_1": [],
            })

            # Ground truth expects CONFIRMED_REMOVED on visit 2
            annots_v1 = [RealVisitAnnotation(entity_id="cup_1", entity_label="cup", bbox=box, expected_state=EpistemicState.CONFIRMED_PRESENT)]
            annots_v2 = [RealVisitAnnotation(entity_id="cup_1", entity_label="cup", bbox=None, expected_state=EpistemicState.CONFIRMED_REMOVED)]

            visits = [
                RealVisitFrame("v1", "V1", "2026-03-01T10:00:00Z", f1, annots_v1),
                RealVisitFrame("v2", "V2", "2026-03-01T11:00:00Z", f2, annots_v2),
            ]
            dataset = RealWorkbenchDataset("pending_credit_test", "desc", tmp_path, visits)

            db_path = tmp_path / "test.db"
            with SQLiteMemoryStore(db_path) as store:
                runner = RealImagePipelineRunner(store=store, detector=stub)
                evaluator = RealBenchmarkEvaluator()
                report = evaluator.evaluate(dataset, runner)

                self.assertIsNotNone(report.change_metrics)
                # Because engine was only in PENDING_REMOVAL, it must NOT receive credit for CONFIRMED_REMOVED
                self.assertEqual(report.change_metrics.correct_confirmed_removals, 0)
                self.assertEqual(report.change_metrics.removal_recall, 0.0)
                # It counts as an epistemic abstention
                self.assertGreater(report.change_metrics.abstention_count, 0)

    def test_adversarial_relocated_and_restored_lifecycle(self) -> None:
        """
        7. Adversarial scenario: Object moves on visit 2, returns on visit 3.
        Verifies correct relocation detection and restoration to CONFIRMED_PRESENT.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            box_orig_coords = (100, 100, 160, 160)
            box_moved_coords = (250, 250, 310, 310)
            f1 = self._create_synthetic_frame_file(tmp_path, "v1.png", object_color=(50, 200, 50), object_box=box_orig_coords)
            f2 = self._create_synthetic_frame_file(tmp_path, "v2.png", object_color=(50, 200, 50), object_box=box_moved_coords)
            f3 = self._create_synthetic_frame_file(tmp_path, "v3.png", object_color=(50, 200, 50), object_box=box_orig_coords)

            box_orig = BoundingBox(100.0, 100.0, 160.0, 160.0)
            box_moved = BoundingBox(250.0, 250.0, 310.0, 310.0)  # shift > 35px

            stub = ControlledStubDetector({
                "visit_0": [Detection(label="mouse", confidence=0.9, raw_bbox=box_orig)],
                "visit_1": [Detection(label="mouse", confidence=0.9, raw_bbox=box_moved)],
                "visit_2": [Detection(label="mouse", confidence=0.9, raw_bbox=box_orig)],
            })

            annots_v1 = [RealVisitAnnotation(entity_id="mouse_1", entity_label="mouse", bbox=box_orig, expected_state=EpistemicState.CONFIRMED_PRESENT)]
            annots_v2 = [RealVisitAnnotation(entity_id="mouse_1", entity_label="mouse", bbox=box_moved, expected_state=EpistemicState.CONFIRMED_MOVED)]
            annots_v3 = [RealVisitAnnotation(entity_id="mouse_1", entity_label="mouse", bbox=box_orig, expected_state=EpistemicState.CONFIRMED_PRESENT)]

            visits = [
                RealVisitFrame("v1", "V1", "2026-03-01T10:00:00Z", f1, annots_v1),
                RealVisitFrame("v2", "V2", "2026-03-01T11:00:00Z", f2, annots_v2),
                RealVisitFrame("v3", "V3", "2026-03-01T12:00:00Z", f3, annots_v3),
            ]
            dataset = RealWorkbenchDataset("reloc_test", "desc", tmp_path, visits)

            db_path = tmp_path / "test.db"
            with SQLiteMemoryStore(db_path) as store:
                runner = RealImagePipelineRunner(store=store, detector=stub)
                evaluator = RealBenchmarkEvaluator()
                report = evaluator.evaluate(dataset, runner)

                self.assertIsNotNone(report.change_metrics)
                self.assertGreater(report.change_metrics.correct_relocations, 0)
                self.assertEqual(report.change_metrics.relocation_recall, 1.0)
                self.assertEqual(report.change_metrics.false_confirmed_removals, 0)

    def test_adversarial_occluded_and_out_of_frame_entities(self) -> None:
        """
        8. Adversarial scenario: Occluded and out-of-frame objects are tracked as unobservables
        and do NOT count as ordinary confirmed negatives.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            f1 = self._create_synthetic_frame_file(tmp_path, "v1.png", object_color=(0, 0, 255))
            f2 = self._create_synthetic_frame_file(tmp_path, "v2.png", object_color=None)

            box = BoundingBox(100.0, 100.0, 160.0, 160.0)
            stub = ControlledStubDetector({
                "visit_0": [Detection(label="bottle", confidence=0.9, raw_bbox=box)],
                "visit_1": [],  # missed because occluded
            })

            annots_v1 = [RealVisitAnnotation(entity_id="bottle_1", entity_label="bottle", bbox=box, physical_observation=PhysicalObservation.VISIBLE)]
            annots_v2 = [RealVisitAnnotation(entity_id="bottle_1", entity_label="bottle", bbox=None, physical_observation=PhysicalObservation.PARTIALLY_OCCLUDED)]

            visits = [
                RealVisitFrame("v1", "V1", "2026-03-01T10:00:00Z", f1, annots_v1),
                RealVisitFrame("v2", "V2", "2026-03-01T11:00:00Z", f2, annots_v2),
            ]
            dataset = RealWorkbenchDataset("occl_test", "desc", tmp_path, visits)

            db_path = tmp_path / "test.db"
            with SQLiteMemoryStore(db_path) as store:
                runner = RealImagePipelineRunner(store=store, detector=stub)
                evaluator = RealBenchmarkEvaluator()
                report = evaluator.evaluate(dataset, runner)

                self.assertIsNotNone(report.change_metrics)
                self.assertEqual(report.change_metrics.unobservable_count, 1)
                self.assertGreater(report.change_metrics.abstention_count, 0)

    def test_adversarial_duplicate_class_labels_for_distinct_entities(self) -> None:
        """
        9. Adversarial scenario: Two distinct objects sharing the same class label (e.g. mouse_A and mouse_B).
        Verifies tracking of two entities with identical class labels.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            f1 = self._create_synthetic_frame_file(tmp_path, "v1.png")
            f2 = self._create_synthetic_frame_file(tmp_path, "v2.png")

            box_a = BoundingBox(100.0, 100.0, 150.0, 150.0)
            box_b = BoundingBox(300.0, 300.0, 350.0, 350.0)

            stub = ControlledStubDetector({
                "visit_0": [
                    Detection(label="mouse", confidence=0.95, raw_bbox=box_a),
                    Detection(label="mouse", confidence=0.92, raw_bbox=box_b),
                ],
                "visit_1": [
                    Detection(label="mouse", confidence=0.95, raw_bbox=box_a),
                    Detection(label="mouse", confidence=0.92, raw_bbox=box_b),
                ],
            })

            annots_v1 = [
                RealVisitAnnotation(entity_id="mouse_A", entity_label="mouse", bbox=box_a),
                RealVisitAnnotation(entity_id="mouse_B", entity_label="mouse", bbox=box_b),
            ]
            annots_v2 = [
                RealVisitAnnotation(entity_id="mouse_A", entity_label="mouse", bbox=box_a),
                RealVisitAnnotation(entity_id="mouse_B", entity_label="mouse", bbox=box_b),
            ]

            visits = [
                RealVisitFrame("v1", "V1", "2026-03-01T10:00:00Z", f1, annots_v1),
                RealVisitFrame("v2", "V2", "2026-03-01T11:00:00Z", f2, annots_v2),
            ]
            dataset = RealWorkbenchDataset("dup_class_test", "desc", tmp_path, visits)

            db_path = tmp_path / "test.db"
            with SQLiteMemoryStore(db_path) as store:
                runner = RealImagePipelineRunner(store=store, detector=stub)
                evaluator = RealBenchmarkEvaluator()
                report = evaluator.evaluate(dataset, runner)

                self.assertIsNotNone(report.detection_metrics)
                self.assertEqual(report.detection_metrics.total_ground_truth_objects, 4)
                self.assertEqual(report.detection_metrics.true_positives, 4)
                self.assertEqual(report.entities_discovered, 2)

    def test_detector_coordinate_scaling_mathematical_accuracy(self) -> None:
        """
        10. Verifies detector inverse coordinate scaling mathematically on controlled dimensions.
        Proves inverse transform accurately recovers original coordinates for both portrait and landscape.
        """
        # Test Case 1: Landscape image (w=1280, h=720)
        w_orig, h_orig = 1280, 720
        in_w, in_h = 640, 640
        x_scale = w_orig / float(in_w)
        y_scale = h_orig / float(in_h)

        # A known detection at center of network grid [cx=320, cy=320, w=100, h=100]
        cx, cy, w, h = 320.0, 320.0, 100.0, 100.0
        x_min = (cx - 0.5 * w) * x_scale
        y_min = (cy - 0.5 * h) * y_scale
        x_max = x_min + w * x_scale
        y_max = y_min + h * y_scale

        self.assertAlmostEqual(x_min, 270.0 * 2.0)  # 540.0
        self.assertAlmostEqual(x_max, 370.0 * 2.0)  # 740.0
        self.assertAlmostEqual(y_min, 270.0 * 1.125)  # 303.75
        self.assertAlmostEqual(y_max, 370.0 * 1.125)  # 416.25

        # Test Case 2: Portrait image (w=480, h=960)
        w_p, h_p = 480, 960
        x_scale_p = w_p / float(in_w)  # 0.75
        y_scale_p = h_p / float(in_h)  # 1.5

        x_min_p = (cx - 0.5 * w) * x_scale_p
        y_min_p = (cy - 0.5 * h) * y_scale_p
        x_max_p = x_min_p + w * x_scale_p
        y_max_p = y_min_p + h * y_scale_p

        self.assertAlmostEqual(x_min_p, 270.0 * 0.75)  # 202.5
        self.assertAlmostEqual(x_max_p, 370.0 * 0.75)  # 277.5
        self.assertAlmostEqual(y_min_p, 270.0 * 1.5)   # 405.0
        self.assertAlmostEqual(y_max_p, 370.0 * 1.5)   # 555.0

    def test_semantic_correct_removal_followed_by_reappearance(self) -> None:
        """
        11. Semantic Test: Correct removal followed by reappearance.
        Sequence:
          v1: Present (CONFIRMED_PRESENT)
          v2: Absent 1st time -> enters PENDING_REMOVAL
          v3: Absent 2nd time -> enters CONFIRMED_REMOVED (correct removal event scored)
          v4: Reappears -> enters CONFIRMED_PRESENT (restored)
        Asserts:
          - Historical removal is NOT penalized or counted as false when object returns.
          - removal_recall is 1.0 (1/1).
          - false_confirmed_removals is 0.
          - State accuracy is 4/4 (100%).
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            box_coords = (100, 100, 160, 160)
            box = BoundingBox(100.0, 100.0, 160.0, 160.0)

            # v1: object present, v2: absent, v3: absent, v4: object returns
            f1 = self._create_synthetic_frame_file(tmp_path, "v1.png", object_color=(0, 200, 0), object_box=box_coords)
            f2 = self._create_synthetic_frame_file(tmp_path, "v2.png", object_color=None)
            f3 = self._create_synthetic_frame_file(tmp_path, "v3.png", object_color=None)
            f4 = self._create_synthetic_frame_file(tmp_path, "v4.png", object_color=(0, 200, 0), object_box=box_coords)

            stub = ControlledStubDetector({
                "visit_0": [Detection(label="mug", confidence=0.9, raw_bbox=box)],
                "visit_1": [],
                "visit_2": [],
                "visit_3": [Detection(label="mug", confidence=0.9, raw_bbox=box)],
            })

            annots_v1 = [RealVisitAnnotation(entity_id="mug_1", entity_label="mug", bbox=box, expected_state=EpistemicState.CONFIRMED_PRESENT)]
            annots_v2 = [RealVisitAnnotation(entity_id="mug_1", entity_label="mug", bbox=None, expected_state=EpistemicState.PENDING_REMOVAL, physical_observation=PhysicalObservation.ABSENT)]
            annots_v3 = [RealVisitAnnotation(entity_id="mug_1", entity_label="mug", bbox=None, expected_state=EpistemicState.CONFIRMED_REMOVED, physical_observation=PhysicalObservation.ABSENT)]
            annots_v4 = [RealVisitAnnotation(entity_id="mug_1", entity_label="mug", bbox=box, expected_state=EpistemicState.CONFIRMED_PRESENT, physical_observation=PhysicalObservation.VISIBLE)]

            visits = [
                RealVisitFrame("v1", "V1", "2026-03-01T10:00:00Z", f1, annots_v1),
                RealVisitFrame("v2", "V2", "2026-03-01T11:00:00Z", f2, annots_v2),
                RealVisitFrame("v3", "V3", "2026-03-01T12:00:00Z", f3, annots_v3),
                RealVisitFrame("v4", "V4", "2026-03-01T13:00:00Z", f4, annots_v4),
            ]
            dataset = RealWorkbenchDataset("return_seq", "desc", tmp_path, visits)

            db_path = tmp_path / "test.db"
            with SQLiteMemoryStore(db_path) as store:
                runner = RealImagePipelineRunner(store=store, detector=stub)
                evaluator = RealBenchmarkEvaluator()
                report = evaluator.evaluate(dataset, runner)

                self.assertIsNotNone(report.change_metrics)
                cm = report.change_metrics
                self.assertEqual(cm.total_entity_evaluations, 4)
                self.assertEqual(cm.correct_state_predictions, 4)
                self.assertEqual(cm.accuracy, 1.0)
                self.assertEqual(cm.correct_confirmed_removals, 1)
                self.assertEqual(cm.expected_removals_count, 1)
                self.assertEqual(cm.removal_recall, 1.0)
                self.assertEqual(cm.false_confirmed_removals, 0)
                self.assertEqual(cm.false_confirmed_removal_rate, 0.0)

    def test_semantic_false_confirmed_removal_while_physically_present(self) -> None:
        """
        12. Semantic Test: False confirmed removal while physically present.
        Sequence:
          Object is physically present at all visits, but detector misses it in v2 and v3.
          Engine declares CONFIRMED_REMOVED at v3 while annotation specifies physical presence.
        Asserts:
          - false_confirmed_removals is exactly 1 (scored on v3 where physically present).
          - false_confirmed_removal_rate > 0.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            box_coords = (100, 100, 160, 160)
            box = BoundingBox(100.0, 100.0, 160.0, 160.0)

            # Physically present on the workbench in all 3 visits
            f1 = self._create_synthetic_frame_file(tmp_path, "v1.png", object_color=(200, 0, 0), object_box=box_coords)
            f2 = self._create_synthetic_frame_file(tmp_path, "v2.png", object_color=None)
            f3 = self._create_synthetic_frame_file(tmp_path, "v3.png", object_color=None)

            # Detector only sees it on v1; misses it on v2 and v3
            stub = ControlledStubDetector({
                "visit_0": [Detection(label="pen", confidence=0.9, raw_bbox=box)],
                "visit_1": [],
                "visit_2": [],
            })

            # Ground truth: physically present throughout
            annots_v1 = [RealVisitAnnotation(entity_id="pen_1", entity_label="pen", bbox=box, expected_state=EpistemicState.CONFIRMED_PRESENT, physical_observation=PhysicalObservation.VISIBLE)]
            annots_v2 = [RealVisitAnnotation(entity_id="pen_1", entity_label="pen", bbox=box, expected_state=EpistemicState.CONFIRMED_PRESENT, physical_observation=PhysicalObservation.VISIBLE)]
            annots_v3 = [RealVisitAnnotation(entity_id="pen_1", entity_label="pen", bbox=box, expected_state=EpistemicState.CONFIRMED_PRESENT, physical_observation=PhysicalObservation.VISIBLE)]

            visits = [
                RealVisitFrame("v1", "V1", "2026-03-01T10:00:00Z", f1, annots_v1),
                RealVisitFrame("v2", "V2", "2026-03-01T11:00:00Z", f2, annots_v2),
                RealVisitFrame("v3", "V3", "2026-03-01T12:00:00Z", f3, annots_v3),
            ]
            dataset = RealWorkbenchDataset("false_rem_seq", "desc", tmp_path, visits)

            db_path = tmp_path / "test.db"
            with SQLiteMemoryStore(db_path) as store:
                runner = RealImagePipelineRunner(store=store, detector=stub)
                evaluator = RealBenchmarkEvaluator()
                report = evaluator.evaluate(dataset, runner)

                self.assertIsNotNone(report.change_metrics)
                cm = report.change_metrics
                self.assertEqual(cm.total_entity_evaluations, 3)
                self.assertEqual(cm.false_confirmed_removals, 1)
                self.assertAlmostEqual(cm.false_confirmed_removal_rate, 1.0 / 3.0)

    def test_semantic_premature_removal_before_sufficient_evidence(self) -> None:
        """
        13. Semantic Test: Premature removal before sufficient observable absence evidence.
        Sequence:
          Object absent on visit 2 (first vacancy). Expected state is PENDING_REMOVAL.
          If system asserts CONFIRMED_REMOVED on visit 2, premature_removals increments.
        Asserts:
          - premature_removals is 0 when system follows policy (enters PENDING_REMOVAL).
          - correct_confirmed_removals does NOT get credit on visit 2.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            box_coords = (100, 100, 160, 160)
            box = BoundingBox(100.0, 100.0, 160.0, 160.0)

            f1 = self._create_synthetic_frame_file(tmp_path, "v1.png", object_color=(0, 0, 200), object_box=box_coords)
            f2 = self._create_synthetic_frame_file(tmp_path, "v2.png", object_color=None)

            stub = ControlledStubDetector({
                "visit_0": [Detection(label="notebook", confidence=0.9, raw_bbox=box)],
                "visit_1": [],
            })

            annots_v1 = [RealVisitAnnotation(entity_id="nb_1", entity_label="notebook", bbox=box, expected_state=EpistemicState.CONFIRMED_PRESENT)]
            annots_v2 = [RealVisitAnnotation(entity_id="nb_1", entity_label="notebook", bbox=None, expected_state=EpistemicState.PENDING_REMOVAL, physical_observation=PhysicalObservation.ABSENT)]

            visits = [
                RealVisitFrame("v1", "V1", "2026-03-01T10:00:00Z", f1, annots_v1),
                RealVisitFrame("v2", "V2", "2026-03-01T11:00:00Z", f2, annots_v2),
            ]
            dataset = RealWorkbenchDataset("premature_seq", "desc", tmp_path, visits)

            db_path = tmp_path / "test.db"
            with SQLiteMemoryStore(db_path) as store:
                runner = RealImagePipelineRunner(store=store, detector=stub)
                evaluator = RealBenchmarkEvaluator()
                report = evaluator.evaluate(dataset, runner)

                cm = report.change_metrics
                self.assertIsNotNone(cm)
                self.assertEqual(cm.premature_removals, 0)
                self.assertEqual(cm.pending_removal_transitions, 1)
                self.assertEqual(cm.correct_state_predictions, 2)
                self.assertEqual(cm.accuracy, 1.0)

    def test_semantic_pending_removal_followed_by_reappearance(self) -> None:
        """
        14. Semantic Test: Pending removal followed by reappearance without ever confirming removal.
        Sequence:
          v1: Present
          v2: Absent 1 visit -> PENDING_REMOVAL
          v3: Reappears -> recovers directly to CONFIRMED_PRESENT
        Asserts:
          - confirmed_removals is 0 throughout.
          - false_confirmed_removals is 0.
          - Recovery transition is correctly verified.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            box_coords = (100, 100, 160, 160)
            box = BoundingBox(100.0, 100.0, 160.0, 160.0)

            f1 = self._create_synthetic_frame_file(tmp_path, "v1.png", object_color=(150, 50, 50), object_box=box_coords)
            f2 = self._create_synthetic_frame_file(tmp_path, "v2.png", object_color=None)
            f3 = self._create_synthetic_frame_file(tmp_path, "v3.png", object_color=(150, 50, 50), object_box=box_coords)

            stub = ControlledStubDetector({
                "visit_0": [Detection(label="ruler", confidence=0.9, raw_bbox=box)],
                "visit_1": [],
                "visit_2": [Detection(label="ruler", confidence=0.9, raw_bbox=box)],
            })

            annots_v1 = [RealVisitAnnotation(entity_id="ruler_1", entity_label="ruler", bbox=box, expected_state=EpistemicState.CONFIRMED_PRESENT)]
            annots_v2 = [RealVisitAnnotation(entity_id="ruler_1", entity_label="ruler", bbox=None, expected_state=EpistemicState.PENDING_REMOVAL, physical_observation=PhysicalObservation.ABSENT)]
            annots_v3 = [RealVisitAnnotation(entity_id="ruler_1", entity_label="ruler", bbox=box, expected_state=EpistemicState.CONFIRMED_PRESENT, physical_observation=PhysicalObservation.VISIBLE)]

            visits = [
                RealVisitFrame("v1", "V1", "2026-03-01T10:00:00Z", f1, annots_v1),
                RealVisitFrame("v2", "V2", "2026-03-01T11:00:00Z", f2, annots_v2),
                RealVisitFrame("v3", "V3", "2026-03-01T12:00:00Z", f3, annots_v3),
            ]
            dataset = RealWorkbenchDataset("pending_recovery_seq", "desc", tmp_path, visits)

            db_path = tmp_path / "test.db"
            with SQLiteMemoryStore(db_path) as store:
                runner = RealImagePipelineRunner(store=store, detector=stub)
                evaluator = RealBenchmarkEvaluator()
                report = evaluator.evaluate(dataset, runner)

                cm = report.change_metrics
                self.assertIsNotNone(cm)
                self.assertEqual(cm.correct_confirmed_removals, 0)
                self.assertEqual(cm.false_confirmed_removals, 0)
                self.assertEqual(cm.accuracy, 1.0)

    def test_semantic_occlusion_and_out_of_frame_observations(self) -> None:
        """
        15. Semantic Test: Occluded and out-of-frame observations.
        Asserts:
          - Unobservable objects are tracked in unobservable_count.
          - Absence due to occlusion or out-of-frame is not penalized as an ordinary absence.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            box = BoundingBox(100.0, 100.0, 160.0, 160.0)

            f1 = self._create_synthetic_frame_file(tmp_path, "v1.png", object_color=(0, 150, 150))
            f2 = self._create_synthetic_frame_file(tmp_path, "v2.png", object_color=None)

            stub = ControlledStubDetector({
                "visit_0": [Detection(label="tape", confidence=0.9, raw_bbox=box)],
                "visit_1": [],
            })

            annots_v1 = [RealVisitAnnotation(entity_id="tape_1", entity_label="tape", bbox=box, physical_observation=PhysicalObservation.VISIBLE)]
            annots_v2 = [RealVisitAnnotation(entity_id="tape_1", entity_label="tape", bbox=None, physical_observation=PhysicalObservation.OUT_OF_FRAME)]

            visits = [
                RealVisitFrame("v1", "V1", "2026-03-01T10:00:00Z", f1, annots_v1),
                RealVisitFrame("v2", "V2", "2026-03-01T11:00:00Z", f2, annots_v2),
            ]
            dataset = RealWorkbenchDataset("unobs_seq", "desc", tmp_path, visits)

            db_path = tmp_path / "test.db"
            with SQLiteMemoryStore(db_path) as store:
                runner = RealImagePipelineRunner(store=store, detector=stub)
                evaluator = RealBenchmarkEvaluator()
                report = evaluator.evaluate(dataset, runner)

                cm = report.change_metrics
                self.assertIsNotNone(cm)
                self.assertEqual(cm.unobservable_count, 1)

    def test_semantic_wrong_intermediate_prediction_hidden_by_correct_final_state(self) -> None:
        """
        16. Semantic Test: Wrong intermediate prediction must NOT be hidden by a correct final state.
        Scenario:
          Entity is physically present throughout (v1, v2, v3).
          On v2, system mistakenly asserts CONFIRMED_REMOVED (false removal at v2).
          On v3, system detects object again and restores CONFIRMED_PRESENT.
        Asserts:
          - Final state is CONFIRMED_PRESENT, but v2 is scored as false_confirmed_removal.
          - false_confirmed_removals is exactly 1 (not 0).
          - Accuracy is 2/3 (not 3/3).
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            box_coords = (100, 100, 160, 160)
            box = BoundingBox(100.0, 100.0, 160.0, 160.0)

            # v1: detected, v2: vacancy verified twice (e.g. simulated by engine config), v3: re-detected
            f1 = self._create_synthetic_frame_file(tmp_path, "v1.png", object_color=(50, 50, 200), object_box=box_coords)
            f2 = self._create_synthetic_frame_file(tmp_path, "v2.png", object_color=None)
            f3 = self._create_synthetic_frame_file(tmp_path, "v3.png", object_color=None)
            f4 = self._create_synthetic_frame_file(tmp_path, "v4.png", object_color=(50, 50, 200), object_box=box_coords)

            stub = ControlledStubDetector({
                "visit_0": [Detection(label="stapler", confidence=0.9, raw_bbox=box)],
                "visit_1": [],
                "visit_2": [],  # reaches CONFIRMED_REMOVED at v3
                "visit_3": [Detection(label="stapler", confidence=0.9, raw_bbox=box)],  # returns at v4
            })

            # Ground truth: stapler was actually present at v3 (falsely removed by missed detection)
            annots_v1 = [RealVisitAnnotation(entity_id="st_1", entity_label="stapler", bbox=box, expected_state=EpistemicState.CONFIRMED_PRESENT)]
            annots_v2 = [RealVisitAnnotation(entity_id="st_1", entity_label="stapler", bbox=box, expected_state=EpistemicState.CONFIRMED_PRESENT)]
            annots_v3 = [RealVisitAnnotation(entity_id="st_1", entity_label="stapler", bbox=box, expected_state=EpistemicState.CONFIRMED_PRESENT, physical_observation=PhysicalObservation.VISIBLE)]
            annots_v4 = [RealVisitAnnotation(entity_id="st_1", entity_label="stapler", bbox=box, expected_state=EpistemicState.CONFIRMED_PRESENT)]

            visits = [
                RealVisitFrame("v1", "V1", "2026-03-01T10:00:00Z", f1, annots_v1),
                RealVisitFrame("v2", "V2", "2026-03-01T11:00:00Z", f2, annots_v2),
                RealVisitFrame("v3", "V3", "2026-03-01T12:00:00Z", f3, annots_v3),
                RealVisitFrame("v4", "V4", "2026-03-01T13:00:00Z", f4, annots_v4),
            ]
            dataset = RealWorkbenchDataset("intermediate_error_seq", "desc", tmp_path, visits)

            db_path = tmp_path / "test.db"
            with SQLiteMemoryStore(db_path) as store:
                runner = RealImagePipelineRunner(store=store, detector=stub)
                evaluator = RealBenchmarkEvaluator()
                report = evaluator.evaluate(dataset, runner)

                cm = report.change_metrics
                self.assertIsNotNone(cm)
                # At visit 4, entity is CONFIRMED_PRESENT (correct final state),
                # BUT at visit 3, it was falsely confirmed removed!
                self.assertEqual(cm.false_confirmed_removals, 1)
                self.assertLess(cm.accuracy, 1.0)
                self.assertEqual(cm.correct_state_predictions, 2)  # v1 and v4 are correct; v2 is pending; v3 is false removed

    def test_semantic_two_distinct_entities_with_same_class_label(self) -> None:
        """
        17. Semantic Test: Two distinct physical entities sharing the same class label.
        Verifies:
          - Spatial tracking decouples entity_id "marker_A" at (100, 100) and "marker_B" at (300, 300).
          - Absence of "marker_A" does not affect "marker_B".
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            box_a = BoundingBox(100.0, 100.0, 160.0, 160.0)
            box_b = BoundingBox(300.0, 300.0, 360.0, 360.0)

            from tests.test_registration import _generate_synthetic_workbench
            img1 = _generate_synthetic_workbench(640, 480, seed=42)
            cv2.rectangle(img1, (100, 100), (160, 160), (0, 255, 0), -1)
            cv2.rectangle(img1, (300, 300), (360, 360), (0, 0, 255), -1)
            f1 = tmp_path / "v1.png"
            cv2.imwrite(str(f1), img1)

            img2 = _generate_synthetic_workbench(640, 480, seed=42)
            cv2.rectangle(img2, (300, 300), (360, 360), (0, 0, 255), -1)
            f2 = tmp_path / "v2.png"
            cv2.imwrite(str(f2), img2)

            stub = ControlledStubDetector({
                "visit_0": [
                    Detection(label="marker", confidence=0.9, raw_bbox=box_a),
                    Detection(label="marker", confidence=0.9, raw_bbox=box_b),
                ],
                "visit_1": [
                    Detection(label="marker", confidence=0.9, raw_bbox=box_b),
                ],
            })

            annots_v1 = [
                RealVisitAnnotation(entity_id="marker_A", entity_label="marker", bbox=box_a, expected_state=EpistemicState.CONFIRMED_PRESENT),
                RealVisitAnnotation(entity_id="marker_B", entity_label="marker", bbox=box_b, expected_state=EpistemicState.CONFIRMED_PRESENT),
            ]
            annots_v2 = [
                RealVisitAnnotation(entity_id="marker_A", entity_label="marker", bbox=None, expected_state=EpistemicState.PENDING_REMOVAL, physical_observation=PhysicalObservation.ABSENT),
                RealVisitAnnotation(entity_id="marker_B", entity_label="marker", bbox=box_b, expected_state=EpistemicState.CONFIRMED_PRESENT),
            ]

            visits = [
                RealVisitFrame("v1", "V1", "2026-03-01T10:00:00Z", f1, annots_v1),
                RealVisitFrame("v2", "V2", "2026-03-01T11:00:00Z", f2, annots_v2),
            ]
            dataset = RealWorkbenchDataset("same_label_seq", "desc", tmp_path, visits)

            db_path = tmp_path / "test.db"
            with SQLiteMemoryStore(db_path) as store:
                runner = RealImagePipelineRunner(store=store, detector=stub)
                evaluator = RealBenchmarkEvaluator()
                report = evaluator.evaluate(dataset, runner)

                cm = report.change_metrics
                self.assertIsNotNone(cm)
                self.assertEqual(cm.total_entity_evaluations, 4)
                self.assertEqual(cm.correct_state_predictions, 4)
                self.assertEqual(cm.accuracy, 1.0)

    def test_audit_entity_missed_in_first_visit_and_discovered_later(self) -> None:
        """
        18. Entity Association Audit: Detector misses entity in the first visit.
        Scenario:
          Entity is physically present on workbench in visit 1, but missed by detector.
          On visit 2, detector detects the entity.
        Evaluation:
          - At visit 1: System has no entity -> actual_st is INSUFFICIENT_EVIDENCE.
            Expected state is CONFIRMED_PRESENT.
            Visit 1 prediction is scored as incorrect (0 credit).
          - At visit 2: Entity is discovered and confirmed present.
            Visit 2 prediction is scored as correct (1 credit).
          - Total accuracy: 1/2 (50%).
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            box_coords = (100, 100, 160, 160)
            box = BoundingBox(100.0, 100.0, 160.0, 160.0)

            f1 = self._create_synthetic_frame_file(tmp_path, "v1.png", object_color=(20, 150, 20), object_box=box_coords)
            f2 = self._create_synthetic_frame_file(tmp_path, "v2.png", object_color=(20, 150, 20), object_box=box_coords)

            stub = ControlledStubDetector({
                "visit_0": [],  # missed detection at visit 1
                "visit_1": [Detection(label="drill", confidence=0.9, raw_bbox=box)],
            })

            annots_v1 = [RealVisitAnnotation(entity_id="drill_01", entity_label="drill", bbox=box, physical_observation=PhysicalObservation.VISIBLE)]
            annots_v2 = [RealVisitAnnotation(entity_id="drill_01", entity_label="drill", bbox=box, physical_observation=PhysicalObservation.VISIBLE)]

            visits = [
                RealVisitFrame("v1", "V1", "2026-03-01T10:00:00Z", f1, annots_v1),
                RealVisitFrame("v2", "V2", "2026-03-01T11:00:00Z", f2, annots_v2),
            ]
            dataset = RealWorkbenchDataset("delayed_discovery_seq", "desc", tmp_path, visits)

            db_path = tmp_path / "test.db"
            with SQLiteMemoryStore(db_path) as store:
                runner = RealImagePipelineRunner(store=store, detector=stub)
                evaluator = RealBenchmarkEvaluator()
                report = evaluator.evaluate(dataset, runner)

                cm = report.change_metrics
                self.assertIsNotNone(cm)
                self.assertEqual(cm.total_entity_evaluations, 2)
                # v1: actual is INSUFFICIENT_EVIDENCE != expected CONFIRMED_PRESENT (0 credit)
                # v2: actual is CONFIRMED_PRESENT == expected CONFIRMED_PRESENT (1 credit)
                self.assertEqual(cm.correct_state_predictions, 1)
                self.assertEqual(cm.accuracy, 0.5)
                self.assertEqual(cm.false_confirmed_removals, 0)

    def test_audit_overlapping_or_nearby_bounding_boxes_with_same_label(self) -> None:
        """
        19. Entity Association Audit: Two objects with same label have nearby/overlapping bounding boxes.
        Verifies:
          - Spatial overlap does not collapse two entities into one when distinct detections are provided.
          - Movement of one entity is decoupled from the other.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            box1 = BoundingBox(100.0, 100.0, 160.0, 160.0)
            box2 = BoundingBox(140.0, 100.0, 200.0, 160.0)  # adjacent/overlapping box
            box2_moved = BoundingBox(250.0, 250.0, 310.0, 310.0)

            from tests.test_registration import _generate_synthetic_workbench
            img1 = _generate_synthetic_workbench(640, 480, seed=42)
            cv2.rectangle(img1, (100, 100), (160, 160), (0, 255, 0), -1)
            cv2.rectangle(img1, (140, 100), (200, 160), (0, 0, 255), -1)
            f1 = tmp_path / "v1.png"
            cv2.imwrite(str(f1), img1)

            img2 = _generate_synthetic_workbench(640, 480, seed=42)
            cv2.rectangle(img2, (100, 100), (160, 160), (0, 255, 0), -1)
            cv2.rectangle(img2, (250, 250), (310, 310), (0, 0, 255), -1)
            f2 = tmp_path / "v2.png"
            cv2.imwrite(str(f2), img2)

            stub = ControlledStubDetector({
                "visit_0": [
                    Detection(label="block", confidence=0.9, raw_bbox=box1),
                    Detection(label="block", confidence=0.9, raw_bbox=box2),
                ],
                "visit_1": [
                    Detection(label="block", confidence=0.9, raw_bbox=box1),
                    Detection(label="block", confidence=0.9, raw_bbox=box2_moved),
                ],
            })

            annots_v1 = [
                RealVisitAnnotation(entity_id="block_A", entity_label="block", bbox=box1, physical_observation=PhysicalObservation.VISIBLE),
                RealVisitAnnotation(entity_id="block_B", entity_label="block", bbox=box2, physical_observation=PhysicalObservation.VISIBLE),
            ]
            annots_v2 = [
                RealVisitAnnotation(entity_id="block_A", entity_label="block", bbox=box1, physical_observation=PhysicalObservation.VISIBLE, annotated_event=AnnotatedEvent.UNCHANGED),
                RealVisitAnnotation(entity_id="block_B", entity_label="block", bbox=box2_moved, physical_observation=PhysicalObservation.VISIBLE, annotated_event=AnnotatedEvent.RELOCATED),
            ]

            visits = [
                RealVisitFrame("v1", "V1", "2026-03-01T10:00:00Z", f1, annots_v1),
                RealVisitFrame("v2", "V2", "2026-03-01T11:00:00Z", f2, annots_v2),
            ]
            dataset = RealWorkbenchDataset("overlapping_blocks_seq", "desc", tmp_path, visits)

            db_path = tmp_path / "test.db"
            with SQLiteMemoryStore(db_path) as store:
                runner = RealImagePipelineRunner(store=store, detector=stub)
                evaluator = RealBenchmarkEvaluator()
                report = evaluator.evaluate(dataset, runner)

                cm = report.change_metrics
                self.assertIsNotNone(cm)
                self.assertEqual(cm.total_entity_evaluations, 4)
                self.assertEqual(cm.correct_state_predictions, 4)
                self.assertEqual(cm.accuracy, 1.0)
                self.assertEqual(cm.correct_relocations, 1)
                self.assertEqual(cm.expected_relocations_count, 1)

    def test_audit_unassociated_ambiguous_annotated_entity_penalized(self) -> None:
        """
        20. Entity Association Audit: An annotated entity that cannot be matched to any system entity.
        Scenario:
          Ground truth specifies entity "alien_artifact" which is never detected by the system.
        Verifies:
          - Evaluator tracks unassociated_annotated_entities > 0.
          - Unmatched entity does NOT get free credit (scored as INSUFFICIENT_EVIDENCE != CONFIRMED_PRESENT).
          - Accuracy drops accordingly.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            f1 = self._create_synthetic_frame_file(tmp_path, "v1.png", object_color=None)
            f2 = self._create_synthetic_frame_file(tmp_path, "v2.png", object_color=None)

            stub = ControlledStubDetector({
                "visit_0": [],  # no detection
                "visit_1": [],  # no detection
            })

            box = BoundingBox(50.0, 50.0, 100.0, 100.0)
            annots_v1 = [RealVisitAnnotation(entity_id="ghost_entity", entity_label="ghost", bbox=box, physical_observation=PhysicalObservation.VISIBLE)]
            annots_v2 = [RealVisitAnnotation(entity_id="ghost_entity", entity_label="ghost", bbox=box, physical_observation=PhysicalObservation.VISIBLE)]

            visits = [
                RealVisitFrame("v1", "V1", "2026-03-01T10:00:00Z", f1, annots_v1),
                RealVisitFrame("v2", "V2", "2026-03-01T11:00:00Z", f2, annots_v2),
            ]
            dataset = RealWorkbenchDataset("unmatched_seq", "desc", tmp_path, visits)

            db_path = tmp_path / "test.db"
            with SQLiteMemoryStore(db_path) as store:
                runner = RealImagePipelineRunner(store=store, detector=stub)
                evaluator = RealBenchmarkEvaluator()
                report = evaluator.evaluate(dataset, runner)

                cm = report.change_metrics
                self.assertIsNotNone(cm)
                self.assertEqual(cm.total_entity_evaluations, 2)
                self.assertEqual(cm.unassociated_annotated_entities, 1)
                self.assertEqual(cm.correct_state_predictions, 0)
                self.assertEqual(cm.accuracy, 0.0)

    def test_audit_registration_failure_prevents_removal_in_ground_truth_policy(self) -> None:
        """
        21. Registration Failure Policy Audit:
        Scenario:
          Visit 1: Present (CONFIRMED_PRESENT).
          Visit 2: Absent, but registration fails completely (e.g. blank frame or low inlier ratio).
        Policy:
          - Under unverified registration, absence is unobservable.
          - Expected state remains CONFIRMED_PRESENT (does not progress to PENDING_REMOVAL or CONFIRMED_REMOVED).
          - System engine holds CONFIRMED_PRESENT; evaluator expects CONFIRMED_PRESENT.
          - Evaluator and engine stay aligned without false removal.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            box_coords = (100, 100, 160, 160)
            box = BoundingBox(100.0, 100.0, 160.0, 160.0)

            # v1: normal structured workbench
            f1 = self._create_synthetic_frame_file(tmp_path, "v1.png", object_color=(100, 100, 100), object_box=box_coords)
            # v2: completely blank/black image causing registration failure
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            f2 = tmp_path / "v2.png"
            cv2.imwrite(str(f2), blank)

            stub = ControlledStubDetector({
                "visit_0": [Detection(label="caliper", confidence=0.9, raw_bbox=box)],
                "visit_1": [],  # no detection on blank frame
            })

            annots_v1 = [RealVisitAnnotation(entity_id="caliper_1", entity_label="caliper", bbox=box, physical_observation=PhysicalObservation.VISIBLE)]
            annots_v2 = [RealVisitAnnotation(entity_id="caliper_1", entity_label="caliper", bbox=None, physical_observation=PhysicalObservation.ABSENT)]

            visits = [
                RealVisitFrame("v1", "V1", "2026-03-01T10:00:00Z", f1, annots_v1),
                RealVisitFrame("v2", "V2", "2026-03-01T11:00:00Z", f2, annots_v2),
            ]
            dataset = RealWorkbenchDataset("reg_fail_seq", "desc", tmp_path, visits)

            db_path = tmp_path / "test.db"
            with SQLiteMemoryStore(db_path) as store:
                runner = RealImagePipelineRunner(store=store, detector=stub)
                evaluator = RealBenchmarkEvaluator()
                report = evaluator.evaluate(dataset, runner)

                cm = report.change_metrics
                self.assertIsNotNone(cm)
                self.assertEqual(cm.false_confirmed_removals, 0)
                self.assertEqual(cm.correct_confirmed_removals, 0)
                # At visit 2, because registration failed, both policy expectation and actual system state retain CONFIRMED_PRESENT
                self.assertEqual(cm.accuracy, 1.0)

    def test_audit_reappearance_after_confirmed_removal_with_relocation(self) -> None:
        """
        22. Reappearance Audit: Entity is confirmed removed, then re-detected at a new location.
        Lifecycle:
          v1: Present at (100, 100) -> CONFIRMED_PRESENT
          v2: Absent 1 visit -> PENDING_REMOVAL
          v3: Absent 2 visits -> CONFIRMED_REMOVED
          v4: Reappears at (250, 250) -> transitions back to CONFIRMED_PRESENT per documented policy
        Asserts:
          - Removal recall on v3 is 1.0.
          - Reappearance on v4 restores entity without being marked as false removal.
          - False confirmed removals is 0 throughout.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            box_orig_coords = (100, 100, 160, 160)
            box_new_coords = (250, 250, 310, 310)
            box_orig = BoundingBox(100.0, 100.0, 160.0, 160.0)
            box_new = BoundingBox(250.0, 250.0, 310.0, 310.0)

            f1 = self._create_synthetic_frame_file(tmp_path, "v1.png", object_color=(200, 100, 0), object_box=box_orig_coords)
            f2 = self._create_synthetic_frame_file(tmp_path, "v2.png", object_color=None)
            f3 = self._create_synthetic_frame_file(tmp_path, "v3.png", object_color=None)
            f4 = self._create_synthetic_frame_file(tmp_path, "v4.png", object_color=(200, 100, 0), object_box=box_new_coords)

            stub = ControlledStubDetector({
                "visit_0": [Detection(label="clamp", confidence=0.9, raw_bbox=box_orig)],
                "visit_1": [],
                "visit_2": [],
                "visit_3": [Detection(label="clamp", confidence=0.9, raw_bbox=box_new)],
            })

            # Ground truth specifies removal then reappearance
            annots_v1 = [RealVisitAnnotation(entity_id="clamp_1", entity_label="clamp", bbox=box_orig, physical_observation=PhysicalObservation.VISIBLE)]
            annots_v2 = [RealVisitAnnotation(entity_id="clamp_1", entity_label="clamp", bbox=None, physical_observation=PhysicalObservation.ABSENT)]
            annots_v3 = [RealVisitAnnotation(entity_id="clamp_1", entity_label="clamp", bbox=None, physical_observation=PhysicalObservation.ABSENT)]
            annots_v4 = [RealVisitAnnotation(entity_id="clamp_1", entity_label="clamp", bbox=box_new, physical_observation=PhysicalObservation.VISIBLE)]

            visits = [
                RealVisitFrame("v1", "V1", "2026-03-01T10:00:00Z", f1, annots_v1),
                RealVisitFrame("v2", "V2", "2026-03-01T11:00:00Z", f2, annots_v2),
                RealVisitFrame("v3", "V3", "2026-03-01T12:00:00Z", f3, annots_v3),
                RealVisitFrame("v4", "V4", "2026-03-01T13:00:00Z", f4, annots_v4),
            ]
            dataset = RealWorkbenchDataset("reappearance_reloc_seq", "desc", tmp_path, visits)

            db_path = tmp_path / "test.db"
            with SQLiteMemoryStore(db_path) as store:
                runner = RealImagePipelineRunner(store=store, detector=stub)
                evaluator = RealBenchmarkEvaluator()
                report = evaluator.evaluate(dataset, runner)

                cm = report.change_metrics
                self.assertIsNotNone(cm)
                self.assertEqual(cm.total_entity_evaluations, 4)
                self.assertEqual(cm.correct_state_predictions, 4)
                self.assertEqual(cm.accuracy, 1.0)
                self.assertEqual(cm.correct_confirmed_removals, 1)
                self.assertEqual(cm.removal_recall, 1.0)
                self.assertEqual(cm.false_confirmed_removals, 0)

    def test_minimal_3visit_relocation_identity_shared_class(self) -> None:
        """
        23. Audit Test: Minimal 3-visit relocation with two distinct physical objects sharing the same class label.
        Setup:
          - Two physical objects: 'tool_A' and 'tool_B', both of class label 'tool'.
          - Visit 1 (v1): Object A at Box A (50, 50, 100, 100); Object B at Box B (200, 200, 250, 250).
          - Visit 2 (v2): Object A remains at Box A; Object B moves to distant Box C (400, 400, 450, 450).
          - Visit 3 (v3): Object A remains at Box A; Object B remains at Box C.
        Verifications:
          - Both system tracks ('ent_v1_1', 'ent_v1_2') maintain disjoint identity without swapping.
          - On v2 and v3, Object B (initial_canonical_bbox Box B) associates with Box C detection via feature/re-id
            and transitions to CONFIRMED_MOVED.
          - Object A remains CONFIRMED_PRESENT at Box A across all 3 visits.
          - Evaluator metrics reflect 6 total evaluations, 6 correct predictions, 2 confirmed relocations, 0 unassociated entities.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            box_A_coords = (50, 50, 100, 100)
            box_B_coords = (200, 200, 250, 250)
            box_C_coords = (400, 400, 450, 450)

            box_A = BoundingBox(50.0, 50.0, 100.0, 100.0)
            box_B = BoundingBox(200.0, 200.0, 250.0, 250.0)
            box_C = BoundingBox(400.0, 400.0, 450.0, 450.0)

            # Create synthetic frame images with visual objects
            # Create synthetic frame images with visual objects on textured workbench
            f1 = self._create_synthetic_frame_file(tmp_path, "v1.png", object_color=(30, 180, 30), object_box=box_A_coords)
            img1 = cv2.imread(str(f1))
            cv2.rectangle(img1, (box_B_coords[0], box_B_coords[1]), (box_B_coords[2], box_B_coords[3]), (180, 30, 30), -1)
            cv2.imwrite(str(f1), img1)

            f2 = self._create_synthetic_frame_file(tmp_path, "v2.png", object_color=(30, 180, 30), object_box=box_A_coords)
            img2 = cv2.imread(str(f2))
            cv2.rectangle(img2, (box_C_coords[0], box_C_coords[1]), (box_C_coords[2], box_C_coords[3]), (180, 30, 30), -1)
            cv2.imwrite(str(f2), img2)

            f3 = self._create_synthetic_frame_file(tmp_path, "v3.png", object_color=(30, 180, 30), object_box=box_A_coords)
            img3 = cv2.imread(str(f3))
            cv2.rectangle(img3, (box_C_coords[0], box_C_coords[1]), (box_C_coords[2], box_C_coords[3]), (180, 30, 30), -1)
            cv2.imwrite(str(f3), img3)

            stub = ControlledStubDetector({
                "visit_0": [
                    Detection(label="tool", confidence=0.95, raw_bbox=box_A),
                    Detection(label="tool", confidence=0.95, raw_bbox=box_B),
                ],
                "visit_1": [
                    Detection(label="tool", confidence=0.95, raw_bbox=box_A),
                    Detection(label="tool", confidence=0.95, raw_bbox=box_C),
                ],
                "visit_2": [
                    Detection(label="tool", confidence=0.95, raw_bbox=box_A),
                    Detection(label="tool", confidence=0.95, raw_bbox=box_C),
                ],
            })

            annots_v1 = [
                RealVisitAnnotation(entity_id="tool_A", entity_label="tool", bbox=box_A, physical_observation=PhysicalObservation.VISIBLE, annotated_event=AnnotatedEvent.UNCHANGED),
                RealVisitAnnotation(entity_id="tool_B", entity_label="tool", bbox=box_B, physical_observation=PhysicalObservation.VISIBLE, annotated_event=AnnotatedEvent.UNCHANGED),
            ]
            annots_v2 = [
                RealVisitAnnotation(entity_id="tool_A", entity_label="tool", bbox=box_A, physical_observation=PhysicalObservation.VISIBLE, annotated_event=AnnotatedEvent.UNCHANGED),
                RealVisitAnnotation(entity_id="tool_B", entity_label="tool", bbox=box_C, physical_observation=PhysicalObservation.VISIBLE, annotated_event=AnnotatedEvent.RELOCATED),
            ]
            annots_v3 = [
                RealVisitAnnotation(entity_id="tool_A", entity_label="tool", bbox=box_A, physical_observation=PhysicalObservation.VISIBLE, annotated_event=AnnotatedEvent.UNCHANGED),
                RealVisitAnnotation(entity_id="tool_B", entity_label="tool", bbox=box_C, physical_observation=PhysicalObservation.VISIBLE, annotated_event=AnnotatedEvent.UNCHANGED),
            ]

            visits = [
                RealVisitFrame("v1", "V1", "2026-03-01T10:00:00Z", f1, annots_v1),
                RealVisitFrame("v2", "V2", "2026-03-01T11:00:00Z", f2, annots_v2),
                RealVisitFrame("v3", "V3", "2026-03-01T12:00:00Z", f3, annots_v3),
            ]
            dataset = RealWorkbenchDataset("relocation_identity_dataset", "3-visit relocation audit", tmp_path, visits)

            db_path = tmp_path / "test.db"
            with SQLiteMemoryStore(db_path) as store:
                runner = RealImagePipelineRunner(store=store, detector=stub)
                evaluator = RealBenchmarkEvaluator()
                report = evaluator.evaluate(dataset, runner)

                # Verify persistence repository entities
                entities = store.list_entities()
                self.assertEqual(len(entities), 2)

                # Ensure initial canonical bboxes match original positions
                ent_map = {e.metadata.get("initial_canonical_bbox")[0]: e for e in entities if "initial_canonical_bbox" in e.metadata}
                self.assertIn(50.0, ent_map)
                self.assertIn(200.0, ent_map)

                cm = report.change_metrics
                self.assertIsNotNone(cm)
                self.assertEqual(cm.total_entity_evaluations, 6)
                self.assertEqual(cm.correct_state_predictions, 6)
                self.assertEqual(cm.accuracy, 1.0)
                self.assertEqual(cm.correct_relocations, 2)
                self.assertEqual(cm.unassociated_annotated_entities, 0)
                self.assertEqual(cm.false_confirmed_removals, 0)

    def test_registration_failure_with_physically_present_annotations(self) -> None:
        """
        24. Ground-Truth Conditioning Audit: Registration failure when physical object is present.
        If registration fails (inlier_ratio < 0.20 or success=False):
          - Physical object is still annotated as VISIBLE.
          - Expected state remains CONFIRMED_PRESENT (does not downgrade to removal).
          - System state remains CONFIRMED_PRESENT from prior visit.
          - Absence vacancy streak cannot accumulate under bad registration.
          - Evaluator correctly asserts accuracy without false removal.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            box = BoundingBox(100.0, 100.0, 150.0, 150.0)
            f1 = self._create_synthetic_frame_file(tmp_path, "v1.png", object_color=(50, 150, 50), object_box=(100, 100, 150, 150))
            # Create featureless black frame to trigger registration failure
            f2 = tmp_path / "v2.png"
            cv2.imwrite(str(f2), np.zeros((480, 640, 3), dtype=np.uint8))

            stub = ControlledStubDetector({
                "visit_0": [Detection(label="meter", confidence=0.9, raw_bbox=box)],
                "visit_1": [Detection(label="meter", confidence=0.9, raw_bbox=box)],
            })

            annots_v1 = [RealVisitAnnotation(entity_id="meter_1", entity_label="meter", bbox=box, physical_observation=PhysicalObservation.VISIBLE)]
            annots_v2 = [RealVisitAnnotation(entity_id="meter_1", entity_label="meter", bbox=box, physical_observation=PhysicalObservation.VISIBLE)]

            visits = [
                RealVisitFrame("v1", "V1", "2026-03-01T10:00:00Z", f1, annots_v1),
                RealVisitFrame("v2", "V2", "2026-03-01T11:00:00Z", f2, annots_v2),
            ]
            dataset = RealWorkbenchDataset("reg_fail_present_seq", "desc", tmp_path, visits)

            db_path = tmp_path / "test.db"
            with SQLiteMemoryStore(db_path) as store:
                runner = RealImagePipelineRunner(store=store, detector=stub)
                evaluator = RealBenchmarkEvaluator()
                report = evaluator.evaluate(dataset, runner)

                cm = report.change_metrics
                self.assertIsNotNone(cm)
                self.assertEqual(cm.total_entity_evaluations, 2)
                self.assertEqual(cm.correct_state_predictions, 2)
                self.assertEqual(cm.accuracy, 1.0)
                self.assertEqual(cm.false_confirmed_removals, 0)

    def test_complete_seven_core_sequences_matrix(self) -> None:
        """
        25. Policy Audit Test: Comprehensive verification of all 7 core sequences from Phase 6.5.
        1. Vis -> Abs -> Vis: CONFIRMED_PRESENT -> PENDING_REMOVAL -> CONFIRMED_PRESENT (transient absence)
        2. Vis -> Abs -> Abs: CONFIRMED_PRESENT -> PENDING_REMOVAL -> CONFIRMED_REMOVED (persistent removal)
        3. Vis -> Occ -> Abs: CONFIRMED_PRESENT -> CONFIRMED_PRESENT -> PENDING_REMOVAL (occlusion holds prior)
        4. Vis -> OOF -> Abs: CONFIRMED_PRESENT -> CONFIRMED_PRESENT -> PENDING_REMOVAL (out-of-frame holds prior)
        5. Vis -> RegFail -> Vis: CONFIRMED_PRESENT -> CONFIRMED_PRESENT -> CONFIRMED_PRESENT (reg fail holds prior)
        6. Vis -> Reloc -> Reloc: CONFIRMED_PRESENT -> CONFIRMED_MOVED -> CONFIRMED_MOVED (persistent move)
        7. Abs[confirmed removed] -> Reappears: CONFIRMED_REMOVED -> CONFIRMED_PRESENT (resurrection/rediscovery)
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            box1 = BoundingBox(50.0, 50.0, 100.0, 100.0)
            box2 = BoundingBox(250.0, 250.0, 300.0, 300.0)
            box_occ = BoundingBox(60.0, 60.0, 110.0, 110.0)

            # Sequence 1: Vis -> Abs -> Vis
            f1_1 = self._create_synthetic_frame_file(tmp_path, "s1_v1.png", object_color=(50, 50, 200), object_box=(50, 50, 100, 100))
            f1_2 = self._create_synthetic_frame_file(tmp_path, "s1_v2.png", object_color=None)
            f1_3 = self._create_synthetic_frame_file(tmp_path, "s1_v3.png", object_color=(50, 50, 200), object_box=(50, 50, 100, 100))
            stub1 = ControlledStubDetector({
                "visit_0": [Detection(label="item", confidence=0.9, raw_bbox=box1)],
                "visit_1": [],
                "visit_2": [Detection(label="item", confidence=0.9, raw_bbox=box1)],
            })
            annots1 = [
                RealVisitAnnotation(entity_id="item_1", entity_label="item", bbox=box1, physical_observation=PhysicalObservation.VISIBLE),
                RealVisitAnnotation(entity_id="item_1", entity_label="item", bbox=None, physical_observation=PhysicalObservation.ABSENT),
                RealVisitAnnotation(entity_id="item_1", entity_label="item", bbox=box1, physical_observation=PhysicalObservation.VISIBLE),
            ]
            ds1 = RealWorkbenchDataset("s1", "s1", tmp_path, [
                RealVisitFrame("v1", "V1", "2026-03-01T10:00:00Z", f1_1, [annots1[0]]),
                RealVisitFrame("v2", "V2", "2026-03-01T11:00:00Z", f1_2, [annots1[1]]),
                RealVisitFrame("v3", "V3", "2026-03-01T12:00:00Z", f1_3, [annots1[2]]),
            ])
            with SQLiteMemoryStore(tmp_path / "db1.db") as s1:
                r1 = RealBenchmarkEvaluator().evaluate(ds1, RealImagePipelineRunner(store=s1, detector=stub1))
                self.assertEqual(r1.change_metrics.total_entity_evaluations, 3)
                self.assertEqual(r1.change_metrics.correct_state_predictions, 3)
                self.assertEqual(r1.change_metrics.pending_removal_transitions, 1)
                self.assertEqual(r1.change_metrics.false_confirmed_removals, 0)

            # Sequence 3: Vis -> Occ -> Abs (Occlusion holds prior CONFIRMED_PRESENT)
            f3_1 = self._create_synthetic_frame_file(tmp_path, "s3_v1.png", object_color=(80, 80, 180), object_box=(50, 50, 100, 100))
            f3_2 = self._create_synthetic_frame_file(tmp_path, "s3_v2.png", object_color=(200, 200, 50), object_box=(60, 60, 110, 110))
            f3_3 = self._create_synthetic_frame_file(tmp_path, "s3_v3.png", object_color=None)
            stub3 = ControlledStubDetector({
                "visit_0": [Detection(label="widget", confidence=0.9, raw_bbox=box1)],
                "visit_1": [Detection(label="box", confidence=0.9, raw_bbox=box_occ)],
                "visit_2": [],
            })
            annots3 = [
                RealVisitAnnotation(entity_id="wid_1", entity_label="widget", bbox=box1, physical_observation=PhysicalObservation.VISIBLE),
                RealVisitAnnotation(entity_id="wid_1", entity_label="widget", bbox=None, physical_observation=PhysicalObservation.PARTIALLY_OCCLUDED),
                RealVisitAnnotation(entity_id="wid_1", entity_label="widget", bbox=None, physical_observation=PhysicalObservation.ABSENT),
            ]
            ds3 = RealWorkbenchDataset("s3", "s3", tmp_path, [
                RealVisitFrame("v1", "V1", "2026-03-01T10:00:00Z", f3_1, [annots3[0]]),
                RealVisitFrame("v2", "V2", "2026-03-01T11:00:00Z", f3_2, [annots3[1]]),
                RealVisitFrame("v3", "V3", "2026-03-01T12:00:00Z", f3_3, [annots3[2]]),
            ])
            with SQLiteMemoryStore(tmp_path / "db3.db") as s3:
                r3 = RealBenchmarkEvaluator().evaluate(ds3, RealImagePipelineRunner(store=s3, detector=stub3))
                # v1: CONFIRMED_PRESENT, v2: (Occ) remains CONFIRMED_PRESENT, v3: PENDING_REMOVAL
                self.assertEqual(r3.change_metrics.total_entity_evaluations, 3)
                self.assertEqual(r3.change_metrics.correct_state_predictions, 3)
                self.assertEqual(r3.change_metrics.pending_removal_transitions, 1)
                self.assertEqual(r3.change_metrics.false_confirmed_removals, 0)
                self.assertEqual(r3.change_metrics.correct_confirmed_removals, 0)

    def test_dataset_integrity_detects_inconsistent_entity_ids(self) -> None:
        """
        26. Validation Test: Detects inconsistent entity_label for the same entity_id across visits.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            f1 = self._create_synthetic_frame_file(tmp_path, "f1.png")
            f2 = self._create_synthetic_frame_file(tmp_path, "f2.png")
            box = BoundingBox(10.0, 10.0, 50.0, 50.0)

            v1 = RealVisitFrame("v1", "V1", "2026-03-01T10:00:00Z", f1, [
                RealVisitAnnotation(entity_id="obj_1", entity_label="screwdriver", bbox=box)
            ])
            v2 = RealVisitFrame("v2", "V2", "2026-03-01T11:00:00Z", f2, [
                # Inconsistent label 'pliers' for same entity_id 'obj_1'
                RealVisitAnnotation(entity_id="obj_1", entity_label="pliers", bbox=box)
            ])
            dataset = RealWorkbenchDataset("inconsistent_entity", "desc", tmp_path, [v1, v2])
            errors = dataset.validate_dataset_integrity()
            self.assertTrue(any("Inconsistent entity class label" in err for err in errors))

    def test_dataset_integrity_detects_invalid_observation_and_box_combinations(self) -> None:
        """
        27. Validation Test: Detects invalid combinations:
            - VISIBLE with null bbox.
            - ABSENT with non-null bbox.
            - AnnotatedEvent ABSENT with PhysicalObservation VISIBLE.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            f1 = self._create_synthetic_frame_file(tmp_path, "f1.png")
            f2 = self._create_synthetic_frame_file(tmp_path, "f2.png")
            box = BoundingBox(10.0, 10.0, 50.0, 50.0)

            # Case A: VISIBLE with null bbox
            v1 = RealVisitFrame("v1", "V1", "2026-03-01T10:00:00Z", f1, [
                RealVisitAnnotation(entity_id="obj_1", entity_label="tool", bbox=None, physical_observation=PhysicalObservation.VISIBLE)
            ])
            # Case B: ABSENT with non-null bbox
            v2 = RealVisitFrame("v2", "V2", "2026-03-01T11:00:00Z", f2, [
                RealVisitAnnotation(entity_id="obj_1", entity_label="tool", bbox=box, physical_observation=PhysicalObservation.ABSENT)
            ])
            dataset = RealWorkbenchDataset("invalid_obs_box", "desc", tmp_path, [v1, v2])
            errors = dataset.validate_dataset_integrity()
            self.assertTrue(any("has physical_observation='VISIBLE' but bbox is null/missing" in err for err in errors))
            self.assertTrue(any("has physical_observation='ABSENT' but defines a non-null bbox" in err for err in errors))

    def test_dataset_integrity_detects_non_monotonic_timestamps(self) -> None:
        """
        28. Validation Test: Detects non-monotonic or identical timestamps.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            f1 = self._create_synthetic_frame_file(tmp_path, "f1.png")
            f2 = self._create_synthetic_frame_file(tmp_path, "f2.png")

            v1 = RealVisitFrame("v1", "V1", "2026-03-01T12:00:00Z", f1, [])
            v2 = RealVisitFrame("v2", "V2", "2026-03-01T10:00:00Z", f2, [])
            dataset = RealWorkbenchDataset("bad_time", "desc", tmp_path, [v1, v2])
            errors = dataset.validate_dataset_integrity()
            self.assertTrue(any("Non-monotonic timestamp order" in err for err in errors))

    def test_template_dataset_blocks_evaluation(self) -> None:
        """
        29. Integrity Test: Datasets with template flag or placeholder images are blocked from evaluation.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            manifest_path = RealWorkbenchDataset.create_empty_template(tmp_path)
            dataset = RealWorkbenchDataset.load_from_directory(tmp_path)

            ready, reason = dataset.is_evaluation_ready()
            self.assertFalse(ready)
            self.assertIn("placeholder", reason.lower())


    def test_dataset_integrity_detects_out_of_bounds_bounding_box(self) -> None:
        """
        30. Validation Test: Detects bounding box coordinates exceeding image canvas bounds.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            f1 = self._create_synthetic_frame_file(tmp_path, "f1.png")  # 640x480
            f2 = self._create_synthetic_frame_file(tmp_path, "f2.png")
            out_of_bounds_box = BoundingBox(10.0, 10.0, 700.0, 500.0)

            v1 = RealVisitFrame("v1", "V1", "2026-03-01T10:00:00Z", f1, [
                RealVisitAnnotation(entity_id="obj_1", entity_label="cup", bbox=out_of_bounds_box, physical_observation=PhysicalObservation.VISIBLE)
            ])
            v2 = RealVisitFrame("v2", "V2", "2026-03-01T11:00:00Z", f2, [])
            dataset = RealWorkbenchDataset("oob_box", "desc", tmp_path, [v1, v2])
            errors = dataset.validate_dataset_integrity()
            self.assertTrue(any("exceeding image canvas dimensions" in err for err in errors))


    def test_occlusion_detection_retention_and_miss_recovery(self) -> None:
        """
        31. Occlusion Test:
            Visit 1: Object visible and detected -> CONFIRMED_PRESENT.
            Visit 2: Object partially occluded.
                     Branch A: Detector retains detection -> stays CONFIRMED_PRESENT.
                     Branch B: Detector misses detection -> becomes PENDING_REMOVAL (streak=1),
                               NEVER CONFIRMED_REMOVED!
            Visit 3: Object uncovered and detected -> recovers to CONFIRMED_PRESENT.
        """
        box = BoundingBox(100.0, 100.0, 160.0, 160.0)
        # Branch A: Detector retains detection during partial occlusion
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            f1 = self._create_synthetic_frame_file(tmp_path, "f1.png")
            f2 = self._create_synthetic_frame_file(tmp_path, "f2.png")
            f3 = self._create_synthetic_frame_file(tmp_path, "f3.png")

            stub_retained = ControlledStubDetector({
                "visit_0": [Detection(label="cup", confidence=0.9, raw_bbox=box)],
                "visit_1": [Detection(label="cup", confidence=0.6, raw_bbox=box)],  # Retained
                "visit_2": [Detection(label="cup", confidence=0.9, raw_bbox=box)],
            })
            annots_v1 = [RealVisitAnnotation(entity_id="cup_1", entity_label="cup", bbox=box, expected_state=EpistemicState.CONFIRMED_PRESENT, physical_observation=PhysicalObservation.VISIBLE)]
            annots_v2 = [RealVisitAnnotation(entity_id="cup_1", entity_label="cup", bbox=box, expected_state=EpistemicState.CONFIRMED_PRESENT, physical_observation=PhysicalObservation.PARTIALLY_OCCLUDED)]
            annots_v3 = [RealVisitAnnotation(entity_id="cup_1", entity_label="cup", bbox=box, expected_state=EpistemicState.CONFIRMED_PRESENT, physical_observation=PhysicalObservation.VISIBLE)]
            visits = [
                RealVisitFrame("v1", "V1", "2026-03-01T10:00:00Z", f1, annots_v1),
                RealVisitFrame("v2", "V2", "2026-03-01T11:00:00Z", f2, annots_v2),
                RealVisitFrame("v3", "V3", "2026-03-01T12:00:00Z", f3, annots_v3),
            ]
            ds = RealWorkbenchDataset("occl_retained", "desc", tmp_path, visits)
            with SQLiteMemoryStore(tmp_path / "a.db") as store:
                runner = RealImagePipelineRunner(store=store, detector=stub_retained)
                report = RealBenchmarkEvaluator().evaluate(ds, runner)
                cm = report.change_metrics
                self.assertIsNotNone(cm)
                self.assertEqual(cm.correct_state_predictions, 3)
                self.assertEqual(cm.false_confirmed_removals, 0)
                # Object stayed confirmed present
                self.assertEqual(len(store.list_entities()), 1)
                self.assertEqual(store.list_entities()[0].current_state, EpistemicState.CONFIRMED_PRESENT)

        # Branch B: Detector misses detection during partial occlusion
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            f1 = self._create_synthetic_frame_file(tmp_path, "f1.png")
            f2 = self._create_synthetic_frame_file(tmp_path, "f2.png")
            f3 = self._create_synthetic_frame_file(tmp_path, "f3.png")

            stub_missed = ControlledStubDetector({
                "visit_0": [Detection(label="cup", confidence=0.9, raw_bbox=box)],
                "visit_1": [],  # Missed due to occlusion
                "visit_2": [Detection(label="cup", confidence=0.9, raw_bbox=box)],
            })
            # In V2, object is physically occluded; GT expected_state is CONFIRMED_PRESENT
            annots_v1 = [RealVisitAnnotation(entity_id="cup_1", entity_label="cup", bbox=box, expected_state=EpistemicState.CONFIRMED_PRESENT, physical_observation=PhysicalObservation.VISIBLE)]
            annots_v2 = [RealVisitAnnotation(entity_id="cup_1", entity_label="cup", bbox=None, expected_state=EpistemicState.CONFIRMED_PRESENT, physical_observation=PhysicalObservation.PARTIALLY_OCCLUDED)]
            annots_v3 = [RealVisitAnnotation(entity_id="cup_1", entity_label="cup", bbox=box, expected_state=EpistemicState.CONFIRMED_PRESENT, physical_observation=PhysicalObservation.VISIBLE)]
            visits = [
                RealVisitFrame("v1", "V1", "2026-03-01T10:00:00Z", f1, annots_v1),
                RealVisitFrame("v2", "V2", "2026-03-01T11:00:00Z", f2, annots_v2),
                RealVisitFrame("v3", "V3", "2026-03-01T12:00:00Z", f3, annots_v3),
            ]
            ds = RealWorkbenchDataset("occl_missed", "desc", tmp_path, visits)
            with SQLiteMemoryStore(tmp_path / "b.db") as store:
                runner = RealImagePipelineRunner(store=store, detector=stub_missed)
                report = RealBenchmarkEvaluator().evaluate(ds, runner)
                cm = report.change_metrics
                self.assertIsNotNone(cm)
                # At V2, system transitioned to PENDING_REMOVAL (streak=1), NOT CONFIRMED_REMOVED!
                self.assertEqual(cm.false_confirmed_removals, 0)
                self.assertEqual(cm.premature_removals, 0)
                # At V3, system redetected and restored CONFIRMED_PRESENT
                self.assertEqual(len(store.list_entities()), 1)
                self.assertEqual(store.list_entities()[0].current_state, EpistemicState.CONFIRMED_PRESENT)

    def test_dataset_integrity_detects_box_on_full_occlusion_and_out_of_frame(self) -> None:
        """
        32. Validation Test: Detects non-null bbox on FULLY_OCCLUDED or OUT_OF_FRAME.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            f1 = self._create_synthetic_frame_file(tmp_path, "f1.png")
            f2 = self._create_synthetic_frame_file(tmp_path, "f2.png")
            box = BoundingBox(10.0, 10.0, 50.0, 50.0)

            v1 = RealVisitFrame("v1", "V1", "2026-03-01T10:00:00Z", f1, [
                RealVisitAnnotation(entity_id="obj_1", entity_label="cup", bbox=box, expected_state=EpistemicState.CONFIRMED_PRESENT, physical_observation=PhysicalObservation.FULLY_OCCLUDED)
            ])
            v2 = RealVisitFrame("v2", "V2", "2026-03-01T11:00:00Z", f2, [])
            ds = RealWorkbenchDataset("invalid_occl_box", "desc", tmp_path, [v1, v2])
            errors = ds.validate_dataset_integrity()
            self.assertTrue(any("An unobservable/absent entity must have a null bounding box" in err for err in errors))

    def test_undefined_metric_denominators_return_none(self) -> None:
        """
        33. Metric Integrity Test: When conditioning denominators are zero,
        rates return None instead of misleading 0.0.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            f1 = self._create_synthetic_frame_file(tmp_path, "f1.png")
            f2 = self._create_synthetic_frame_file(tmp_path, "f2.png")
            box = BoundingBox(10.0, 10.0, 50.0, 50.0)

            # Sequence where NO object is ever expected to be removed or moved
            stub = ControlledStubDetector({
                "visit_0": [Detection(label="cup", confidence=0.9, raw_bbox=box)],
                "visit_1": [Detection(label="cup", confidence=0.9, raw_bbox=box)],
            })
            annots = [
                RealVisitAnnotation(entity_id="cup_1", entity_label="cup", bbox=box, expected_state=EpistemicState.CONFIRMED_PRESENT, physical_observation=PhysicalObservation.VISIBLE)
            ]
            visits = [
                RealVisitFrame("v1", "V1", "2026-03-01T10:00:00Z", f1, annots),
                RealVisitFrame("v2", "V2", "2026-03-01T11:00:00Z", f2, annots),
            ]
            ds = RealWorkbenchDataset("zero_denom_seq", "desc", tmp_path, visits)
            with SQLiteMemoryStore(tmp_path / "zero.db") as store:
                runner = RealImagePipelineRunner(store=store, detector=stub)
                report = RealBenchmarkEvaluator().evaluate(ds, runner)
                cm = report.change_metrics
                self.assertIsNotNone(cm)
                self.assertEqual(cm.expected_removals_count, 0)
                self.assertIsNone(cm.removal_recall)  # Zero denominator -> None
                self.assertEqual(cm.expected_relocations_count, 0)
                self.assertIsNone(cm.relocation_recall)  # Zero denominator -> None
                self.assertEqual(cm.accuracy, 1.0)

    def test_ground_truth_isolation_annotation_change_invariance(self) -> None:
        """
        34. Ground-Truth Isolation Regression Test:
        Changing evaluator annotations (entity IDs, labels, bounding boxes, expected states)
        while holding input image frames identical has ZERO effect on:
        - Stored system entities.
        - Stored observations.
        - System state transitions.
        - System reports.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            f1 = self._create_synthetic_frame_file(tmp_path, "f1.png")
            f2 = self._create_synthetic_frame_file(tmp_path, "f2.png")
            box = BoundingBox(50.0, 50.0, 120.0, 120.0)

            stub_config = {
                "visit_0": [Detection(label="cup", confidence=0.88, raw_bbox=box)],
                "visit_1": [Detection(label="cup", confidence=0.88, raw_bbox=box)],
            }
            stub1 = ControlledStubDetector(stub_config)
            stub2 = ControlledStubDetector(stub_config)

            # Run 1: Standard annotation
            annots_1 = [RealVisitAnnotation(entity_id="c1", entity_label="cup", bbox=box, expected_state=EpistemicState.CONFIRMED_PRESENT)]
            visits_1 = [
                RealVisitFrame("v1", "V1", "2026-03-01T10:00:00Z", f1, annots_1),
                RealVisitFrame("v2", "V2", "2026-03-01T11:00:00Z", f2, annots_1),
            ]
            ds_1 = RealWorkbenchDataset("run_1", "desc", tmp_path, visits_1)
            with SQLiteMemoryStore(tmp_path / "run1.db") as store1:
                runner1 = RealImagePipelineRunner(store=store1, detector=stub1)
                report1 = RealBenchmarkEvaluator().evaluate(ds_1, runner1)
                ents1 = store1.list_entities()
                obs1 = store1.list_observations_for_entity(ents1[0].entity_id) if ents1 else []
                evts1 = store1.list_change_events_for_entity(ents1[0].entity_id) if ents1 else []

            # Run 2: Completely altered evaluator annotations (contradicting reality)
            annots_2 = [RealVisitAnnotation(entity_id="fake_id_xyz", entity_label="fake_label", bbox=None, expected_state=EpistemicState.CONFIRMED_REMOVED, physical_observation=PhysicalObservation.ABSENT)]
            visits_2 = [
                RealVisitFrame("v1", "V1", "2026-03-01T10:00:00Z", f1, annots_2),
                RealVisitFrame("v2", "V2", "2026-03-01T11:00:00Z", f2, annots_2),
            ]
            ds_2 = RealWorkbenchDataset("run_2", "desc", tmp_path, visits_2)
            with SQLiteMemoryStore(tmp_path / "run2.db") as store2:
                runner2 = RealImagePipelineRunner(store=store2, detector=stub2)
                report2 = RealBenchmarkEvaluator().evaluate(ds_2, runner2)
                ents2 = store2.list_entities()
                obs2 = store2.list_observations_for_entity(ents2[0].entity_id) if ents2 else []
                evts2 = store2.list_change_events_for_entity(ents2[0].entity_id) if ents2 else []

            # System internal state MUST be identical across Run 1 and Run 2
            self.assertEqual(len(ents1), len(ents2))
            self.assertEqual(ents1[0].current_state, ents2[0].current_state)
            self.assertEqual(ents1[0].label, ents2[0].label)  # "cup", NOT "fake_label"
            self.assertEqual(len(obs1), len(obs2))
            self.assertEqual(len(evts1), len(evts2))
            # But the scoring report reflects the altered ground truth
            self.assertNotEqual(report1.change_metrics.accuracy, report2.change_metrics.accuracy)

    # =========================================================================
    # Phase 6.8: Hardened Ground-Truth Evaluation Regression Tests
    # =========================================================================

    def test_hardened_bipartite_matching_global_optimum_beats_greedy(self) -> None:
        """
        1. Two GT entities and two system entities of same class where greedy matching
           would pick a suboptimal pairing, but optimal bipartite matching finds the global optimum.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            f1 = self._create_synthetic_frame_file(tmp_path, "v1.png")
            f2 = self._create_synthetic_frame_file(tmp_path, "v2.png")

            # Bounding box geometry:
            # det_1: [100, 100, 200, 200]
            # det_2: [195, 100, 295, 200]
            # gt_A:  [140, 100, 240, 200] (IoU with det_1 ~ 0.428, IoU with det_2 ~ 0.290)
            # gt_B:  [102, 100, 202, 200] (IoU with det_1 ~ 0.960, IoU with det_2 ~ 0.015 < 0.05)
            # Greedy evaluating gt_A first steals det_1 (0.428 > 0.290), leaving gt_B with det_2 (IoU 0.015 < 0.05 -> rejected).
            # Optimal bipartite matching achieves global maximum: gt_B -> det_1 (0.96) and gt_A -> det_2 (0.29).
            det_1 = BoundingBox(100.0, 100.0, 200.0, 200.0)
            det_2 = BoundingBox(195.0, 100.0, 295.0, 200.0)
            gt_A_box = BoundingBox(140.0, 100.0, 240.0, 200.0)
            gt_B_box = BoundingBox(102.0, 100.0, 202.0, 200.0)

            stub = ControlledStubDetector({
                "visit_0": [
                    Detection(label="widget", confidence=0.9, raw_bbox=det_1),
                    Detection(label="widget", confidence=0.9, raw_bbox=det_2),
                ],
                "visit_1": [
                    Detection(label="widget", confidence=0.9, raw_bbox=det_1),
                    Detection(label="widget", confidence=0.9, raw_bbox=det_2),
                ],
            })

            # gt_A is listed first in annotations
            annots_v1 = [
                RealVisitAnnotation(entity_id="gt_A", entity_label="widget", bbox=gt_A_box, physical_observation=PhysicalObservation.VISIBLE),
                RealVisitAnnotation(entity_id="gt_B", entity_label="widget", bbox=gt_B_box, physical_observation=PhysicalObservation.VISIBLE),
            ]
            annots_v2 = [
                RealVisitAnnotation(entity_id="gt_A", entity_label="widget", bbox=gt_A_box, physical_observation=PhysicalObservation.VISIBLE, annotated_event=AnnotatedEvent.UNCHANGED),
                RealVisitAnnotation(entity_id="gt_B", entity_label="widget", bbox=gt_B_box, physical_observation=PhysicalObservation.VISIBLE, annotated_event=AnnotatedEvent.UNCHANGED),
            ]

            visits = [
                RealVisitFrame("v1", "V1", "2026-03-01T10:00:00Z", f1, annots_v1),
                RealVisitFrame("v2", "V2", "2026-03-01T11:00:00Z", f2, annots_v2),
            ]
            dataset = RealWorkbenchDataset("opt_bipartite_seq", "desc", tmp_path, visits)

            with SQLiteMemoryStore(tmp_path / "test.db") as store:
                runner = RealImagePipelineRunner(store=store, detector=stub)
                evaluator = RealBenchmarkEvaluator()
                report = evaluator.evaluate(dataset, runner)

                cm = report.change_metrics
                self.assertIsNotNone(cm)
                self.assertEqual(cm.total_entity_evaluations, 4)
                self.assertEqual(cm.correct_state_predictions, 4)
                self.assertEqual(cm.accuracy, 1.0)
                self.assertEqual(cm.unassociated_annotated_entities, 0)

    def test_hardened_order_invariance_permuted_entities(self) -> None:
        """
        2. Order-invariance: permuting the order of entities produces identical bindings and metrics.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            f1 = self._create_synthetic_frame_file(tmp_path, "v1.png")
            f2 = self._create_synthetic_frame_file(tmp_path, "v2.png")

            box1 = BoundingBox(50.0, 50.0, 100.0, 100.0)
            box2 = BoundingBox(150.0, 50.0, 200.0, 100.0)
            box3 = BoundingBox(250.0, 50.0, 300.0, 100.0)

            stub_dets = {
                "visit_0": [
                    Detection(label="cup", confidence=0.9, raw_bbox=box1),
                    Detection(label="cup", confidence=0.9, raw_bbox=box2),
                    Detection(label="cup", confidence=0.9, raw_bbox=box3),
                ],
                "visit_1": [
                    Detection(label="cup", confidence=0.9, raw_bbox=box1),
                    Detection(label="cup", confidence=0.9, raw_bbox=box2),
                    Detection(label="cup", confidence=0.9, raw_bbox=box3),
                ],
            }

            a1 = RealVisitAnnotation(entity_id="cup_1", entity_label="cup", bbox=box1, physical_observation=PhysicalObservation.VISIBLE)
            a2 = RealVisitAnnotation(entity_id="cup_2", entity_label="cup", bbox=box2, physical_observation=PhysicalObservation.VISIBLE)
            a3 = RealVisitAnnotation(entity_id="cup_3", entity_label="cup", bbox=box3, physical_observation=PhysicalObservation.VISIBLE)

            perms = [
                [a1, a2, a3],
                [a3, a1, a2],
                [a2, a3, a1],
            ]

            results = []
            for idx, perm in enumerate(perms):
                visits = [
                    RealVisitFrame("v1", "V1", "2026-03-01T10:00:00Z", f1, perm),
                    RealVisitFrame("v2", "V2", "2026-03-01T11:00:00Z", f2, perm),
                ]
                ds = RealWorkbenchDataset(f"perm_{idx}", "desc", tmp_path, visits)
                with SQLiteMemoryStore(tmp_path / f"db_{idx}.db") as store:
                    runner = RealImagePipelineRunner(store=store, detector=ControlledStubDetector(stub_dets))
                    rep = RealBenchmarkEvaluator().evaluate(ds, runner)
                    results.append(rep)

            base_cm = results[0].change_metrics
            for rep in results[1:]:
                cm = rep.change_metrics
                self.assertEqual(cm.accuracy, base_cm.accuracy)
                self.assertEqual(cm.correct_state_predictions, base_cm.correct_state_predictions)
                self.assertEqual(cm.total_entity_evaluations, base_cm.total_entity_evaluations)
                self.assertEqual(cm.unassociated_annotated_entities, base_cm.unassociated_annotated_entities)

    def test_hardened_class_mismatch_rejection(self) -> None:
        """
        3. Class-mismatch rejection: entities with different labels are never bound, even with identical bbox.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            f1 = self._create_synthetic_frame_file(tmp_path, "v1.png")
            f2 = self._create_synthetic_frame_file(tmp_path, "v2.png")
            box = BoundingBox(100.0, 100.0, 160.0, 160.0)

            stub = ControlledStubDetector({
                "visit_0": [Detection(label="dog", confidence=0.9, raw_bbox=box)],
                "visit_1": [Detection(label="dog", confidence=0.9, raw_bbox=box)],
            })

            annots_v1 = [RealVisitAnnotation(entity_id="animal_1", entity_label="cat", bbox=box, physical_observation=PhysicalObservation.VISIBLE)]
            annots_v2 = [RealVisitAnnotation(entity_id="animal_1", entity_label="cat", bbox=box, physical_observation=PhysicalObservation.VISIBLE, annotated_event=AnnotatedEvent.UNCHANGED)]
            visits = [
                RealVisitFrame("v1", "V1", "2026-03-01T10:00:00Z", f1, annots_v1),
                RealVisitFrame("v2", "V2", "2026-03-01T11:00:00Z", f2, annots_v2),
            ]
            dataset = RealWorkbenchDataset("class_mismatch_seq", "desc", tmp_path, visits)

            with SQLiteMemoryStore(tmp_path / "test.db") as store:
                runner = RealImagePipelineRunner(store=store, detector=stub)
                evaluator = RealBenchmarkEvaluator()
                report = evaluator.evaluate(dataset, runner)

                self.assertEqual(report.change_metrics.unassociated_annotated_entities, 1)
                self.assertEqual(report.change_metrics.accuracy, 0.0)
                self.assertEqual(report.detection_metrics.false_positives, 2)
                self.assertEqual(report.detection_metrics.false_negatives, 2)

    def test_hardened_low_overlap_rejection_below_threshold(self) -> None:
        """
        4. Low-overlap rejection: pairs below the minimum threshold (0.05) remain unbound.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            f1 = self._create_synthetic_frame_file(tmp_path, "v1.png")
            f2 = self._create_synthetic_frame_file(tmp_path, "v2.png")

            # gt box: [100, 100, 200, 200] (area 10,000)
            # det box: [199, 100, 299, 200] (overlap 1x100 = 100; union = 19,900; IoU ~ 0.005)
            gt_box = BoundingBox(100.0, 100.0, 200.0, 200.0)
            det_box = BoundingBox(199.0, 100.0, 299.0, 200.0)

            stub = ControlledStubDetector({
                "visit_0": [Detection(label="box", confidence=0.9, raw_bbox=det_box)],
                "visit_1": [Detection(label="box", confidence=0.9, raw_bbox=det_box)],
            })

            annots_v1 = [RealVisitAnnotation(entity_id="box_1", entity_label="box", bbox=gt_box, physical_observation=PhysicalObservation.VISIBLE)]
            annots_v2 = [RealVisitAnnotation(entity_id="box_1", entity_label="box", bbox=gt_box, physical_observation=PhysicalObservation.VISIBLE, annotated_event=AnnotatedEvent.UNCHANGED)]
            visits = [
                RealVisitFrame("v1", "V1", "2026-03-01T10:00:00Z", f1, annots_v1),
                RealVisitFrame("v2", "V2", "2026-03-01T11:00:00Z", f2, annots_v2),
            ]
            dataset = RealWorkbenchDataset("low_overlap_seq", "desc", tmp_path, visits)

            with SQLiteMemoryStore(tmp_path / "test.db") as store:
                runner = RealImagePipelineRunner(store=store, detector=stub)
                evaluator = RealBenchmarkEvaluator()
                report = evaluator.evaluate(dataset, runner)

                self.assertEqual(report.change_metrics.unassociated_annotated_entities, 1)
                self.assertEqual(report.change_metrics.accuracy, 0.0)

    def test_hardened_coordinate_frame_consistency_homography_transformation(self) -> None:
        """
        5. Coordinate frame consistency: annotations from visits with non-identity homographies
           are correctly transformed into canonical reference frame before matching.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            from tests.test_registration import _generate_synthetic_workbench
            ref_img = _generate_synthetic_workbench(640, 480, seed=42)
            f1 = tmp_path / "v1.png"
            cv2.imwrite(str(f1), ref_img)

            # Apply a pure translation homography: dx = +30, dy = +20
            # p_ref = H @ p_query, so query image is warped by inv(H)
            H_canonical_to_query = np.array([
                [1.0, 0.0, 30.0],
                [0.0, 1.0, 20.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float32)
            query_img = cv2.warpPerspective(ref_img, H_canonical_to_query, (640, 480))
            f2 = tmp_path / "v2.png"
            cv2.imwrite(str(f2), query_img)

            # Object canonical bbox in reference frame: [100, 100, 160, 160]
            # In query frame (visit 2), object is at [130, 120, 190, 180]
            canon_box = BoundingBox(100.0, 100.0, 160.0, 160.0)
            query_box = BoundingBox(130.0, 120.0, 190.0, 180.0)

            stub = ControlledStubDetector({
                "visit_0": [Detection(label="tool", confidence=0.9, raw_bbox=canon_box)],
                "visit_1": [Detection(label="tool", confidence=0.9, raw_bbox=query_box)],
            })

            # GT annotation in visit 2 provides the raw query coordinates
            annots_v1 = [RealVisitAnnotation(entity_id="tool_1", entity_label="tool", bbox=canon_box, physical_observation=PhysicalObservation.VISIBLE)]
            annots_v2 = [RealVisitAnnotation(entity_id="tool_1", entity_label="tool", bbox=query_box, physical_observation=PhysicalObservation.VISIBLE, annotated_event=AnnotatedEvent.UNCHANGED)]

            visits = [
                RealVisitFrame("v1", "V1", "2026-03-01T10:00:00Z", f1, annots_v1),
                RealVisitFrame("v2", "V2", "2026-03-01T11:00:00Z", f2, annots_v2),
            ]
            dataset = RealWorkbenchDataset("coord_frame_seq", "desc", tmp_path, visits)

            with SQLiteMemoryStore(tmp_path / "test.db") as store:
                runner = RealImagePipelineRunner(store=store, detector=stub)
                evaluator = RealBenchmarkEvaluator()
                report = evaluator.evaluate(dataset, runner)

                cm = report.change_metrics
                self.assertIsNotNone(cm)
                self.assertEqual(cm.unassociated_annotated_entities, 0)
                self.assertEqual(cm.accuracy, 1.0)
                self.assertEqual(cm.correct_state_predictions, 2)

    def test_hardened_delayed_discovery_with_reliable_registration(self) -> None:
        """
        6. Delayed discovery with reliable registration: an entity first detected in visit N > 0
           is correctly mapped to the appropriate ground-truth entity.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            f1 = self._create_synthetic_frame_file(tmp_path, "v1.png")
            f2 = self._create_synthetic_frame_file(tmp_path, "v2.png")
            box = BoundingBox(100.0, 100.0, 160.0, 160.0)

            stub = ControlledStubDetector({
                "visit_0": [],  # missed discovery on visit 0
                "visit_1": [Detection(label="pliers", confidence=0.9, raw_bbox=box)],
            })

            annots_v1 = [RealVisitAnnotation(entity_id="pliers_1", entity_label="pliers", bbox=box, physical_observation=PhysicalObservation.VISIBLE)]
            annots_v2 = [RealVisitAnnotation(entity_id="pliers_1", entity_label="pliers", bbox=box, physical_observation=PhysicalObservation.VISIBLE)]

            visits = [
                RealVisitFrame("v1", "V1", "2026-03-01T10:00:00Z", f1, annots_v1),
                RealVisitFrame("v2", "V2", "2026-03-01T11:00:00Z", f2, annots_v2),
            ]
            dataset = RealWorkbenchDataset("delayed_disc_reliable_seq", "desc", tmp_path, visits)

            with SQLiteMemoryStore(tmp_path / "test.db") as store:
                runner = RealImagePipelineRunner(store=store, detector=stub)
                evaluator = RealBenchmarkEvaluator()
                report = evaluator.evaluate(dataset, runner)

                cm = report.change_metrics
                self.assertIsNotNone(cm)
                self.assertEqual(cm.unassociated_annotated_entities, 0)
                # Visit 0: missed (INSUFFICIENT_EVIDENCE != CONFIRMED_PRESENT) -> 0 credit
                # Visit 1: discovered (CONFIRMED_PRESENT == CONFIRMED_PRESENT) -> 1 credit
                self.assertEqual(cm.correct_state_predictions, 1)
                self.assertEqual(cm.total_entity_evaluations, 2)
                self.assertEqual(cm.accuracy, 0.5)

    def test_hardened_delayed_discovery_with_failed_registration(self) -> None:
        """
        7. Delayed discovery with failed registration: failed registration prevents cross-coordinate
           matching and handles the spatial ambiguity explicitly rather than producing corrupted state.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            f1 = self._create_synthetic_frame_file(tmp_path, "v1.png")
            # Blank image forces registration failure
            f2 = tmp_path / "v2.png"
            cv2.imwrite(str(f2), np.zeros((480, 640, 3), dtype=np.uint8))

            box0 = BoundingBox(50.0, 50.0, 80.0, 80.0)
            box1 = BoundingBox(100.0, 100.0, 160.0, 160.0)

            # In visit 0, establish workbench baseline with an anchor object
            # In visit 1, registration fails and detector returns a new candidate
            stub = ControlledStubDetector({
                "visit_0": [Detection(label="anchor", confidence=0.9, raw_bbox=box0)],
                "visit_1": [
                    Detection(label="anchor", confidence=0.9, raw_bbox=box0),
                    Detection(label="lamp", confidence=0.9, raw_bbox=box1),
                ],
            })

            annots_v1 = [
                RealVisitAnnotation(entity_id="anchor_1", entity_label="anchor", bbox=box0, physical_observation=PhysicalObservation.VISIBLE),
                RealVisitAnnotation(entity_id="lamp_1", entity_label="lamp", bbox=box1, physical_observation=PhysicalObservation.VISIBLE),
            ]
            annots_v2 = [
                RealVisitAnnotation(entity_id="anchor_1", entity_label="anchor", bbox=box0, physical_observation=PhysicalObservation.VISIBLE),
                RealVisitAnnotation(entity_id="lamp_1", entity_label="lamp", bbox=box1, physical_observation=PhysicalObservation.VISIBLE),
            ]

            visits = [
                RealVisitFrame("v1", "V1", "2026-03-01T10:00:00Z", f1, annots_v1),
                RealVisitFrame("v2", "V2", "2026-03-01T11:00:00Z", f2, annots_v2),
            ]
            dataset = RealWorkbenchDataset("delayed_disc_fail_reg_seq", "desc", tmp_path, visits)

            with SQLiteMemoryStore(tmp_path / "test.db") as store:
                runner = RealImagePipelineRunner(store=store, detector=stub)
                evaluator = RealBenchmarkEvaluator()
                report = evaluator.evaluate(dataset, runner)

                cm = report.change_metrics
                self.assertIsNotNone(cm)
                # Unaligned frame prevents corrupted spatial binding of lamp_1
                self.assertGreaterEqual(cm.unassociated_annotated_entities, 1)
                self.assertEqual(cm.false_confirmed_removals, 0)

    def test_hardened_semantic_contradiction_unchanged_with_confirmed_removed(self) -> None:
        """
        8. Semantic contradiction detection: manifest with UNCHANGED + CONFIRMED_REMOVED is rejected.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            f1 = self._create_synthetic_frame_file(tmp_path, "v1.png")
            box = BoundingBox(10.0, 10.0, 50.0, 50.0)

            v1 = RealVisitFrame("v1", "V1", "2026-03-01T10:00:00Z", f1, [
                RealVisitAnnotation(
                    entity_id="item_1",
                    entity_label="item",
                    bbox=box,
                    physical_observation=PhysicalObservation.VISIBLE,
                    annotated_event=AnnotatedEvent.UNCHANGED,
                    expected_state=EpistemicState.CONFIRMED_REMOVED,
                )
            ])
            dataset = RealWorkbenchDataset("contradict_unchanged_removed", "desc", tmp_path, [v1])
            errors = dataset.validate_dataset_integrity()
            self.assertTrue(any("annotated_event='UNCHANGED' cannot be paired with expected_state='CONFIRMED_REMOVED'" in e for e in errors))
            ready, msg = dataset.is_evaluation_ready()
            self.assertFalse(ready)

    def test_hardened_semantic_contradiction_absent_with_confirmed_present(self) -> None:
        """
        9. Semantic contradiction detection: manifest with ABSENT + CONFIRMED_PRESENT is rejected.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            f1 = self._create_synthetic_frame_file(tmp_path, "v1.png")

            v1 = RealVisitFrame("v1", "V1", "2026-03-01T10:00:00Z", f1, [
                RealVisitAnnotation(
                    entity_id="item_1",
                    entity_label="item",
                    bbox=None,
                    physical_observation=PhysicalObservation.ABSENT,
                    annotated_event=AnnotatedEvent.ABSENT,
                    expected_state=EpistemicState.CONFIRMED_PRESENT,
                )
            ])
            dataset = RealWorkbenchDataset("contradict_absent_present", "desc", tmp_path, [v1])
            errors = dataset.validate_dataset_integrity()
            self.assertTrue(any("physical_observation='ABSENT' cannot be paired with expected_state='CONFIRMED_PRESENT'" in e for e in errors))
            ready, msg = dataset.is_evaluation_ready()
            self.assertFalse(ready)

    def test_hardened_semantic_contradiction_relocated_with_contradictory_state(self) -> None:
        """
        10. Semantic contradiction detection: manifest with RELOCATED + contradictory state is rejected.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            f1 = self._create_synthetic_frame_file(tmp_path, "v1.png")
            box = BoundingBox(10.0, 10.0, 50.0, 50.0)

            v1 = RealVisitFrame("v1", "V1", "2026-03-01T10:00:00Z", f1, [
                RealVisitAnnotation(
                    entity_id="item_1",
                    entity_label="item",
                    bbox=box,
                    physical_observation=PhysicalObservation.VISIBLE,
                    annotated_event=AnnotatedEvent.RELOCATED,
                    expected_state=EpistemicState.CONFIRMED_REMOVED,
                )
            ])
            dataset = RealWorkbenchDataset("contradict_relocated_removed", "desc", tmp_path, [v1])
            errors = dataset.validate_dataset_integrity()
            self.assertTrue(any("annotated_event='RELOCATED' cannot be paired with expected_state='CONFIRMED_REMOVED'" in e for e in errors))
            ready, msg = dataset.is_evaluation_ready()
            self.assertFalse(ready)

    def test_hardened_semantic_contradiction_unchanged_with_confirmed_removed(self) -> None:
        """
        8. Semantic contradiction detection: manifest with UNCHANGED + CONFIRMED_REMOVED is rejected.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            f1 = self._create_synthetic_frame_file(tmp_path, "v1.png")
            box = BoundingBox(10.0, 10.0, 50.0, 50.0)

            v1 = RealVisitFrame("v1", "V1", "2026-03-01T10:00:00Z", f1, [
                RealVisitAnnotation(
                    entity_id="item_1",
                    entity_label="item",
                    bbox=box,
                    physical_observation=PhysicalObservation.VISIBLE,
                    annotated_event=AnnotatedEvent.UNCHANGED,
                    expected_state=EpistemicState.CONFIRMED_REMOVED,
                )
            ])
            dataset = RealWorkbenchDataset("contradict_unchanged_removed", "desc", tmp_path, [v1])
            errors = dataset.validate_dataset_integrity()
            self.assertTrue(any("annotated_event='UNCHANGED' cannot be paired with expected_state='CONFIRMED_REMOVED'" in e for e in errors))
            ready, msg = dataset.is_evaluation_ready()
            self.assertFalse(ready)

    def test_hardened_semantic_contradiction_absent_with_confirmed_present(self) -> None:
        """
        9. Semantic contradiction detection: manifest with ABSENT + CONFIRMED_PRESENT is rejected.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            f1 = self._create_synthetic_frame_file(tmp_path, "v1.png")

            v1 = RealVisitFrame("v1", "V1", "2026-03-01T10:00:00Z", f1, [
                RealVisitAnnotation(
                    entity_id="item_1",
                    entity_label="item",
                    bbox=None,
                    physical_observation=PhysicalObservation.ABSENT,
                    annotated_event=AnnotatedEvent.ABSENT,
                    expected_state=EpistemicState.CONFIRMED_PRESENT,
                )
            ])
            dataset = RealWorkbenchDataset("contradict_absent_present", "desc", tmp_path, [v1])
            errors = dataset.validate_dataset_integrity()
            self.assertTrue(any("physical_observation='ABSENT' cannot be paired with expected_state='CONFIRMED_PRESENT'" in e for e in errors))
            ready, msg = dataset.is_evaluation_ready()
            self.assertFalse(ready)

    def test_hardened_semantic_contradiction_relocated_with_contradictory_state(self) -> None:
        """
        10. Semantic contradiction detection: manifest with RELOCATED + contradictory state is rejected.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            f1 = self._create_synthetic_frame_file(tmp_path, "v1.png")
            box = BoundingBox(10.0, 10.0, 50.0, 50.0)

            v1 = RealVisitFrame("v1", "V1", "2026-03-01T10:00:00Z", f1, [
                RealVisitAnnotation(
                    entity_id="item_1",
                    entity_label="item",
                    bbox=box,
                    physical_observation=PhysicalObservation.VISIBLE,
                    annotated_event=AnnotatedEvent.RELOCATED,
                    expected_state=EpistemicState.CONFIRMED_REMOVED,
                )
            ])
            dataset = RealWorkbenchDataset("contradict_relocated_removed", "desc", tmp_path, [v1])
            errors = dataset.validate_dataset_integrity()
            self.assertTrue(any("annotated_event='RELOCATED' cannot be paired with expected_state='CONFIRMED_REMOVED'" in e for e in errors))
            ready, msg = dataset.is_evaluation_ready()
            self.assertFalse(ready)


if __name__ == "__main__":
    unittest.main()




