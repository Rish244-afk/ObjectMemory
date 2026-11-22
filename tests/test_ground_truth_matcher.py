"""
Unit tests for GroundTruthMatcher and evaluation event scoring (Phase 9B).
Tests:
- Multi-object matching with shared class labels (two cups, two books).
- Return-to-baseline event scoring.
- Temporal sequence: baseline -> moved -> baseline -> unchanged.
- Reporting of unmatched GT and unmatched system tracks.
"""

from __future__ import annotations

import sys
from pathlib import Path
import unittest
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from objectmemory.benchmarks.ground_truth_matcher import (
    GroundTruthMatcher,
    SequenceEvaluationReport,
)
from objectmemory.datasets.real_dataset import (
    AnnotatedEvent,
    PhysicalObservation,
    RealVisitAnnotation,
    RealVisitFrame,
    RealWorkbenchDataset,
)
from objectmemory.domain.models import BoundingBox, WorkbenchEntity, ChangeEvent
from objectmemory.domain.states import EpistemicState
from objectmemory.temporal.models import TemporalEngineConfig
from objectmemory.temporal.engine import TemporalMemoryEngine
from objectmemory.persistence.repository import SQLiteMemoryStore


class TestGroundTruthMatcher(unittest.TestCase):
    def setUp(self) -> None:
        self.matcher = GroundTruthMatcher(min_match_iou=0.10)

    def test_multi_object_shared_label_matching(self) -> None:
        """Verifies two cups and two books in the same scene map to distinct system entities."""
        # GT annotations
        gt_annots = [
            RealVisitAnnotation(
                entity_id="cup_left",
                entity_label="cup",
                bbox=BoundingBox(10.0, 100.0, 110.0, 200.0),
                physical_observation=PhysicalObservation.VISIBLE,
            ),
            RealVisitAnnotation(
                entity_id="cup_right",
                entity_label="cup",
                bbox=BoundingBox(300.0, 100.0, 400.0, 200.0),
                physical_observation=PhysicalObservation.VISIBLE,
            ),
            RealVisitAnnotation(
                entity_id="book_top",
                entity_label="book",
                bbox=BoundingBox(50.0, 10.0, 250.0, 80.0),
                physical_observation=PhysicalObservation.VISIBLE,
            ),
            RealVisitAnnotation(
                entity_id="book_bottom",
                entity_label="book",
                bbox=BoundingBox(50.0, 300.0, 250.0, 400.0),
                physical_observation=PhysicalObservation.VISIBLE,
            ),
        ]

        # System entities (with slight detection jitter)
        sys_entities = [
            WorkbenchEntity(
                entity_id="sys_cup_2",
                label="cup",
                canonical_bbox=BoundingBox(295.0, 95.0, 405.0, 205.0),
                current_state=EpistemicState.CONFIRMED_PRESENT,
                created_at="2026-10-21T00:00:00Z",
            ),
            WorkbenchEntity(
                entity_id="sys_cup_1",
                label="cup",
                canonical_bbox=BoundingBox(12.0, 98.0, 108.0, 198.0),
                current_state=EpistemicState.CONFIRMED_PRESENT,
                created_at="2026-10-21T00:00:00Z",
            ),
            WorkbenchEntity(
                entity_id="sys_book_bot",
                label="book",
                canonical_bbox=BoundingBox(48.0, 298.0, 252.0, 402.0),
                current_state=EpistemicState.CONFIRMED_PRESENT,
                created_at="2026-10-21T00:00:00Z",
            ),
            WorkbenchEntity(
                entity_id="sys_book_top",
                label="book",
                canonical_bbox=BoundingBox(52.0, 12.0, 248.0, 82.0),
                current_state=EpistemicState.CONFIRMED_PRESENT,
                created_at="2026-10-21T00:00:00Z",
            ),
            # An extra duplicate/untracked entity
            WorkbenchEntity(
                entity_id="sys_duplicate_cup",
                label="cup",
                canonical_bbox=BoundingBox(600.0, 400.0, 700.0, 500.0),
                current_state=EpistemicState.CONFIRMED_PRESENT,
                created_at="2026-10-21T00:00:00Z",
            ),
        ]

        mapping = self.matcher.compute_bipartite_entity_mapping(gt_annots, sys_entities)

        # Each GT entity must map to its spatially closest system entity
        self.assertEqual(mapping["cup_left"], "sys_cup_1")
        self.assertEqual(mapping["cup_right"], "sys_cup_2")
        self.assertEqual(mapping["book_top"], "sys_book_top")
        self.assertEqual(mapping["book_bottom"], "sys_book_bot")

        # Confirm 1-to-1 mapping (no two GTs map to same sys_id)
        mapped_sys_ids = list(mapping.values())
        self.assertEqual(len(mapped_sys_ids), len(set(mapped_sys_ids)))

    def test_unmatched_gt_and_duplicate_tracking(self) -> None:
        """Verifies unmatched GT entities and duplicate tracks are accurately identified."""
        gt_annots = [
            RealVisitAnnotation(
                entity_id="cup_01",
                entity_label="cup",
                bbox=BoundingBox(10.0, 10.0, 50.0, 50.0),
                physical_observation=PhysicalObservation.VISIBLE,
            ),
            RealVisitAnnotation(
                entity_id="cup_missed",
                entity_label="cup",
                bbox=BoundingBox(200.0, 200.0, 250.0, 250.0),
                physical_observation=PhysicalObservation.VISIBLE,
            ),
        ]

        sys_entities = [
            WorkbenchEntity(
                entity_id="sys_cup_01",
                label="cup",
                canonical_bbox=BoundingBox(12.0, 12.0, 48.0, 48.0),
                current_state=EpistemicState.CONFIRMED_PRESENT,
                created_at="2026-10-21T00:00:00Z",
            ),
            WorkbenchEntity(
                entity_id="sys_spurious_bottle",
                label="bottle",
                canonical_bbox=BoundingBox(500.0, 10.0, 550.0, 90.0),
                current_state=EpistemicState.CONFIRMED_PRESENT,
                created_at="2026-10-21T00:00:00Z",
            ),
        ]

        from pathlib import Path
        dataset = RealWorkbenchDataset(
            root_dir=Path("."),
            dataset_name="test_dataset",
            description="test",
            visits=[RealVisitFrame(visit_id="visit_01", session_name="Visit 01", timestamp="2026-10-21T00:00:00Z", image_path="", annotations=gt_annots)],
        )

        store = SQLiteMemoryStore(":memory:")
        report = self.matcher.evaluate_sequence(dataset, sys_entities, [None], store)

        self.assertIn("cup_missed", report.unmatched_gt_entities)
        self.assertIn("sys_spurious_bottle", report.unmatched_system_entities)
        self.assertEqual(report.duplicate_track_count, 1)

    def test_baseline_moved_baseline_unchanged_lifecycle(self) -> None:
        """
        Verifies complete lifecycle:
        Visit 1: Baseline (CONFIRMED_PRESENT)
        Visit 2: Relocated (CONFIRMED_MOVED)
        Visit 3: Return to baseline (CONFIRMED_PRESENT with is_return_to_baseline=True)
        Visit 4: Stationary at baseline (CONFIRMED_PRESENT, UNCHANGED)
        """
        store = SQLiteMemoryStore(":memory:")
        config = TemporalEngineConfig(movement_displacement_threshold=35.0)
        engine = TemporalMemoryEngine(store=store, config=config)

        # Entity at initial position (100, 100)
        box_v1 = BoundingBox(50.0, 50.0, 150.0, 150.0) # centroid (100, 100)
        # Moved to (500, 500)
        box_v2 = BoundingBox(450.0, 450.0, 550.0, 550.0) # centroid (500, 500)
        # Returned to (101, 101)
        box_v3 = BoundingBox(51.0, 51.0, 151.0, 151.0) # centroid (101, 101)
        # Stationary at (101, 101)
        box_v4 = BoundingBox(51.0, 51.0, 151.0, 151.0) # centroid (101, 101)

        ent = WorkbenchEntity(
            entity_id="test_cup",
            label="cup",
            current_state=EpistemicState.CONFIRMED_PRESENT,
            canonical_bbox=box_v1,
            created_at="2026-10-21T00:00:00Z",
            metadata={"initial_canonical_bbox": [50.0, 50.0, 150.0, 150.0]},
        )
        store.add_entity(ent)

        from objectmemory.registration.models import RegistrationResult, AlignmentStatus
        from objectmemory.detection.models import Detection
        from objectmemory.temporal.models import VisitInput

        import numpy as np

        def make_visit(visit_id: str, ts: str) -> VisitInput:
            reg = RegistrationResult(
                success=True,
                status=AlignmentStatus.SUCCESS,
                method_used="TEST",
                ref_size=(800, 600),
                query_size=(800, 600),
                homography=np.eye(3, dtype=np.float64),
                inlier_ratio=0.95,
                num_inliers=300,
                overlap_mask=np.ones((600, 800), dtype=np.uint8) * 255,
            )
            dummy_img = np.zeros((600, 800, 3), dtype=np.uint8)
            return VisitInput(
                session_id=visit_id,
                session_name=visit_id,
                timestamp=ts,
                image=dummy_img,
                registration=reg,
                detections=[],
            )

        # Step 1: Visit 2 - Move to (500, 500)
        det_v2 = Detection(
            label="cup",
            confidence=0.90,
            raw_bbox=box_v2,
            canonical_bbox=box_v2,
        )
        v2_input = make_visit("v2", "2026-10-21T01:00:00Z")
        c_disp, b_disp, is_c_mov, is_ret, is_rel, target_st, expl = engine._evaluate_displacement(
            entity=ent,
            det=det_v2,
            visit=v2_input,
            initial_bbox=box_v1,
            last_reliable_bbox=box_v1,
            det_canon=box_v2,
        )
        self.assertTrue(is_c_mov)
        self.assertFalse(is_ret)
        self.assertEqual(target_st, EpistemicState.CONFIRMED_MOVED)
        self.assertAlmostEqual(b_disp, 565.685, delta=1.0)

        # Update entity state to CONFIRMED_MOVED
        ent.current_state = EpistemicState.CONFIRMED_MOVED
        ent.canonical_bbox = box_v2
        ent.metadata["last_reliable_canonical_bbox"] = [450.0, 450.0, 550.0, 550.0]

        # Step 2: Visit 3 - Return to baseline (101, 101)
        det_v3 = Detection(
            label="cup",
            confidence=0.90,
            raw_bbox=box_v3,
            canonical_bbox=box_v3,
        )
        v3_input = make_visit("v3", "2026-10-21T02:00:00Z")
        c_disp, b_disp, is_c_mov, is_ret, is_rel, target_st, expl = engine._evaluate_displacement(
            entity=ent,
            det=det_v3,
            visit=v3_input,
            initial_bbox=box_v1,
            last_reliable_bbox=box_v2,
            det_canon=box_v3,
        )
        self.assertTrue(is_c_mov) # Moved 564px relative to previous observation
        self.assertTrue(is_ret) # Returned to baseline!
        self.assertEqual(target_st, EpistemicState.CONFIRMED_PRESENT)
        self.assertAlmostEqual(b_disp, 1.414, delta=0.5)

        # Update entity state to CONFIRMED_PRESENT at baseline
        ent.current_state = EpistemicState.CONFIRMED_PRESENT
        ent.canonical_bbox = box_v3
        ent.metadata["last_reliable_canonical_bbox"] = [51.0, 51.0, 151.0, 151.0]

        # Step 3: Visit 4 - Stationary at baseline
        det_v4 = Detection(
            label="cup",
            confidence=0.90,
            raw_bbox=box_v4,
            canonical_bbox=box_v4,
        )
        v4_input = make_visit("v4", "2026-10-21T03:00:00Z")
        c_disp, b_disp, is_c_mov, is_ret, is_rel, target_st, expl = engine._evaluate_displacement(
            entity=ent,
            det=det_v4,
            visit=v4_input,
            initial_bbox=box_v1,
            last_reliable_bbox=box_v3,
            det_canon=box_v4,
        )
        self.assertFalse(is_c_mov) # Stationary
        self.assertFalse(is_ret)
        self.assertEqual(target_st, EpistemicState.CONFIRMED_PRESENT)
        self.assertAlmostEqual(b_disp, 1.414, delta=0.5)

    def test_historical_state_v2_moved_v4_returned_to_baseline(self) -> None:
        """
        Regression Test A:
        An entity moves at V2 and returns to baseline at V4.
        Verifies that V2 is evaluated against V2's state (CONFIRMED_MOVED)
        even though terminal state after V4 is CONFIRMED_PRESENT.
        """
        from pathlib import Path
        from objectmemory.temporal.models import EntityBeliefUpdate, VisitProcessingReport

        gt_v1 = [RealVisitAnnotation(entity_id="cup_1", entity_label="cup", bbox=BoundingBox(50, 50, 150, 150), physical_observation=PhysicalObservation.VISIBLE, annotated_event=AnnotatedEvent.UNCHANGED, expected_state=EpistemicState.CONFIRMED_PRESENT)]
        gt_v2 = [RealVisitAnnotation(entity_id="cup_1", entity_label="cup", bbox=BoundingBox(450, 450, 550, 550), physical_observation=PhysicalObservation.VISIBLE, annotated_event=AnnotatedEvent.RELOCATED, expected_state=EpistemicState.CONFIRMED_MOVED)]
        gt_v3 = [RealVisitAnnotation(entity_id="cup_1", entity_label="cup", bbox=BoundingBox(450, 450, 550, 550), physical_observation=PhysicalObservation.VISIBLE, annotated_event=AnnotatedEvent.UNCHANGED, expected_state=EpistemicState.CONFIRMED_MOVED)]
        gt_v4 = [RealVisitAnnotation(entity_id="cup_1", entity_label="cup", bbox=BoundingBox(51, 51, 151, 151), physical_observation=PhysicalObservation.VISIBLE, annotated_event=AnnotatedEvent.RELOCATED, expected_state=EpistemicState.CONFIRMED_PRESENT, notes="return to baseline")]

        dataset = RealWorkbenchDataset(
            root_dir=Path("."),
            dataset_name="test_v2_v4",
            description="test",
            visits=[
                RealVisitFrame("v1", "Visit 1", "2026-10-21T01:00:00Z", "", gt_v1),
                RealVisitFrame("v2", "Visit 2", "2026-10-21T02:00:00Z", "", gt_v2),
                RealVisitFrame("v3", "Visit 3", "2026-10-21T03:00:00Z", "", gt_v3),
                RealVisitFrame("v4", "Visit 4", "2026-10-21T04:00:00Z", "", gt_v4),
            ],
        )

        sys_id = "sys_cup_1"
        # Notice: terminal state is CONFIRMED_PRESENT because it returned at V4!
        sys_entities = [
            WorkbenchEntity(
                entity_id=sys_id,
                label="cup",
                canonical_bbox=BoundingBox(51, 51, 151, 151),
                current_state=EpistemicState.CONFIRMED_PRESENT,
                created_at="2026-10-21T01:00:00Z",
            )
        ]

        # Simulate VisitProcessingReports for each visit
        rep_v1 = VisitProcessingReport("v1", "v1", "2026-10-21T01:00:00Z", "SUCCESS", 1, 1, 1, 0, [
            EntityBeliefUpdate(sys_id, "cup", EpistemicState.INSUFFICIENT_EVIDENCE, EpistemicState.CONFIRMED_PRESENT, False)
        ])
        rep_v2 = VisitProcessingReport("v2", "v2", "2026-10-21T02:00:00Z", "SUCCESS", 1, 1, 0, 1, [
            EntityBeliefUpdate(sys_id, "cup", EpistemicState.CONFIRMED_PRESENT, EpistemicState.CONFIRMED_MOVED, True, supporting_evidence={"consecutive_displacement": 565.7})
        ])
        rep_v3 = VisitProcessingReport("v3", "v3", "2026-10-21T03:00:00Z", "SUCCESS", 1, 1, 0, 0, [
            EntityBeliefUpdate(sys_id, "cup", EpistemicState.CONFIRMED_MOVED, EpistemicState.CONFIRMED_MOVED, False)
        ])
        rep_v4 = VisitProcessingReport("v4", "v4", "2026-10-21T04:00:00Z", "SUCCESS", 1, 1, 0, 1, [
            EntityBeliefUpdate(sys_id, "cup", EpistemicState.CONFIRMED_MOVED, EpistemicState.CONFIRMED_PRESENT, True, supporting_evidence={"is_return_to_baseline": True, "baseline_displacement": 1.4, "consecutive_displacement": 564.0})
        ])

        report = self.matcher.evaluate_sequence(dataset, sys_entities, [rep_v1, rep_v2, rep_v3, rep_v4])

        # Assert outcomes
        self.assertEqual(report.total_evaluations, 4)
        self.assertEqual(report.matched_evaluations, 4)
        self.assertEqual(report.overall_state_accuracy, 1.0)
        self.assertEqual(report.missed_movements, 0)
        self.assertEqual(report.false_movements, 0)
        self.assertEqual(report.correct_relocations, 2)  # V2 movement + V4 return
        self.assertEqual(report.expected_returns_to_baseline, 1)
        self.assertEqual(report.detected_returns_to_baseline, 1)

        # Inspect per-visit evaluation records
        self.assertEqual(report.evaluations[0].observed_state, EpistemicState.CONFIRMED_PRESENT)
        self.assertEqual(report.evaluations[1].observed_state, EpistemicState.CONFIRMED_MOVED)
        self.assertEqual(report.evaluations[2].observed_state, EpistemicState.CONFIRMED_MOVED)
        self.assertEqual(report.evaluations[3].observed_state, EpistemicState.CONFIRMED_PRESENT)
        self.assertTrue(report.evaluations[3].is_return_to_baseline_observed)

    def test_stationary_entity_throughout_sequence(self) -> None:
        """
        Regression Test B:
        An entity remains stationary throughout the sequence.
        No movement should be reported.
        """
        from pathlib import Path
        from objectmemory.temporal.models import EntityBeliefUpdate, VisitProcessingReport

        gt_box = BoundingBox(100, 100, 200, 200)
        gt_v1 = [RealVisitAnnotation(entity_id="bot_1", entity_label="bottle", bbox=gt_box, physical_observation=PhysicalObservation.VISIBLE, annotated_event=AnnotatedEvent.UNCHANGED, expected_state=EpistemicState.CONFIRMED_PRESENT)]
        gt_v2 = [RealVisitAnnotation(entity_id="bot_1", entity_label="bottle", bbox=gt_box, physical_observation=PhysicalObservation.VISIBLE, annotated_event=AnnotatedEvent.UNCHANGED, expected_state=EpistemicState.CONFIRMED_PRESENT)]
        gt_v3 = [RealVisitAnnotation(entity_id="bot_1", entity_label="bottle", bbox=gt_box, physical_observation=PhysicalObservation.VISIBLE, annotated_event=AnnotatedEvent.UNCHANGED, expected_state=EpistemicState.CONFIRMED_PRESENT)]

        dataset = RealWorkbenchDataset(
            root_dir=Path("."),
            dataset_name="test_stationary",
            description="test",
            visits=[
                RealVisitFrame("v1", "Visit 1", "2026-10-21T01:00:00Z", "", gt_v1),
                RealVisitFrame("v2", "Visit 2", "2026-10-21T02:00:00Z", "", gt_v2),
                RealVisitFrame("v3", "Visit 3", "2026-10-21T03:00:00Z", "", gt_v3),
            ],
        )

        sys_id = "sys_bot_1"
        sys_entities = [
            WorkbenchEntity(sys_id, "bottle", "2026-10-21T01:00:00Z", gt_box, EpistemicState.CONFIRMED_PRESENT)
        ]
        reports = [
            VisitProcessingReport("v1", "v1", "2026-10-21T01:00:00Z", "SUCCESS", 1, 1, 1, 0, [EntityBeliefUpdate(sys_id, "bottle", EpistemicState.INSUFFICIENT_EVIDENCE, EpistemicState.CONFIRMED_PRESENT, False)]),
            VisitProcessingReport("v2", "v2", "2026-10-21T02:00:00Z", "SUCCESS", 1, 1, 0, 0, [EntityBeliefUpdate(sys_id, "bottle", EpistemicState.CONFIRMED_PRESENT, EpistemicState.CONFIRMED_PRESENT, False)]),
            VisitProcessingReport("v3", "v3", "2026-10-21T03:00:00Z", "SUCCESS", 1, 1, 0, 0, [EntityBeliefUpdate(sys_id, "bottle", EpistemicState.CONFIRMED_PRESENT, EpistemicState.CONFIRMED_PRESENT, False)]),
        ]

        report = self.matcher.evaluate_sequence(dataset, sys_entities, reports)
        self.assertEqual(report.total_evaluations, 3)
        self.assertEqual(report.matched_evaluations, 3)
        self.assertEqual(report.false_movements, 0)
        self.assertEqual(report.false_movement_rate, 0.0)
        self.assertEqual(report.correct_relocations, 0)
        self.assertEqual(report.missed_movements, 0)

    def test_persistent_movement_evaluation(self) -> None:
        """
        Regression Test C:
        An entity moves and stays at its new position.
        The persistent movement should be evaluated correctly without duplicate movement counts.
        """
        from pathlib import Path
        from objectmemory.temporal.models import EntityBeliefUpdate, VisitProcessingReport

        gt_box1 = BoundingBox(10, 10, 50, 50)
        gt_box2 = BoundingBox(300, 300, 340, 340)
        gt_v1 = [RealVisitAnnotation(entity_id="box_1", entity_label="box", bbox=gt_box1, physical_observation=PhysicalObservation.VISIBLE, annotated_event=AnnotatedEvent.UNCHANGED, expected_state=EpistemicState.CONFIRMED_PRESENT)]
        gt_v2 = [RealVisitAnnotation(entity_id="box_1", entity_label="box", bbox=gt_box2, physical_observation=PhysicalObservation.VISIBLE, annotated_event=AnnotatedEvent.RELOCATED, expected_state=EpistemicState.CONFIRMED_MOVED)]
        gt_v3 = [RealVisitAnnotation(entity_id="box_1", entity_label="box", bbox=gt_box2, physical_observation=PhysicalObservation.VISIBLE, annotated_event=AnnotatedEvent.UNCHANGED, expected_state=EpistemicState.CONFIRMED_MOVED)]

        dataset = RealWorkbenchDataset(
            root_dir=Path("."),
            dataset_name="test_persistent_move",
            description="test",
            visits=[
                RealVisitFrame("v1", "Visit 1", "2026-10-21T01:00:00Z", "", gt_v1),
                RealVisitFrame("v2", "Visit 2", "2026-10-21T02:00:00Z", "", gt_v2),
                RealVisitFrame("v3", "Visit 3", "2026-10-21T03:00:00Z", "", gt_v3),
            ],
        )

        sys_id = "sys_box_1"
        sys_entities = [
            WorkbenchEntity(sys_id, "box", "2026-10-21T01:00:00Z", gt_box2, EpistemicState.CONFIRMED_MOVED)
        ]
        reports = [
            VisitProcessingReport("v1", "v1", "2026-10-21T01:00:00Z", "SUCCESS", 1, 1, 1, 0, [EntityBeliefUpdate(sys_id, "box", EpistemicState.INSUFFICIENT_EVIDENCE, EpistemicState.CONFIRMED_PRESENT, False)]),
            VisitProcessingReport("v2", "v2", "2026-10-21T02:00:00Z", "SUCCESS", 1, 1, 0, 1, [EntityBeliefUpdate(sys_id, "box", EpistemicState.CONFIRMED_PRESENT, EpistemicState.CONFIRMED_MOVED, True)]),
            VisitProcessingReport("v3", "v3", "2026-10-21T03:00:00Z", "SUCCESS", 1, 1, 0, 0, [EntityBeliefUpdate(sys_id, "box", EpistemicState.CONFIRMED_MOVED, EpistemicState.CONFIRMED_MOVED, False)]),
        ]

        report = self.matcher.evaluate_sequence(dataset, sys_entities, reports)
        self.assertEqual(report.total_evaluations, 3)
        self.assertEqual(report.matched_evaluations, 3)
        self.assertEqual(report.correct_relocations, 1)
        self.assertEqual(report.missed_movements, 0)
        self.assertEqual(report.false_movements, 0)

    def test_missing_historical_records_explicit_safeguard(self) -> None:
        """
        Regression Test D:
        When historical records are missing, evaluator must return INSUFFICIENT_EVIDENCE
        with an explicit explanation and NOT silently use current_state.
        """
        from pathlib import Path

        gt_v1 = [RealVisitAnnotation(entity_id="item_1", entity_label="item", bbox=BoundingBox(10, 10, 50, 50), physical_observation=PhysicalObservation.VISIBLE, annotated_event=AnnotatedEvent.UNCHANGED, expected_state=EpistemicState.CONFIRMED_PRESENT)]
        dataset = RealWorkbenchDataset(
            root_dir=Path("."),
            dataset_name="test_missing_hist",
            description="test",
            visits=[RealVisitFrame("v1", "Visit 1", "2026-10-21T01:00:00Z", "", gt_v1)],
        )

        sys_id = "sys_item_1"
        # Entity has current_state=CONFIRMED_MOVED in memory, but no visit reports or store records exist!
        sys_entities = [
            WorkbenchEntity(sys_id, "item", "2026-10-21T01:00:00Z", BoundingBox(10, 10, 50, 50), EpistemicState.CONFIRMED_MOVED)
        ]

        # Pass empty reports and empty store
        report = self.matcher.evaluate_sequence(dataset, sys_entities, [None], store=SQLiteMemoryStore(":memory:"))
        self.assertEqual(len(report.evaluations), 1)
        rec = report.evaluations[0]
        # Must NOT silently use CONFIRMED_MOVED from sys_ent.current_state!
        self.assertEqual(rec.observed_state, EpistemicState.INSUFFICIENT_EVIDENCE)
        self.assertFalse(rec.is_state_match)
        self.assertIn("Historical state record missing", rec.explanation)

    def test_multiple_same_class_entities_identity_preservation(self) -> None:
        """
        Regression Test E:
        Multiple same-class entities appear in the same sequence.
        Historical evaluation must preserve entity identity.
        """
        from pathlib import Path
        from objectmemory.temporal.models import EntityBeliefUpdate, VisitProcessingReport

        box_left = BoundingBox(10, 10, 50, 50)
        box_right = BoundingBox(300, 10, 340, 50)
        box_left_moved = BoundingBox(10, 200, 50, 240)

        gt_v1 = [
            RealVisitAnnotation(entity_id="cup_left", entity_label="cup", bbox=box_left, physical_observation=PhysicalObservation.VISIBLE, annotated_event=AnnotatedEvent.UNCHANGED, expected_state=EpistemicState.CONFIRMED_PRESENT),
            RealVisitAnnotation(entity_id="cup_right", entity_label="cup", bbox=box_right, physical_observation=PhysicalObservation.VISIBLE, annotated_event=AnnotatedEvent.UNCHANGED, expected_state=EpistemicState.CONFIRMED_PRESENT),
        ]
        gt_v2 = [
            RealVisitAnnotation(entity_id="cup_left", entity_label="cup", bbox=box_left_moved, physical_observation=PhysicalObservation.VISIBLE, annotated_event=AnnotatedEvent.RELOCATED, expected_state=EpistemicState.CONFIRMED_MOVED),
            RealVisitAnnotation(entity_id="cup_right", entity_label="cup", bbox=box_right, physical_observation=PhysicalObservation.VISIBLE, annotated_event=AnnotatedEvent.UNCHANGED, expected_state=EpistemicState.CONFIRMED_PRESENT),
        ]

        dataset = RealWorkbenchDataset(
            root_dir=Path("."),
            dataset_name="test_multi_same_class",
            description="test",
            visits=[
                RealVisitFrame("v1", "Visit 1", "2026-10-21T01:00:00Z", "", gt_v1),
                RealVisitFrame("v2", "Visit 2", "2026-10-21T02:00:00Z", "", gt_v2),
            ],
        )

        sys_entities = [
            WorkbenchEntity("sys_left", "cup", "2026-10-21T01:00:00Z", box_left_moved, EpistemicState.CONFIRMED_MOVED),
            WorkbenchEntity("sys_right", "cup", "2026-10-21T01:00:00Z", box_right, EpistemicState.CONFIRMED_PRESENT),
        ]
        reports = [
            VisitProcessingReport("v1", "v1", "2026-10-21T01:00:00Z", "SUCCESS", 2, 2, 2, 0, [
                EntityBeliefUpdate("sys_left", "cup", EpistemicState.INSUFFICIENT_EVIDENCE, EpistemicState.CONFIRMED_PRESENT, False),
                EntityBeliefUpdate("sys_right", "cup", EpistemicState.INSUFFICIENT_EVIDENCE, EpistemicState.CONFIRMED_PRESENT, False),
            ]),
            VisitProcessingReport("v2", "v2", "2026-10-21T02:00:00Z", "SUCCESS", 2, 2, 0, 1, [
                EntityBeliefUpdate("sys_left", "cup", EpistemicState.CONFIRMED_PRESENT, EpistemicState.CONFIRMED_MOVED, True),
                EntityBeliefUpdate("sys_right", "cup", EpistemicState.CONFIRMED_PRESENT, EpistemicState.CONFIRMED_PRESENT, False),
            ]),
        ]

        report = self.matcher.evaluate_sequence(dataset, sys_entities, reports)
        self.assertEqual(report.entity_mappings["cup_left"], "sys_left")
        self.assertEqual(report.entity_mappings["cup_right"], "sys_right")
        self.assertEqual(report.matched_evaluations, 4)
        self.assertEqual(report.correct_relocations, 1)  # Only cup_left moved
        self.assertEqual(report.false_movements, 0)      # cup_right did not move

    def test_event_ordering_preserved_across_visits(self) -> None:
        """
        Regression Test F:
        Event ordering is preserved chronologically across visits.
        """
        from pathlib import Path
        from objectmemory.temporal.models import EntityBeliefUpdate, VisitProcessingReport

        gt_box = BoundingBox(10, 10, 50, 50)
        gt_v1 = [RealVisitAnnotation(entity_id="item_1", entity_label="item", bbox=gt_box, physical_observation=PhysicalObservation.VISIBLE, annotated_event=AnnotatedEvent.UNCHANGED, expected_state=EpistemicState.CONFIRMED_PRESENT)]
        gt_v2 = [RealVisitAnnotation(entity_id="item_1", entity_label="item", bbox=gt_box, physical_observation=PhysicalObservation.VISIBLE, annotated_event=AnnotatedEvent.UNCHANGED, expected_state=EpistemicState.CONFIRMED_PRESENT)]
        gt_v3 = [RealVisitAnnotation(entity_id="item_1", entity_label="item", bbox=gt_box, physical_observation=PhysicalObservation.VISIBLE, annotated_event=AnnotatedEvent.UNCHANGED, expected_state=EpistemicState.CONFIRMED_PRESENT)]

        dataset = RealWorkbenchDataset(
            root_dir=Path("."),
            dataset_name="test_order",
            description="test",
            visits=[
                RealVisitFrame("visit_01", "Visit 1", "2026-10-21T01:00:00Z", "", gt_v1),
                RealVisitFrame("visit_02", "Visit 2", "2026-10-21T02:00:00Z", "", gt_v2),
                RealVisitFrame("visit_03", "Visit 3", "2026-10-21T03:00:00Z", "", gt_v3),
            ],
        )

        sys_id = "sys_item_1"
        sys_entities = [WorkbenchEntity(sys_id, "item", "2026-10-21T01:00:00Z", gt_box, EpistemicState.CONFIRMED_PRESENT)]
        reports = [
            VisitProcessingReport("visit_01", "v1", "2026-10-21T01:00:00Z", "SUCCESS", 1, 1, 1, 0, [EntityBeliefUpdate(sys_id, "item", EpistemicState.INSUFFICIENT_EVIDENCE, EpistemicState.CONFIRMED_PRESENT, False)]),
            VisitProcessingReport("visit_02", "v2", "2026-10-21T02:00:00Z", "SUCCESS", 1, 1, 0, 0, [EntityBeliefUpdate(sys_id, "item", EpistemicState.CONFIRMED_PRESENT, EpistemicState.CONFIRMED_PRESENT, False)]),
            VisitProcessingReport("visit_03", "v3", "2026-10-21T03:00:00Z", "SUCCESS", 1, 1, 0, 0, [EntityBeliefUpdate(sys_id, "item", EpistemicState.CONFIRMED_PRESENT, EpistemicState.CONFIRMED_PRESENT, False)]),
        ]

        report = self.matcher.evaluate_sequence(dataset, sys_entities, reports)
        ordered_visit_ids = [rec.visit_id for rec in report.evaluations]
        self.assertEqual(ordered_visit_ids, ["visit_01", "visit_02", "visit_03"])


if __name__ == "__main__":
    unittest.main()
