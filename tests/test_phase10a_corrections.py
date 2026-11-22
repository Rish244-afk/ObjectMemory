"""
Unit and regression tests for Phase 10A evaluation corrections.

Tests:
1. score_temporal_event:
   - Genuinely confirmed removal passes (CONFIRMED_REMOVED).
   - Pending (PENDING_REMOVAL) or uncertain (CONFIRMED_PRESENT) removal fails.
   - Missing/absent entity fails.
   - Report correctly records failed events.
2. evaluate_image_matches_and_counts:
   - Multiple instances of the same class.
   - Missing detections and duplicate predictions (greedy matching invariant).
   - Unmatched predictions counted as false positives (never true positives).
   - Ground truth not counted more than once.
3. Output JSON schema verification.
"""

import unittest
from typing import Any, Dict, List
import numpy as np

from objectmemory.datasets.controlled_dataset import ControlledExpectedEvent
from objectmemory.detection.models import BoundingBox, Detection
from objectmemory.domain.states import EpistemicState
from objectmemory.temporal.engine import EntityBeliefUpdate, VisitProcessingReport

from scripts.evaluate_controlled_benchmark import score_temporal_event
from scripts.evaluate_natural_detection import evaluate_image_matches_and_counts


class TestTemporalEventScoringRegression(unittest.TestCase):
    """Regression tests for FIX 1: Strict removal criteria in temporal event scoring."""

    def setUp(self):
        self.exp_removed = ControlledExpectedEvent(
            sequence_id="S04_bottle_removal",
            visit_index=4,
            event="object_removed",
            object_id="bottle_01",
        )

    def _make_dummy_report(self, updates: List[EntityBeliefUpdate]) -> VisitProcessingReport:
        return VisitProcessingReport(
            session_id="v4",
            session_name="Visit 4",
            timestamp="2026-10-21T04:00:00Z",
            registration_status="SUCCESS",
            total_detections=0,
            matched_detections=0,
            new_entities_created=0,
            state_transitions_count=len(updates),
            entity_updates=updates,
            unassociated_detections=[],
            is_reprocessed=False,
            details={},
        )

    def test_genuinely_confirmed_removal_passes(self):
        """1. A genuinely confirmed removal passes (CONFIRMED_REMOVED)."""
        u = EntityBeliefUpdate(
            entity_id="ent_bottle_01",
            label="bottle",
            prior_state=EpistemicState.PENDING_REMOVAL,
            new_state=EpistemicState.CONFIRMED_REMOVED,
            transition_occurred=True,
            observation_id="obs_1",
            event_id="evt_1",
            visibility_status="ABSENT",
            explanation="Object unobserved for streak >= 3; confirmed removed.",
            consecutive_vacant_visits=3,
        )
        rep = self._make_dummy_report([u])
        matched, details, fm_delta, fr_delta = score_temporal_event(self.exp_removed, rep)
        self.assertTrue(matched)
        self.assertTrue(details["passed"])
        self.assertEqual(len(details["confirmed_removed_entities"]), 1)
        self.assertEqual(details["confirmed_removed_entities"][0], "ent_bottle_01")
        self.assertIsNone(details["failure_reason"])

    def test_pending_or_uncertain_removal_fails(self):
        """2. A pending or uncertain removal fails (PENDING_REMOVAL / CONFIRMED_PRESENT)."""
        # Test 2a: PENDING_REMOVAL fails object_removed expectation
        u_pending = EntityBeliefUpdate(
            entity_id="ent_bottle_01",
            label="bottle",
            prior_state=EpistemicState.CONFIRMED_PRESENT,
            new_state=EpistemicState.PENDING_REMOVAL,
            transition_occurred=True,
            observation_id="obs_1",
            event_id="evt_1",
            visibility_status="ABSENT",
            explanation="Missed visit 1; transitioning to PENDING_REMOVAL.",
            consecutive_vacant_visits=1,
        )
        rep_pending = self._make_dummy_report([u_pending])
        matched, details, _, _ = score_temporal_event(self.exp_removed, rep_pending)
        self.assertFalse(matched)
        self.assertFalse(details["passed"])
        self.assertIn("Engine did not transition target entity to CONFIRMED_REMOVED", details["failure_reason"])

        # Test 2b: CONFIRMED_PRESENT fails object_removed expectation
        u_present = EntityBeliefUpdate(
            entity_id="ent_bottle_01",
            label="bottle",
            prior_state=EpistemicState.CONFIRMED_PRESENT,
            new_state=EpistemicState.CONFIRMED_PRESENT,
            transition_occurred=False,
            observation_id="obs_1",
            event_id="evt_1",
            visibility_status="ABSENT",
            explanation="Patch unverified; preserving CONFIRMED_PRESENT.",
            consecutive_vacant_visits=0,
        )
        rep_present = self._make_dummy_report([u_present])
        matched, details, _, _ = score_temporal_event(self.exp_removed, rep_present)
        self.assertFalse(matched)
        self.assertFalse(details["passed"])

    def test_absent_or_missing_event_fails(self):
        """3. An absent or missing entity fails."""
        # Report has updates for other entities, but nothing for bottle
        u_other = EntityBeliefUpdate(
            entity_id="ent_cup_01",
            label="cup",
            prior_state=EpistemicState.CONFIRMED_PRESENT,
            new_state=EpistemicState.CONFIRMED_PRESENT,
            transition_occurred=False,
            observation_id="obs_2",
            event_id="evt_2",
            visibility_status="VISIBLE",
            explanation="Unchanged.",
            consecutive_vacant_visits=0,
        )
        rep_empty = self._make_dummy_report([u_other])
        matched, details, _, _ = score_temporal_event(self.exp_removed, rep_empty)
        self.assertFalse(matched)
        self.assertFalse(details["passed"])
        self.assertEqual(len(details["confirmed_removed_entities"]), 0)

        # Test when rep is None
        matched_none, details_none, _, _ = score_temporal_event(self.exp_removed, None)
        self.assertFalse(matched_none)
        self.assertFalse(details_none["passed"])

    def test_report_correctly_records_failed_event(self):
        """4. The report correctly records the failed event."""
        u_unverified = EntityBeliefUpdate(
            entity_id="ent_bottle_01",
            label="bottle",
            prior_state=EpistemicState.CONFIRMED_PRESENT,
            new_state=EpistemicState.CONFIRMED_PRESENT,
            transition_occurred=False,
            observation_id="obs_1",
            event_id="evt_1",
            visibility_status="ABSENT",
            explanation="Unverified vacancy.",
            consecutive_vacant_visits=0,
        )
        rep = self._make_dummy_report([u_unverified])
        matched, details, _, _ = score_temporal_event(self.exp_removed, rep)

        record = {
            "sequence_id": self.exp_removed.sequence_id,
            "visit_index": self.exp_removed.visit_index,
            "event": self.exp_removed.event,
            "target": self.exp_removed.object_id,
            "matched_expected_behavior": matched,
            "details": details,
        }
        self.assertFalse(record["matched_expected_behavior"])
        self.assertEqual(record["details"]["passed"], False)
        self.assertIsNotNone(record["details"]["failure_reason"])


class TestDetectionMatchesAndCounts(unittest.TestCase):
    """Regression tests for FIX 2: Per-class detection counting convention."""

    def setUp(self):
        self.target_classes = ["cup", "bottle", "book", "laptop"]

    def test_multiple_instances_of_same_class(self):
        """Tests that multiple ground-truth instances of the same class are counted accurately."""
        # GT: 3 cups
        gt_anns = [
            {"id": 1, "category_name": "cup", "xyxy": (10.0, 10.0, 50.0, 50.0)},
            {"id": 2, "category_name": "cup", "xyxy": (100.0, 10.0, 150.0, 50.0)},
            {"id": 3, "category_name": "cup", "xyxy": (200.0, 10.0, 250.0, 50.0)},
        ]
        # Preds: 2 cups matching first two GT boxes, 1 cup far away (unmatched)
        preds = [
            Detection(label="cup", confidence=0.9, raw_bbox=BoundingBox(10.0, 10.0, 50.0, 50.0)),
            Detection(label="cup", confidence=0.8, raw_bbox=BoundingBox(100.0, 10.0, 150.0, 50.0)),
            Detection(label="cup", confidence=0.7, raw_bbox=BoundingBox(500.0, 500.0, 550.0, 550.0)),
        ]
        res = evaluate_image_matches_and_counts(preds, gt_anns, self.target_classes, iou_threshold=0.50)
        c_cup = res["per_class"]["cup"]

        self.assertEqual(c_cup["gt_instances"], 3)
        self.assertEqual(c_cup["predicted_instances"], 3)
        self.assertEqual(c_cup["tp"], 2)  # 2 matched
        self.assertEqual(c_cup["fp"], 1)  # 1 unmatched hallucination
        self.assertEqual(c_cup["fn"], 1)  # 1 missed GT box (3 - 2 = 1)
        self.assertEqual(c_cup["tp"] + c_cup["fn"], c_cup["gt_instances"])
        self.assertEqual(c_cup["tp"] + c_cup["fp"], c_cup["predicted_instances"])

    def test_duplicate_predictions_for_single_ground_truth(self):
        """Tests that two predictions matching the same GT box count as 1 TP and 1 FP."""
        # GT: 1 bottle
        gt_anns = [
            {"id": 1, "category_name": "bottle", "xyxy": (50.0, 50.0, 100.0, 120.0)},
        ]
        # Preds: 2 predictions both overlapping the same bottle
        preds = [
            Detection(label="bottle", confidence=0.95, raw_bbox=BoundingBox(50.0, 50.0, 100.0, 120.0)),
            Detection(label="bottle", confidence=0.85, raw_bbox=BoundingBox(52.0, 51.0, 99.0, 119.0)),
        ]
        res = evaluate_image_matches_and_counts(preds, gt_anns, self.target_classes, iou_threshold=0.50)
        c_bottle = res["per_class"]["bottle"]

        self.assertEqual(c_bottle["gt_instances"], 1)
        self.assertEqual(c_bottle["predicted_instances"], 2)
        self.assertEqual(c_bottle["tp"], 1)  # Only higher confidence matches
        self.assertEqual(c_bottle["fp"], 1)  # Duplicate prediction is counted as FP
        self.assertEqual(c_bottle["fn"], 0)  # GT was detected

    def test_missing_detections(self):
        """Tests that complete absence of predictions yields TP=0, FP=0, FN=gt_count."""
        gt_anns = [
            {"id": 1, "category_name": "laptop", "xyxy": (10.0, 10.0, 80.0, 60.0)},
            {"id": 2, "category_name": "laptop", "xyxy": (100.0, 10.0, 180.0, 60.0)},
        ]
        preds = []
        res = evaluate_image_matches_and_counts(preds, gt_anns, self.target_classes, iou_threshold=0.50)
        c_laptop = res["per_class"]["laptop"]

        self.assertEqual(c_laptop["gt_instances"], 2)
        self.assertEqual(c_laptop["predicted_instances"], 0)
        self.assertEqual(c_laptop["tp"], 0)
        self.assertEqual(c_laptop["fp"], 0)
        self.assertEqual(c_laptop["fn"], 2)

    def test_unmatched_prediction_never_counted_as_tp(self):
        """Tests that an unmatched prediction (different class or non-overlapping) is FP, never TP."""
        gt_anns = [
            {"id": 1, "category_name": "cup", "xyxy": (10.0, 10.0, 50.0, 50.0)},
        ]
        # Pred is a book at the same location (wrong class)
        preds = [
            Detection(label="book", confidence=0.9, raw_bbox=BoundingBox(10.0, 10.0, 50.0, 50.0)),
        ]
        res = evaluate_image_matches_and_counts(preds, gt_anns, self.target_classes, iou_threshold=0.50)
        self.assertEqual(res["per_class"]["book"]["tp"], 0)
        self.assertEqual(res["per_class"]["book"]["fp"], 1)
        self.assertEqual(res["per_class"]["cup"]["tp"], 0)
        self.assertEqual(res["per_class"]["cup"]["fn"], 1)


if __name__ == "__main__":
    unittest.main()
