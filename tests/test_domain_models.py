"""
Unit tests for ObjectMemory domain models and validation logic.
"""

import math
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from objectmemory.domain.models import (
    BoundingBox,
    ChangeEvent,
    Observation,
    PairwiseBaselineResult,
    Session,
    WorkbenchEntity,
    normalize_iso_timestamp,
)
from objectmemory.domain.states import (
    BaselineChangeType,
    EpistemicState,
    VisibilityStatus,
    is_valid_transition,
)


class TestDomainModels(unittest.TestCase):

    def test_bounding_box_valid(self):
        bbox = BoundingBox(10.0, 20.0, 50.0, 60.0)
        self.assertEqual(bbox.as_tuple(), (10.0, 20.0, 50.0, 60.0))
        self.assertEqual(BoundingBox.from_tuple((0.0, 0.0, 1.0, 1.0)).as_tuple(), (0.0, 0.0, 1.0, 1.0))

    def test_bounding_box_invalid_inversion(self):
        with self.assertRaises(ValueError):
            BoundingBox(50.0, 20.0, 10.0, 60.0)  # x_min > x_max

        with self.assertRaises(ValueError):
            BoundingBox(10.0, 80.0, 50.0, 60.0)  # y_min > y_max

    def test_bounding_box_negative_coords(self):
        with self.assertRaises(ValueError):
            BoundingBox(-5.0, 10.0, 20.0, 30.0)

    def test_bounding_box_nan_and_inf_rejected(self):
        """Priority 3: Rejects NaN, +Inf, -Inf on every coordinate."""
        for bad_val in [float("nan"), float("inf"), float("-inf")]:
            with self.assertRaises(ValueError):
                BoundingBox(bad_val, 10.0, 20.0, 30.0)
            with self.assertRaises(ValueError):
                BoundingBox(10.0, bad_val, 20.0, 30.0)
            with self.assertRaises(ValueError):
                BoundingBox(10.0, 20.0, bad_val, 30.0)
            with self.assertRaises(ValueError):
                BoundingBox(10.0, 20.0, 30.0, bad_val)

    def test_bounding_box_from_tuple_invalid_length(self):
        with self.assertRaises(ValueError):
            BoundingBox.from_tuple((10.0, 20.0, 30.0))  # type: ignore

    def test_unit_interval_validation(self):
        # Valid confidence in [0, 1]
        obs = Observation(
            observation_id="obs-1",
            session_id="sess-1",
            timestamp="2026-10-10T10:00:00Z",
            image_reference="frame_001.jpg",
            detector_confidence=0.85,
            identity_match_score=0.92,
            alignment_quality=0.98,
        )
        self.assertEqual(obs.detector_confidence, 0.85)

        # Invalid: > 1.0
        with self.assertRaises(ValueError):
            Observation(
                observation_id="obs-2",
                session_id="sess-1",
                timestamp="2026-10-10T10:00:00Z",
                image_reference="frame_001.jpg",
                detector_confidence=1.2,
            )

        # Invalid: < 0.0
        with self.assertRaises(ValueError):
            Observation(
                observation_id="obs-3",
                session_id="sess-1",
                timestamp="2026-10-10T10:00:00Z",
                image_reference="frame_001.jpg",
                identity_match_score=-0.1,
            )

        # Invalid type
        with self.assertRaises(TypeError):
            Observation(
                observation_id="obs-4",
                session_id="sess-1",
                timestamp="2026-10-10T10:00:00Z",
                image_reference="frame_001.jpg",
                alignment_quality="high",  # type: ignore
            )

    def test_confidence_scores_nan_and_inf_rejected(self):
        """Priority 3: Rejects NaN and infinity in evidence scores."""
        for bad_score in [float("nan"), float("inf"), float("-inf")]:
            with self.assertRaises(ValueError):
                Observation(
                    observation_id="obs-nan",
                    session_id="sess-1",
                    timestamp="2026-10-10T10:00:00Z",
                    image_reference="img.jpg",
                    detector_confidence=bad_score,
                )
            with self.assertRaises(ValueError):
                Observation(
                    observation_id="obs-inf",
                    session_id="sess-1",
                    timestamp="2026-10-10T10:00:00Z",
                    image_reference="img.jpg",
                    identity_match_score=bad_score,
                )

    def test_timestamp_iso_normalization_and_validation(self):
        """Priority 3: Validates and normalizes ISO-8601 timestamps to UTC."""
        # Valid ISO with Z
        self.assertEqual(
            normalize_iso_timestamp("2026-10-10T10:00:00Z"),
            "2026-10-10T10:00:00Z",
        )
        # Valid naive timestamp deliberately normalized to UTC per policy
        self.assertEqual(
            normalize_iso_timestamp("2026-10-10T10:00:00"),
            "2026-10-10T10:00:00Z",
        )
        # Valid timezone-aware string converted to UTC
        self.assertEqual(
            normalize_iso_timestamp("2026-10-10T15:30:00+05:30"),
            "2026-10-10T10:00:00Z",
        )
        # Datetime object input
        dt_obj = datetime(2026, 10, 10, 10, 0, 0, tzinfo=timezone.utc)
        self.assertEqual(normalize_iso_timestamp(dt_obj), "2026-10-10T10:00:00Z")

        # Invalid formats
        for invalid_ts in ["not-a-timestamp", "2026/10/10", "", "   ", "2026-13-45T99:99:99"]:
            with self.assertRaises(ValueError):
                normalize_iso_timestamp(invalid_ts)

    def test_optional_evidence_not_manufactured(self):
        """Safeguard: Missing evidence scores must default to None, not 0.0 or 1.0."""
        obs = Observation(
            observation_id="obs-empty-eval",
            session_id="sess-1",
            timestamp="2026-10-10T10:00:00Z",
            image_reference="frame_001.jpg",
        )
        self.assertIsNone(obs.detector_confidence)
        self.assertIsNone(obs.identity_match_score)
        self.assertIsNone(obs.alignment_quality)
        self.assertEqual(obs.visibility_status, VisibilityStatus.UNCERTAIN)

    def test_epistemic_states_defined(self):
        """Verifies presence of all required epistemic state enums."""
        required = [
            "CONFIRMED_PRESENT",
            "PENDING_REMOVAL",
            "CONFIRMED_MOVED",
            "CONFIRMED_REMOVED",
            "INSUFFICIENT_EVIDENCE",
            "AMBIGUOUS_IDENTITY",
        ]
        enum_values = [s.value for s in EpistemicState]
        for req in required:
            self.assertIn(req, enum_values)

    def test_state_transition_policy_rules(self):
        """Priority 1: Enforces documented state transition policy."""
        # Allowed transitions
        self.assertTrue(is_valid_transition(EpistemicState.CONFIRMED_PRESENT, EpistemicState.CONFIRMED_MOVED))
        self.assertTrue(is_valid_transition(EpistemicState.CONFIRMED_PRESENT, EpistemicState.PENDING_REMOVAL))
        self.assertTrue(is_valid_transition(EpistemicState.PENDING_REMOVAL, EpistemicState.CONFIRMED_REMOVED))

        # Disallowed transition: direct jump from PRESENT to REMOVED is forbidden by policy
        self.assertFalse(is_valid_transition(EpistemicState.CONFIRMED_PRESENT, EpistemicState.CONFIRMED_REMOVED))

    def test_session_validation(self):
        with self.assertRaises(ValueError):
            Session(
                session_id="",
                session_name="Empty ID",
                timestamp="2026-10-10T10:00:00Z",
                image_path="test.png",
            )
        with self.assertRaises(ValueError):
            Session(
                session_id="sess-1",
                session_name="Empty Path",
                timestamp="2026-10-10T10:00:00Z",
                image_path="   ",
            )
        # Invalid timestamp
        with self.assertRaises(ValueError):
            Session(
                session_id="sess-1",
                session_name="Bad Date",
                timestamp="yesterday",
                image_path="test.png",
            )

    def test_change_event_validation(self):
        event = ChangeEvent(
            event_id="evt-1",
            entity_id="ent-multimeter",
            to_session_id="sess-2",
            timestamp="2026-10-10T10:30:00Z",
            prior_state=EpistemicState.CONFIRMED_PRESENT,
            new_state=EpistemicState.CONFIRMED_MOVED,
            displacement_distance=15.4,
            supporting_evidence={"delta_x": 12.0, "delta_y": 9.5},
            explanation="Object shifted 15.4 cm towards right rail.",
        )
        self.assertEqual(event.new_state, EpistemicState.CONFIRMED_MOVED)
        self.assertEqual(event.displacement_distance, 15.4)


if __name__ == "__main__":
    unittest.main()
