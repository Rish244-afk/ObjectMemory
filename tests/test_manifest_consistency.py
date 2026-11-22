"""
Unit tests for Dataset Manifest Validation and Contradiction Detection (Phase 9B).
Verifies:
- Detection of contradictions between annotated_event and expected_state.
- Detection of contradictions between physical_observation and expected_state.
- Detection of non-monotonic timestamps or unobservable boxes.
- Validation of desk_pilot_01 manifest integrity.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from objectmemory.datasets.real_dataset import (
    AnnotatedEvent,
    PhysicalObservation,
    RealVisitAnnotation,
    RealVisitFrame,
    RealWorkbenchDataset,
)
from objectmemory.domain.models import BoundingBox
from objectmemory.domain.states import EpistemicState


class TestManifestConsistency(unittest.TestCase):
    def test_reconciled_desk_pilot_manifest_validity(self) -> None:
        """Verifies the actual desk_pilot_01 manifest passes all integrity checks."""
        manifest_path = Path("data/real_workbench/manifest.json")
        if not manifest_path.exists():
            self.skipTest(f"Manifest not found at {manifest_path}")

        dataset = RealWorkbenchDataset.load_from_directory(manifest_path.parent)
        errors = dataset.validate_dataset_integrity()
        self.assertEqual(errors, [], f"Dataset integrity failed with errors: {errors}")

    def test_semantic_contradiction_absent_with_confirmed_present(self) -> None:
        """Fails when physical_observation=ABSENT is paired with expected_state=CONFIRMED_PRESENT."""
        annots = [
            RealVisitAnnotation(
                entity_id="cup_01",
                entity_label="cup",
                bbox=None,
                physical_observation=PhysicalObservation.ABSENT,
                annotated_event=AnnotatedEvent.ABSENT,
                expected_state=EpistemicState.CONFIRMED_PRESENT, # Contradiction!
            ),
        ]
        dataset = RealWorkbenchDataset(
            root_dir=Path("."),
            dataset_name="contradictory_dataset",
            description="test",
            visits=[
                RealVisitFrame(visit_id="v1", session_name="v1", timestamp="2026-10-21T00:00:00Z", image_path=Path("data/real_workbench/visits/visit_01.png"), annotations=annots),
                RealVisitFrame(visit_id="v2", session_name="v2", timestamp="2026-10-21T01:00:00Z", image_path=Path("data/real_workbench/visits/visit_02.png"), annotations=[]),
            ],
        )
        errors = dataset.validate_dataset_integrity()
        self.assertTrue(any("cannot be paired with expected_state='CONFIRMED_PRESENT'" in err for err in errors))

    def test_semantic_contradiction_unchanged_with_confirmed_removed(self) -> None:
        """Fails when annotated_event=UNCHANGED is paired with expected_state=CONFIRMED_REMOVED."""
        annots = [
            RealVisitAnnotation(
                entity_id="bottle_01",
                entity_label="bottle",
                bbox=BoundingBox(10.0, 10.0, 50.0, 50.0),
                physical_observation=PhysicalObservation.VISIBLE,
                annotated_event=AnnotatedEvent.UNCHANGED,
                expected_state=EpistemicState.CONFIRMED_REMOVED, # Contradiction!
            ),
        ]
        dataset = RealWorkbenchDataset(
            root_dir=Path("."),
            dataset_name="contradictory_dataset",
            description="test",
            visits=[
                RealVisitFrame(visit_id="v1", session_name="v1", timestamp="2026-10-21T00:00:00Z", image_path=Path("data/real_workbench/visits/visit_01.png"), annotations=annots),
                RealVisitFrame(visit_id="v2", session_name="v2", timestamp="2026-10-21T01:00:00Z", image_path=Path("data/real_workbench/visits/visit_02.png"), annotations=[]),
            ],
        )
        errors = dataset.validate_dataset_integrity()
        self.assertTrue(any("cannot be paired with expected_state='CONFIRMED_REMOVED'" in err for err in errors))

    def test_semantic_contradiction_visible_with_null_bbox(self) -> None:
        """Fails when physical_observation=VISIBLE but bbox is null."""
        annots = [
            RealVisitAnnotation(
                entity_id="bottle_01",
                entity_label="bottle",
                bbox=None, # Contradiction for VISIBLE!
                physical_observation=PhysicalObservation.VISIBLE,
                annotated_event=AnnotatedEvent.UNCHANGED,
                expected_state=EpistemicState.CONFIRMED_PRESENT,
            ),
        ]
        dataset = RealWorkbenchDataset(
            root_dir=Path("."),
            dataset_name="contradictory_dataset",
            description="test",
            visits=[
                RealVisitFrame(visit_id="v1", session_name="v1", timestamp="2026-10-21T00:00:00Z", image_path=Path("data/real_workbench/visits/visit_01.png"), annotations=annots),
                RealVisitFrame(visit_id="v2", session_name="v2", timestamp="2026-10-21T01:00:00Z", image_path=Path("data/real_workbench/visits/visit_02.png"), annotations=[]),
            ],
        )
        errors = dataset.validate_dataset_integrity()
        self.assertTrue(any("physical_observation='VISIBLE' but bbox is null/missing" in err for err in errors))


if __name__ == "__main__":
    unittest.main()
