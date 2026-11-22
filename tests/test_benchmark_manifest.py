"""
Comprehensive Unit & Regression Tests for ObjectMemory Evaluation Manifest Loader (Phase 3A.1).

Covers:
- All 12 baseline tests from Phase 3A.
- 20 specific hardening tests:
  1. Missing image_path
  2. Null image_path
  3. Empty image_path
  4. Malformed bounding-box dictionary
  5. Wrong number of bounding-box coordinates
  6. Non-numeric coordinates
  7. NaN or infinite coordinates
  8. Reversed or zero-area bounding boxes
  9. Empty object ID
  10. Empty class name
  11. Invalid confidence values
  12. Unknown visibility/state values
  13. Invalid declared dataset root
  14. Explicit dataset_root precedence
  15. Reference image existence
  16. Depth and semantic file existence
  17. Symlink escaping dataset_root
  18. Equal timestamps
  19. Timezone-offset comparison
  20. Provenance classification and unknown provenance
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from objectmemory.benchmarks.manifest import (
    DatasetManifest,
    DatasetProvenance,
    ManifestLoader,
    ManifestValidationError,
    ObjectAnnotationEntry,
)
from objectmemory.domain.states import EpistemicState, VisibilityStatus


class TestBenchmarkManifestLoader(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_manifest_"))
        self.addCleanup(lambda: shutil.rmtree(self.temp_dir, ignore_errors=True))

        # Setup standard directory structure
        self.img_dir = self.temp_dir / "images"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.ann_dir = self.temp_dir / "annotations"
        self.ann_dir.mkdir(parents=True, exist_ok=True)
        self.cal_dir = self.temp_dir / "calib"
        self.cal_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy image files
        self.img1 = self.img_dir / "visit_01.png"
        self.img1.write_bytes(b"PNG_MOCK_1")
        self.img2 = self.img_dir / "visit_02.png"
        self.img2.write_bytes(b"PNG_MOCK_2")
        self.img3 = self.img_dir / "visit_03.png"
        self.img3.write_bytes(b"PNG_MOCK_3")

        # Create dummy reference and aux files
        self.ref_img = self.img_dir / "ref.png"
        self.ref_img.write_bytes(b"PNG_REF")
        self.depth1 = self.img_dir / "depth_01.png"
        self.depth1.write_bytes(b"DEPTH_1")
        self.sem1 = self.img_dir / "sem_01.png"
        self.sem1.write_bytes(b"SEM_1")

        # Create dummy annotation and calibration files
        self.ann1 = self.ann_dir / "ann_01.json"
        self.ann1.write_text('{"objects": []}', encoding="utf-8")
        self.cal1 = self.cal_dir / "calib.json"
        self.cal1.write_text('{"matrix": []}', encoding="utf-8")

    # =========================================================================
    # Phase 3A Baseline Tests (Preserved)
    # =========================================================================

    def test_valid_single_sequence_manifest(self) -> None:
        """Loads a valid single-sequence v1.0.0 manifest successfully."""
        data = {
            "schema_version": "1.0.0",
            "dataset_name": "PhysicalWorkbench",
            "environment_id": "bench_01",
            "sequence_id": "SEQ-01",
            "coordinate_convention": "PIXEL_XMIN_YMIN_XMAX_YMAX",
            "metadata": {"provenance": "manual test capture"},
            "observations": [
                {
                    "observation_id": "obs_01",
                    "timestamp": "2026-11-14T10:00:00Z",
                    "image_path": "images/visit_01.png",
                    "source_annotation_path": "annotations/ann_01.json",
                    "objects": [
                        {
                            "object_id": "cup_01",
                            "class_name": "cup",
                            "bounding_box": [10.0, 20.0, 50.0, 60.0],
                            "visibility_status": "CLEARLY_VISIBLE",
                            "object_state": "CONFIRMED_PRESENT",
                            "timestamp": "2026-11-14T10:00:00Z",
                        }
                    ],
                },
                {
                    "observation_id": "obs_02",
                    "timestamp": "2026-11-14T10:05:00Z",
                    "image_path": "images/visit_02.png",
                },
            ],
        }
        manifest_file = self.temp_dir / "single_manifest.json"
        manifest_file.write_text(json.dumps(data), encoding="utf-8")

        manifest = ManifestLoader.load(manifest_file, check_files=True)
        self.assertEqual(manifest.manifest_version, "1.0.0")
        self.assertEqual(manifest.dataset_name, "PhysicalWorkbench")
        self.assertEqual(manifest.sequence_count, 1)
        self.assertEqual(manifest.total_observations, 2)

        seq = manifest.get_sequence("SEQ-01")
        self.assertIsNotNone(seq)
        assert seq is not None
        self.assertEqual(len(seq.observations), 2)
        obs1 = seq.observations[0]
        self.assertEqual(obs1.observation_id, "obs_01")
        self.assertEqual(obs1.image_path, self.img1.resolve())
        self.assertEqual(len(obs1.objects), 1)
        self.assertEqual(obs1.objects[0].object_id, "cup_01")
        self.assertIsNotNone(obs1.objects[0].bounding_box)

    def test_valid_multiple_sequences_manifest(self) -> None:
        """Loads a valid multi-sequence v2.0 manifest successfully."""
        data = {
            "manifest_version": "2.0",
            "dataset_name": "PhysicalWorkbench-Multi",
            "provenance": "GENUINE_PHYSICAL",
            "metadata": {"total_benchmark_hours": 4},
            "sequences": [
                {
                    "sequence_id": "PHYS-01-STAT",
                    "environment_id": "bench_01",
                    "challenge_mode": "STATIONARY",
                    "observations": [
                        {
                            "observation_id": "stat_01",
                            "timestamp": "2026-11-14T09:00:00Z",
                            "image_path": "images/visit_01.png",
                        }
                    ],
                },
                {
                    "sequence_id": "PHYS-02-MOVE",
                    "environment_id": "bench_01",
                    "challenge_mode": "RELOCATION",
                    "observations": [
                        {
                            "observation_id": "move_01",
                            "timestamp": "2026-11-14T10:00:00Z",
                            "image_path": "images/visit_02.png",
                        },
                        {
                            "observation_id": "move_02",
                            "timestamp": "2026-11-14T10:15:00Z",
                            "image_path": "images/visit_03.png",
                        },
                    ],
                },
            ],
        }
        manifest_file = self.temp_dir / "multi_manifest.json"
        manifest_file.write_text(json.dumps(data), encoding="utf-8")

        manifest = ManifestLoader.load(manifest_file, check_files=True)
        self.assertEqual(manifest.manifest_version, "2.0")
        self.assertEqual(manifest.sequence_count, 2)
        self.assertEqual(manifest.total_observations, 3)
        self.assertTrue(manifest.is_genuine_physical)

    def test_missing_image_file_detection(self) -> None:
        """Asserts physical existence checks flag missing image files."""
        data = {
            "manifest_version": "2.0",
            "dataset_name": "TestMissing",
            "sequences": [
                {
                    "sequence_id": "SEQ-01",
                    "observations": [
                        {
                            "observation_id": "obs_01",
                            "timestamp": "2026-11-14T10:00:00Z",
                            "image_path": "images/non_existent.png",
                        }
                    ],
                }
            ],
        }
        manifest = ManifestLoader.parse_dict(data, dataset_root=self.temp_dir, check_files=False)
        errors = manifest.validate_files(check_images=True)
        self.assertTrue(any("Image file not found on disk" in e for e in errors))

        with self.assertRaises(ManifestValidationError) as ctx:
            ManifestLoader.parse_dict(data, dataset_root=self.temp_dir, check_files=True)
        self.assertTrue(any("Image file not found on disk" in e for e in ctx.exception.errors))

    def test_missing_annotation_file_detection(self) -> None:
        """Asserts physical existence checks flag missing linked annotation files."""
        data = {
            "manifest_version": "2.0",
            "dataset_name": "TestMissingAnn",
            "sequences": [
                {
                    "sequence_id": "SEQ-01",
                    "ground_truth_file": "annotations/missing_gt.json",
                    "observations": [
                        {
                            "observation_id": "obs_01",
                            "timestamp": "2026-11-14T10:00:00Z",
                            "image_path": "images/visit_01.png",
                            "source_annotation_path": "annotations/missing_obs_ann.json",
                        }
                    ],
                }
            ],
        }
        manifest = ManifestLoader.parse_dict(data, dataset_root=self.temp_dir, check_files=False)
        errors = manifest.validate_files(check_annotations=True, check_images=False)
        self.assertTrue(any("ground-truth file not found" in e for e in errors))
        self.assertTrue(any("Observation annotation file not found" in e for e in errors))

    def test_duplicate_observation_id_detection(self) -> None:
        """Detects duplicate observation identifiers within and across sequences."""
        data = {
            "manifest_version": "2.0",
            "dataset_name": "TestDupObs",
            "sequences": [
                {
                    "sequence_id": "SEQ-01",
                    "observations": [
                        {"observation_id": "dup_01", "timestamp": "2026-11-14T10:00:00Z", "image_path": "images/visit_01.png"},
                        {"observation_id": "dup_01", "timestamp": "2026-11-14T10:05:00Z", "image_path": "images/visit_02.png"},
                    ],
                }
            ],
        }
        with self.assertRaises(ManifestValidationError) as ctx:
            ManifestLoader.parse_dict(data, dataset_root=self.temp_dir)
        self.assertTrue(any("Duplicate observation_id detected: 'dup_01'" in e for e in ctx.exception.errors))

    def test_duplicate_sequence_id_detection(self) -> None:
        """Detects duplicate sequence identifiers."""
        data = {
            "manifest_version": "2.0",
            "dataset_name": "TestDupSeq",
            "sequences": [
                {
                    "sequence_id": "SEQ-DUP",
                    "observations": [{"observation_id": "obs_1", "timestamp": "2026-11-14T10:00:00Z", "image_path": "images/visit_01.png"}],
                },
                {
                    "sequence_id": "SEQ-DUP",
                    "observations": [{"observation_id": "obs_2", "timestamp": "2026-11-14T10:00:00Z", "image_path": "images/visit_02.png"}],
                },
            ],
        }
        with self.assertRaises(ManifestValidationError) as ctx:
            ManifestLoader.parse_dict(data, dataset_root=self.temp_dir)
        self.assertTrue(any("Duplicate sequence_id detected: 'SEQ-DUP'" in e for e in ctx.exception.errors))

    def test_invalid_timestamp_format(self) -> None:
        """Rejects observations with malformed or unparseable timestamps."""
        data = {
            "manifest_version": "2.0",
            "dataset_name": "TestTimestamp",
            "sequences": [
                {
                    "sequence_id": "SEQ-01",
                    "observations": [
                        {"observation_id": "obs_1", "timestamp": "NOT-A-VALID-DATE", "image_path": "images/visit_01.png"}
                    ],
                }
            ],
        }
        with self.assertRaises(ManifestValidationError) as ctx:
            ManifestLoader.parse_dict(data, dataset_root=self.temp_dir)
        self.assertTrue(any("Invalid timestamp 'NOT-A-VALID-DATE'" in e for e in ctx.exception.errors))

    def test_invalid_sequence_ordering(self) -> None:
        """Rejects sequences whose observations are non-chronological without silently reordering."""
        data = {
            "manifest_version": "2.0",
            "dataset_name": "TestOrdering",
            "sequences": [
                {
                    "sequence_id": "SEQ-01",
                    "observations": [
                        {"observation_id": "obs_1", "timestamp": "2026-11-14T10:30:00Z", "image_path": "images/visit_01.png"},
                        {"observation_id": "obs_2", "timestamp": "2026-11-14T10:15:00Z", "image_path": "images/visit_02.png"},
                    ],
                }
            ],
        }
        with self.assertRaises(ManifestValidationError) as ctx:
            ManifestLoader.parse_dict(data, dataset_root=self.temp_dir)
        self.assertTrue(any("Invalid sequence ordering" in e for e in ctx.exception.errors))
        self.assertTrue(any("strictly precedes" in e for e in ctx.exception.errors))

    def test_invalid_relative_path_traversal(self) -> None:
        """Rejects relative paths that attempt directory traversal outside dataset root."""
        data = {
            "manifest_version": "2.0",
            "dataset_name": "TestTraversal",
            "sequences": [
                {
                    "sequence_id": "SEQ-01",
                    "observations": [
                        {"observation_id": "obs_1", "timestamp": "2026-11-14T10:00:00Z", "image_path": "../../secret.png"}
                    ],
                }
            ],
        }
        with self.assertRaises(ManifestValidationError) as ctx:
            ManifestLoader.parse_dict(data, dataset_root=self.temp_dir)
        self.assertTrue(any("Path traversal violation detected" in e for e in ctx.exception.errors))

    def test_empty_sequence_rejection(self) -> None:
        """Disallows sequences containing zero observations."""
        data = {
            "manifest_version": "2.0",
            "dataset_name": "TestEmpty",
            "sequences": [
                {
                    "sequence_id": "SEQ-EMPTY",
                    "observations": [],
                }
            ],
        }
        with self.assertRaises(ManifestValidationError) as ctx:
            ManifestLoader.parse_dict(data, dataset_root=self.temp_dir)
        self.assertTrue(any("contains 0 observations" in e for e in ctx.exception.errors))

    def test_unsupported_manifest_version(self) -> None:
        """Rejects unknown or unsupported manifest schema versions."""
        data = {
            "manifest_version": "99.0.0",
            "dataset_name": "TestVersion",
            "sequences": [
                {
                    "sequence_id": "SEQ-01",
                    "observations": [
                        {"observation_id": "obs_1", "timestamp": "2026-11-14T10:00:00Z", "image_path": "images/visit_01.png"}
                    ],
                }
            ],
        }
        with self.assertRaises(ManifestValidationError) as ctx:
            ManifestLoader.parse_dict(data, dataset_root=self.temp_dir)
        self.assertTrue(any("Unsupported manifest version '99.0.0'" in e for e in ctx.exception.errors))

    def test_deterministic_loading_and_retrieval(self) -> None:
        """Asserts observations maintain deterministic order and pilot status is accurately flagged."""
        data = {
            "manifest_version": "2.0",
            "dataset_name": "PhysicalWorkbench-Benchmark",
            "provenance": "GENUINE_PHYSICAL",
            "sequences": [
                {
                    "sequence_id": "PHYS-01-STAT",
                    "observations": [
                        {"observation_id": "visit_01", "timestamp": "2026-11-14T10:00:00Z", "image_path": "images/visit_01.png"},
                        {"observation_id": "visit_02", "timestamp": "2026-11-14T10:05:00Z", "image_path": "images/visit_02.png"},
                    ],
                }
            ],
        }
        manifest = ManifestLoader.parse_dict(data, dataset_root=self.temp_dir, check_files=True)
        self.assertFalse(manifest.is_pilot_dataset)

        seq = manifest.get_sequence("PHYS-01-STAT")
        assert seq is not None
        self.assertEqual([o.observation_id for o in seq.observations], ["visit_01", "visit_02"])
        self.assertEqual(seq.get_observation("visit_01").observation_id, "visit_01")
        self.assertIsNone(seq.get_observation("nonexistent"))

    # =========================================================================
    # Phase 3A.1 Hardening Regression Tests (1 - 20)
    # =========================================================================

    def test_01_missing_image_path(self) -> None:
        """Requirement 1: Missing image_path generates contextual error."""
        data = {
            "manifest_version": "2.0",
            "dataset_name": "Test",
            "sequences": [
                {"sequence_id": "S1", "observations": [{"observation_id": "O1", "timestamp": "2026-11-14T10:00:00Z"}]}
            ],
        }
        with self.assertRaises(ManifestValidationError) as ctx:
            ManifestLoader.parse_dict(data, dataset_root=self.temp_dir)
        self.assertTrue(any("Missing or empty required 'image_path'" in e for e in ctx.exception.errors))

    def test_02_null_image_path(self) -> None:
        """Requirement 1: Null image_path generates contextual error."""
        data = {
            "manifest_version": "2.0",
            "dataset_name": "Test",
            "sequences": [
                {"sequence_id": "S1", "observations": [{"observation_id": "O1", "timestamp": "2026-11-14T10:00:00Z", "image_path": None}]}
            ],
        }
        with self.assertRaises(ManifestValidationError) as ctx:
            ManifestLoader.parse_dict(data, dataset_root=self.temp_dir)
        self.assertTrue(any("Missing or empty required 'image_path'" in e for e in ctx.exception.errors))

    def test_03_empty_image_path(self) -> None:
        """Requirement 1: Empty string image_path generates contextual error."""
        data = {
            "manifest_version": "2.0",
            "dataset_name": "Test",
            "sequences": [
                {"sequence_id": "S1", "observations": [{"observation_id": "O1", "timestamp": "2026-11-14T10:00:00Z", "image_path": "   "}]}
            ],
        }
        with self.assertRaises(ManifestValidationError) as ctx:
            ManifestLoader.parse_dict(data, dataset_root=self.temp_dir)
        self.assertTrue(any("Missing or empty required 'image_path'" in e for e in ctx.exception.errors))

    def test_04_malformed_bounding_box_dict(self) -> None:
        """Requirement 2: Bounding box dictionary missing required keys is rejected."""
        raw_ann = {
            "object_id": "obj1",
            "class_name": "cup",
            "bounding_box": {"x_min": 10.0, "y_min": 20.0}, # Missing x_max, y_max
            "visibility_status": "CLEARLY_VISIBLE",
            "object_state": "CONFIRMED_PRESENT",
        }
        entry, errs = ObjectAnnotationEntry.from_dict(raw_ann, scope="[test]")
        self.assertIsNone(entry.bounding_box)
        self.assertTrue(any("missing required key 'x_max'" in e for e in errs))

    def test_05_wrong_number_bbox_coordinates(self) -> None:
        """Requirement 2: Sequence with != 4 coordinates is rejected."""
        raw_ann = {
            "object_id": "obj1",
            "class_name": "cup",
            "bounding_box": [10.0, 20.0, 30.0], # Only 3
            "visibility_status": "CLEARLY_VISIBLE",
            "object_state": "CONFIRMED_PRESENT",
        }
        entry, errs = ObjectAnnotationEntry.from_dict(raw_ann, scope="[test]")
        self.assertIsNone(entry.bounding_box)
        self.assertTrue(any("must have exactly 4 coordinates" in e for e in errs))

    def test_06_non_numeric_coordinates(self) -> None:
        """Requirement 2: Non-numeric coordinates are rejected."""
        raw_ann = {
            "object_id": "obj1",
            "class_name": "cup",
            "bounding_box": [10.0, "twenty", 50.0, 60.0],
            "visibility_status": "CLEARLY_VISIBLE",
            "object_state": "CONFIRMED_PRESENT",
        }
        entry, errs = ObjectAnnotationEntry.from_dict(raw_ann, scope="[test]")
        self.assertIsNone(entry.bounding_box)
        self.assertTrue(any("must be numeric" in e for e in errs))

    def test_07_nan_or_infinite_coordinates(self) -> None:
        """Requirement 2: NaN or Inf coordinates are rejected."""
        for val in [float("nan"), float("inf"), float("-inf")]:
            raw_ann = {
                "object_id": "obj1",
                "class_name": "cup",
                "bounding_box": [10.0, val, 50.0, 60.0],
                "visibility_status": "CLEARLY_VISIBLE",
                "object_state": "CONFIRMED_PRESENT",
            }
            entry, errs = ObjectAnnotationEntry.from_dict(raw_ann, scope="[test]")
            self.assertIsNone(entry.bounding_box)
            self.assertTrue(any("is non-finite" in e for e in errs))

    def test_08_reversed_or_zero_area_bounding_boxes(self) -> None:
        """Requirement 2: Reversed (x_min >= x_max) or zero-area boxes are rejected."""
        # Zero area x
        raw_ann1 = {
            "object_id": "obj1", "class_name": "cup",
            "bounding_box": [50.0, 20.0, 50.0, 60.0],
            "visibility_status": "CLEARLY_VISIBLE", "object_state": "CONFIRMED_PRESENT",
        }
        _, errs1 = ObjectAnnotationEntry.from_dict(raw_ann1, scope="[test]")
        self.assertTrue(any("x_min (50.0) must be strictly less than x_max (50.0)" in e for e in errs1))

        # Reversed y
        raw_ann2 = {
            "object_id": "obj1", "class_name": "cup",
            "bounding_box": [10.0, 80.0, 50.0, 20.0],
            "visibility_status": "CLEARLY_VISIBLE", "object_state": "CONFIRMED_PRESENT",
        }
        _, errs2 = ObjectAnnotationEntry.from_dict(raw_ann2, scope="[test]")
        self.assertTrue(any("y_min (80.0) must be strictly less than y_max (20.0)" in e for e in errs2))

    def test_09_empty_object_id(self) -> None:
        """Requirement 2: Empty object_id is rejected."""
        raw_ann = {
            "object_id": "", "class_name": "cup",
            "visibility_status": "CLEARLY_VISIBLE", "object_state": "CONFIRMED_PRESENT",
        }
        _, errs = ObjectAnnotationEntry.from_dict(raw_ann, scope="[test]")
        self.assertTrue(any("Missing or empty required 'object_id'" in e for e in errs))

    def test_10_empty_class_name(self) -> None:
        """Requirement 2: Empty class_name is rejected."""
        raw_ann = {
            "object_id": "cup_01", "class_name": "   ",
            "visibility_status": "CLEARLY_VISIBLE", "object_state": "CONFIRMED_PRESENT",
        }
        _, errs = ObjectAnnotationEntry.from_dict(raw_ann, scope="[test]")
        self.assertTrue(any("Missing or empty required 'class_name'" in e for e in errs))

    def test_11_invalid_confidence_values(self) -> None:
        """Requirement 2: Confidence outside [0, 1] or non-finite is rejected."""
        for bad_conf in [-0.1, 1.05, float("nan"), "invalid"]:
            raw_ann = {
                "object_id": "cup_01", "class_name": "cup",
                "visibility_status": "CLEARLY_VISIBLE", "object_state": "CONFIRMED_PRESENT",
                "annotation_confidence": bad_conf,
            }
            _, errs = ObjectAnnotationEntry.from_dict(raw_ann, scope="[test]")
            self.assertTrue(any("annotation_confidence" in e for e in errs))

    def test_12_unknown_visibility_and_state_values(self) -> None:
        """Requirement 2: Unknown visibility or state values are rejected, legacy values mapped."""
        # Unknown values rejected
        raw_ann = {
            "object_id": "cup_01", "class_name": "cup",
            "visibility_status": "MAGICAL_TRANSLUCENT", "object_state": "SUPERPOSITION",
        }
        _, errs = ObjectAnnotationEntry.from_dict(raw_ann, scope="[test]")
        self.assertTrue(any("Unknown visibility status 'MAGICAL_TRANSLUCENT'" in e for e in errs))
        self.assertTrue(any("Unknown object state 'SUPERPOSITION'" in e for e in errs))

        # Legacy values mapped cleanly
        raw_legacy = {
            "object_id": "cup_01", "class_name": "cup",
            "visibility_status": "VISIBLE", "object_state": "STATIONARY",
        }
        entry, errs_leg = ObjectAnnotationEntry.from_dict(raw_legacy, scope="[test]")
        self.assertEqual(errs_leg, [])
        self.assertEqual(entry.visibility_status, VisibilityStatus.CLEARLY_VISIBLE.value)
        self.assertEqual(entry.object_state, EpistemicState.CONFIRMED_PRESENT.value)

    def test_13_invalid_declared_dataset_root(self) -> None:
        """Requirement 3: Declared dataset_root that does not exist raises clear error."""
        data = {
            "manifest_version": "2.0",
            "dataset_name": "Test",
            "dataset_root": "non_existent_directory_xyz",
            "sequences": [
                {"sequence_id": "S1", "observations": [{"observation_id": "O1", "timestamp": "2026-11-14T10:00:00Z", "image_path": "images/visit_01.png"}]}
            ],
        }
        with self.assertRaises(ManifestValidationError) as ctx:
            ManifestLoader.parse_dict(data, dataset_root=None, manifest_path=self.temp_dir / "manifest.json")
        self.assertTrue(any("dataset_root does not exist" in e for e in ctx.exception.errors))

    def test_14_explicit_dataset_root_precedence(self) -> None:
        """Requirement 3: Explicit dataset_root argument takes precedence over manifest-declared root."""
        other_root = self.temp_dir / "other_root"
        other_root.mkdir()
        (other_root / "images").mkdir()
        other_img = other_root / "images" / "visit_01.png"
        other_img.write_bytes(b"OTHER_ROOT_IMG")

        data = {
            "manifest_version": "2.0",
            "dataset_name": "Test",
            "dataset_root": "declared_but_ignored", # Not present, but shouldn't matter since explicit passed
            "sequences": [
                {"sequence_id": "S1", "observations": [{"observation_id": "O1", "timestamp": "2026-11-14T10:00:00Z", "image_path": "images/visit_01.png"}]}
            ],
        }
        manifest = ManifestLoader.parse_dict(data, dataset_root=other_root, check_files=True)
        self.assertEqual(manifest.dataset_root, other_root.resolve())
        self.assertEqual(manifest.sequences[0].observations[0].image_path, other_img.resolve())

    def test_15_reference_image_existence(self) -> None:
        """Requirement 4: validate_files checks sequence reference_image."""
        data = {
            "manifest_version": "2.0",
            "dataset_name": "Test",
            "sequences": [
                {
                    "sequence_id": "S1",
                    "reference_image": "images/missing_ref.png",
                    "observations": [{"observation_id": "O1", "timestamp": "2026-11-14T10:00:00Z", "image_path": "images/visit_01.png"}],
                }
            ],
        }
        manifest = ManifestLoader.parse_dict(data, dataset_root=self.temp_dir, check_files=False)
        errors = manifest.validate_files(check_images=True)
        self.assertTrue(any("Sequence reference image not found" in e for e in errors))

    def test_16_depth_and_semantic_file_existence(self) -> None:
        """Requirement 4: validate_files checks depth and semantic files when declared."""
        data = {
            "manifest_version": "2.0",
            "dataset_name": "Test",
            "sequences": [
                {
                    "sequence_id": "S1",
                    "observations": [
                        {
                            "observation_id": "O1",
                            "timestamp": "2026-11-14T10:00:00Z",
                            "image_path": "images/visit_01.png",
                            "depth_path": "images/missing_depth.png",
                            "semantic_path": "images/missing_semantic.png",
                        }
                    ],
                }
            ],
        }
        manifest = ManifestLoader.parse_dict(data, dataset_root=self.temp_dir, check_files=False)
        errors = manifest.validate_files(check_images=False, check_depth=True, check_semantics=True)
        self.assertTrue(any("Depth map file not found" in e for e in errors))
        self.assertTrue(any("Semantic mask file not found" in e for e in errors))

    def test_17_symlink_escaping_dataset_root(self) -> None:
        """Requirement 3 & 4: Symlinks pointing outside dataset_root trigger path traversal violation."""
        # Create outside file and symlink pointing to it
        outside_dir = Path(tempfile.mkdtemp(prefix="test_outside_"))
        self.addCleanup(lambda: shutil.rmtree(outside_dir, ignore_errors=True))
        outside_file = outside_dir / "target.png"
        outside_file.write_bytes(b"OUTSIDE")

        symlink_path = self.temp_dir / "images" / "symlink_escape.png"
        try:
            os.symlink(outside_file, symlink_path)
        except (OSError, NotImplementedError):
            self.skipTest("Symlinks not supported in current environment/permissions.")

        data = {
            "manifest_version": "2.0",
            "dataset_name": "Test",
            "sequences": [
                {"sequence_id": "S1", "observations": [{"observation_id": "O1", "timestamp": "2026-11-14T10:00:00Z", "image_path": "images/symlink_escape.png"}]}
            ],
        }
        with self.assertRaises(ManifestValidationError) as ctx:
            ManifestLoader.parse_dict(data, dataset_root=self.temp_dir)
        self.assertTrue(any("Path traversal violation detected" in e for e in ctx.exception.errors))

    def test_18_equal_timestamps_preserve_order(self) -> None:
        """Requirement 6: Equal timestamps (t_k == t_k+1) are valid and preserved in declared order."""
        data = {
            "manifest_version": "2.0",
            "dataset_name": "Test",
            "sequences": [
                {
                    "sequence_id": "S1",
                    "observations": [
                        {"observation_id": "cam_left", "timestamp": "2026-11-14T10:00:00Z", "image_path": "images/visit_01.png"},
                        {"observation_id": "cam_right", "timestamp": "2026-11-14T10:00:00Z", "image_path": "images/visit_02.png"},
                    ],
                }
            ],
        }
        manifest = ManifestLoader.parse_dict(data, dataset_root=self.temp_dir, check_files=True)
        seq = manifest.sequences[0]
        self.assertEqual([o.observation_id for o in seq.observations], ["cam_left", "cam_right"])
        self.assertEqual(seq.observations[0].timestamp_dt, seq.observations[1].timestamp_dt)

    def test_19_timezone_offset_chronological_comparison(self) -> None:
        """Requirement 6: Timezone offsets are normalized to UTC for chronological validation."""
        # 10:00 UTC vs 11:30 +02:00 (= 09:30 UTC, which is earlier!)
        data = {
            "manifest_version": "2.0",
            "dataset_name": "Test",
            "sequences": [
                {
                    "sequence_id": "S1",
                    "observations": [
                        {"observation_id": "obs_1", "timestamp": "2026-11-14T10:00:00+00:00", "image_path": "images/visit_01.png"},
                        {"observation_id": "obs_2", "timestamp": "2026-11-14T11:30:00+02:00", "image_path": "images/visit_02.png"},
                    ],
                }
            ],
        }
        with self.assertRaises(ManifestValidationError) as ctx:
            ManifestLoader.parse_dict(data, dataset_root=self.temp_dir)
        self.assertTrue(any("strictly precedes" in e for e in ctx.exception.errors))

    def test_20_provenance_classification_and_unknown(self) -> None:
        """Requirement 5: Explicit provenance is classified; unknown provenance is not physical."""
        # 1. Genuine physical
        d_phys = {"manifest_version": "2.0", "dataset_name": "RealDesk", "provenance": "GENUINE_PHYSICAL",
                  "sequences": [{"sequence_id": "S1", "observations": [{"observation_id": "O1", "timestamp": "2026-11-14T10:00:00Z", "image_path": "images/visit_01.png"}]}]}
        m_phys = ManifestLoader.parse_dict(d_phys, dataset_root=self.temp_dir)
        self.assertEqual(m_phys.provenance, DatasetProvenance.GENUINE_PHYSICAL)
        self.assertTrue(m_phys.is_genuine_physical)
        self.assertFalse(m_phys.is_pilot_dataset)

        # 2. Pilot provenance
        d_pilot = {"manifest_version": "2.0", "dataset_name": "AnyName", "provenance": "EXPLORATORY_PILOT",
                   "sequences": [{"sequence_id": "S1", "observations": [{"observation_id": "O1", "timestamp": "2026-11-14T10:00:00Z", "image_path": "images/visit_01.png"}]}]}
        m_pilot = ManifestLoader.parse_dict(d_pilot, dataset_root=self.temp_dir)
        self.assertEqual(m_pilot.provenance, DatasetProvenance.EXPLORATORY_PILOT)
        self.assertTrue(m_pilot.is_pilot_dataset)
        self.assertFalse(m_pilot.is_genuine_physical)

        # 3. Unknown provenance
        d_unknown = {"manifest_version": "2.0", "dataset_name": "RandomData",
                     "sequences": [{"sequence_id": "S1", "observations": [{"observation_id": "O1", "timestamp": "2026-11-14T10:00:00Z", "image_path": "images/visit_01.png"}]}]}
        m_unknown = ManifestLoader.parse_dict(d_unknown, dataset_root=self.temp_dir)
        self.assertEqual(m_unknown.provenance, DatasetProvenance.UNKNOWN)
        self.assertFalse(m_unknown.is_genuine_physical)
        self.assertFalse(m_unknown.is_pilot_dataset)


if __name__ == "__main__":
    unittest.main()
