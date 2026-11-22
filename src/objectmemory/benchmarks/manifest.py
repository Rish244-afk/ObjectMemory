"""
Evaluation Dataset Manifest Loader for ObjectMemory.

Provides strict, versioned manifest ingestion, path resolution, timestamp ordering
validation, and ground-truth linking for multi-dataset and physical workbench evaluation.

Guiding Principles:
1. Versioned JSON schema validation (supporting v1.0.0 single-sequence and v2.0 multi-sequence).
2. Explicit path-traversal prevention: all relative paths and symlinks must resolve within dataset_root.
3. Strict chronological integrity: malformed or out-of-order timestamps are never silently reordered.
4. Clean separation of structure parsing from disk file-existence validation.
5. Strict isolation between legacy pilot data and genuine physical validation sequences.
6. Comprehensive validation of bounding boxes, visibility states, and epistemic states against project vocabularies.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from objectmemory.domain.models import BoundingBox
from objectmemory.domain.states import EpistemicState, VisibilityStatus

SUPPORTED_MANIFEST_VERSIONS: Set[str] = {"1.0.0", "1.0", "2.0.0", "2.0"}


class DatasetProvenance(str, Enum):
    """Explicit provenance classification of a benchmark dataset."""
    GENUINE_PHYSICAL = "GENUINE_PHYSICAL"
    SYNTHETIC_GENERATED = "SYNTHETIC_GENERATED"
    EXPLORATORY_PILOT = "EXPLORATORY_PILOT"
    PUBLIC_BENCHMARK = "PUBLIC_BENCHMARK"
    UNKNOWN = "UNKNOWN"


# Accepted canonical visibility vocabulary and legacy aliases
CANONICAL_VISIBILITY_VOCABULARY: Set[str] = {
    v.value for v in VisibilityStatus
}  # CLEARLY_VISIBLE, PARTIALLY_OCCLUDED, HEAVILY_OCCLUDED, OUT_OF_FRAME, UNOBSERVED, UNCERTAIN

LEGACY_VISIBILITY_ALIASES: Dict[str, str] = {
    "VISIBLE": VisibilityStatus.CLEARLY_VISIBLE.value,
    "FULLY_OCCLUDED": VisibilityStatus.HEAVILY_OCCLUDED.value,
    "ABSENT": VisibilityStatus.UNOBSERVED.value,
}

# Accepted canonical epistemic state vocabulary and legacy aliases
CANONICAL_STATE_VOCABULARY: Set[str] = {
    s.value for s in EpistemicState
}  # CONFIRMED_PRESENT, CONFIRMED_MOVED, CONFIRMED_REMOVED, CONFIRMED_ADDED, PENDING_REMOVAL, INSUFFICIENT_EVIDENCE, AMBIGUOUS_IDENTITY

LEGACY_STATE_ALIASES: Dict[str, str] = {
    "MOVED": EpistemicState.CONFIRMED_MOVED.value,
    "MOVED_BACK_TO_BASELINE": EpistemicState.CONFIRMED_PRESENT.value,
    "STATIONARY": EpistemicState.CONFIRMED_PRESENT.value,
    "REMOVED": EpistemicState.CONFIRMED_REMOVED.value,
    "REAPPEARED": EpistemicState.CONFIRMED_PRESENT.value,
    "ADDED": EpistemicState.CONFIRMED_ADDED.value,
}


class ManifestValidationError(ValueError):
    """
    Raised when a dataset manifest violates structural, relational,
    temporal, or file-integrity constraints.
    """

    def __init__(self, errors: Sequence[str], manifest_path: Optional[Path] = None) -> None:
        self.errors = list(errors)
        self.manifest_path = manifest_path
        prefix = f"Manifest at '{manifest_path}' failed validation" if manifest_path else "Manifest validation failed"
        msg = f"{prefix} with {len(self.errors)} error(s):\n" + "\n".join(f"  - {e}" for e in self.errors)
        super().__init__(msg)


@dataclass
class ObjectAnnotationEntry:
    """Ground-truth object annotation linked to an observation."""
    object_id: str
    class_name: str
    bounding_box: Optional[BoundingBox]
    visibility_status: str
    object_state: str
    timestamp: str
    event_id: Optional[str] = None
    annotation_confidence: float = 1.0
    evidence_notes: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Any, scope: str = "") -> Tuple[Optional["ObjectAnnotationEntry"], List[str]]:
        """
        Parses and validates an object annotation entry dictionary.
        Returns (entry, errors). Never allows uncaught KeyError, TypeError, or ValueError to escape.
        """
        errors: List[str] = []

        if not isinstance(data, dict):
            return None, [f"{scope}: Annotation entry must be a dictionary object, got {type(data).__name__}."]

        # 1. Object ID
        raw_obj_id = data.get("object_id")
        if raw_obj_id is None or str(raw_obj_id).strip() == "":
            errors.append(f"{scope}: Missing or empty required 'object_id'.")
            obj_id = ""
        else:
            obj_id = str(raw_obj_id).strip()

        # 2. Class Name
        raw_class = data.get("class_name") or data.get("class_label")
        if raw_class is None or str(raw_class).strip() == "":
            errors.append(f"{scope}: Missing or empty required 'class_name' (or 'class_label').")
            class_name = ""
        else:
            class_name = str(raw_class).strip()

        # 3. Bounding Box Validation
        bbox: Optional[BoundingBox] = None
        raw_box = data.get("bounding_box") or data.get("bbox")
        if raw_box is not None:
            bbox, bbox_errs = cls._parse_bounding_box(raw_box, scope)
            errors.extend(bbox_errs)

        # 4. Visibility Status
        raw_vis = data.get("visibility_status") or data.get("visibility")
        if raw_vis is None or str(raw_vis).strip() == "":
            errors.append(f"{scope}: Missing or empty required 'visibility_status'.")
            vis_status = VisibilityStatus.CLEARLY_VISIBLE.value
        else:
            vis_str = str(raw_vis).strip().upper()
            if vis_str in CANONICAL_VISIBILITY_VOCABULARY:
                vis_status = vis_str
            elif vis_str in LEGACY_VISIBILITY_ALIASES:
                vis_status = LEGACY_VISIBILITY_ALIASES[vis_str]
            else:
                supported = sorted(CANONICAL_VISIBILITY_VOCABULARY.union(LEGACY_VISIBILITY_ALIASES.keys()))
                errors.append(
                    f"{scope}: Unknown visibility status '{raw_vis}'. "
                    f"Must be one of {supported}."
                )
                vis_status = vis_str

        # 5. Object State
        raw_state = data.get("object_state") or data.get("expected_state")
        if raw_state is None or str(raw_state).strip() == "":
            errors.append(f"{scope}: Missing or empty required 'object_state'.")
            obj_state = EpistemicState.CONFIRMED_PRESENT.value
        else:
            state_str = str(raw_state).strip().upper()
            if state_str in CANONICAL_STATE_VOCABULARY:
                obj_state = state_str
            elif state_str in LEGACY_STATE_ALIASES:
                obj_state = LEGACY_STATE_ALIASES[state_str]
            else:
                supported = sorted(CANONICAL_STATE_VOCABULARY.union(LEGACY_STATE_ALIASES.keys()))
                errors.append(
                    f"{scope}: Unknown object state '{raw_state}'. "
                    f"Must be one of {supported}."
                )
                obj_state = state_str

        # 6. Timestamp
        raw_ts = data.get("timestamp")
        ts_str = str(raw_ts).strip() if raw_ts is not None else ""

        # 7. Confidence Score
        raw_conf = data.get("annotation_confidence", 1.0)
        try:
            conf = float(raw_conf)
            if not math.isfinite(conf) or conf < 0.0 or conf > 1.0:
                errors.append(
                    f"{scope}: 'annotation_confidence' must be a finite number in [0.0, 1.0], got {raw_conf}."
                )
                conf = 1.0
        except (ValueError, TypeError):
            errors.append(
                f"{scope}: 'annotation_confidence' must be numeric in [0.0, 1.0], got {raw_conf}."
            )
            conf = 1.0

        # 8. Optional Fields
        event_id = data.get("event_id") or data.get("expected_event")
        if event_id is not None and not isinstance(event_id, (str, int)):
            errors.append(f"{scope}: 'event_id' must be a string or integer if provided, got {type(event_id).__name__}.")

        notes = data.get("evidence_notes") or data.get("notes")
        if notes is not None and not isinstance(notes, str):
            errors.append(f"{scope}: 'evidence_notes' must be a string if provided, got {type(notes).__name__}.")

        # 9. Retain unrecognized metadata
        reserved = {
            "object_id", "class_name", "class_label", "bounding_box", "bbox",
            "visibility_status", "visibility", "object_state", "expected_state",
            "timestamp", "event_id", "expected_event", "annotation_confidence",
            "evidence_notes", "notes",
        }
        meta = {k: v for k, v in data.items() if k not in reserved}

        entry = cls(
            object_id=obj_id,
            class_name=class_name,
            bounding_box=bbox,
            visibility_status=vis_status,
            object_state=obj_state,
            timestamp=ts_str,
            event_id=str(event_id) if event_id is not None else None,
            annotation_confidence=conf,
            evidence_notes=str(notes) if notes is not None else None,
            metadata=meta,
        )
        return entry, errors

    @staticmethod
    def _parse_bounding_box(raw_box: Any, scope: str) -> Tuple[Optional[BoundingBox], List[str]]:
        """Validates bounding box format: exactly 4 finite numbers, ordering x_min < x_max and y_min < y_max."""
        errors: List[str] = []
        coords: List[float] = []

        if isinstance(raw_box, (list, tuple)):
            if len(raw_box) != 4:
                return None, [f"{scope}: Bounding box sequence must have exactly 4 coordinates, got {len(raw_box)}."]
            for idx, c in enumerate(raw_box):
                if isinstance(c, bool):
                    return None, [f"{scope}: Bounding box coordinate [{idx}] must be numeric, got boolean {c}."]
                try:
                    val = float(c)
                    if not math.isfinite(val):
                        return None, [f"{scope}: Bounding box coordinate [{idx}] is non-finite: {c}."]
                    coords.append(val)
                except (ValueError, TypeError):
                    return None, [f"{scope}: Bounding box coordinate [{idx}] must be numeric, got {c!r}."]
        elif isinstance(raw_box, dict):
            keys = ["x_min", "y_min", "x_max", "y_max"]
            alt_keys = ["xmin", "ymin", "xmax", "ymax"]
            for k, alt in zip(keys, alt_keys):
                c = raw_box.get(k) if k in raw_box else raw_box.get(alt)
                if c is None:
                    return None, [f"{scope}: Bounding box dictionary missing required key '{k}'."]
                if isinstance(c, bool):
                    return None, [f"{scope}: Bounding box coordinate '{k}' must be numeric, got boolean {c}."]
                try:
                    val = float(c)
                    if not math.isfinite(val):
                        return None, [f"{scope}: Bounding box coordinate '{k}' is non-finite: {c}."]
                    coords.append(val)
                except (ValueError, TypeError):
                    return None, [f"{scope}: Bounding box coordinate '{k}' must be numeric, got {c!r}."]
        else:
            return None, [f"{scope}: Bounding box must be a 4-element sequence or coordinate dictionary, got {type(raw_box).__name__}."]

        x_min, y_min, x_max, y_max = coords
        if x_min >= x_max:
            errors.append(f"{scope}: Invalid bounding box geometry: x_min ({x_min}) must be strictly less than x_max ({x_max}).")
        if y_min >= y_max:
            errors.append(f"{scope}: Invalid bounding box geometry: y_min ({y_min}) must be strictly less than y_max ({y_max}).")

        if errors:
            return None, errors

        try:
            return BoundingBox(x_min, y_min, x_max, y_max), []
        except Exception as exc:
            return None, [f"{scope}: Failed to instantiate BoundingBox: {exc}"]


@dataclass
class ObservationEntry:
    """A discrete visual observation within a continuous sequence."""
    observation_id: str
    sequence_id: str
    timestamp: str
    timestamp_dt: datetime
    image_path: Path
    raw_image_path: str
    depth_path: Optional[Path] = None
    raw_depth_path: Optional[str] = None
    semantic_path: Optional[Path] = None
    raw_semantic_path: Optional[str] = None
    camera_pose: Optional[Dict[str, Any]] = None
    source_annotation_path: Optional[Path] = None
    raw_source_annotation_path: Optional[str] = None
    split: str = "TEST"
    objects: List[ObjectAnnotationEntry] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SequenceEntry:
    """An ordered continuous sequence of observations within an environment."""
    sequence_id: str
    environment_id: str
    dataset_name: str
    observations: List[ObservationEntry]
    challenge_mode: Optional[str] = None
    reference_image: Optional[Path] = None
    raw_reference_image: Optional[str] = None
    calibration_file: Optional[Path] = None
    raw_calibration_file: Optional[str] = None
    ground_truth_file: Optional[Path] = None
    raw_ground_truth_file: Optional[str] = None
    coordinate_convention: str = "PIXEL_XMIN_YMIN_XMAX_YMAX"
    split: str = "TEST"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.observations)

    def get_observation(self, observation_id: str) -> Optional[ObservationEntry]:
        for obs in self.observations:
            if obs.observation_id == observation_id:
                return obs
        return None


@dataclass
class DatasetManifest:
    """Top-level representation of a loaded and validated dataset manifest."""
    manifest_version: str
    dataset_name: str
    dataset_root: Path
    sequences: List[SequenceEntry]
    provenance: DatasetProvenance = DatasetProvenance.UNKNOWN
    manifest_path: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def sequence_count(self) -> int:
        return len(self.sequences)

    @property
    def total_observations(self) -> int:
        return sum(len(seq) for seq in self.sequences)

    @property
    def is_pilot_dataset(self) -> bool:
        """Indicates whether this manifest represents an exploratory pilot rather than physical benchmark."""
        if self.provenance == DatasetProvenance.EXPLORATORY_PILOT:
            return True
        d_lower = self.dataset_name.lower()
        if "pilot" in d_lower:
            return True
        for seq in self.sequences:
            if "pilot" in seq.sequence_id.lower() or "pilot" in seq.environment_id.lower():
                return True
        return False

    @property
    def is_genuine_physical(self) -> bool:
        """Strict check asserting verified genuine physical-world capture provenance."""
        return self.provenance == DatasetProvenance.GENUINE_PHYSICAL

    def get_sequence(self, sequence_id: str) -> Optional[SequenceEntry]:
        for seq in self.sequences:
            if seq.sequence_id == sequence_id:
                return seq
        return None

    def validate_files(
        self,
        check_images: bool = True,
        check_annotations: bool = True,
        check_calibration: bool = True,
        check_depth: bool = True,
        check_semantics: bool = True,
    ) -> List[str]:
        """
        Validates the physical on-disk existence of referenced files.
        Covers observation images, reference images, calibration, ground-truth,
        annotation JSON files, depth maps, and semantic masks.

        Note: Verifies on-disk file existence only. Does not assert semantic validity
        of external annotation file contents.
        """
        missing_errors: List[str] = []

        for seq in self.sequences:
            s_id = seq.sequence_id

            if check_images and seq.reference_image is not None:
                if not seq.reference_image.is_file():
                    missing_errors.append(
                        f"[{s_id}] Sequence reference image not found on disk: '{seq.reference_image}' "
                        f"(raw: '{seq.raw_reference_image}')"
                    )

            if check_calibration and seq.calibration_file is not None:
                if not seq.calibration_file.is_file():
                    missing_errors.append(
                        f"[{s_id}] Calibration file not found on disk: '{seq.calibration_file}' "
                        f"(raw: '{seq.raw_calibration_file}')"
                    )

            if check_annotations and seq.ground_truth_file is not None:
                if not seq.ground_truth_file.is_file():
                    missing_errors.append(
                        f"[{s_id}] Sequence ground-truth file not found on disk: '{seq.ground_truth_file}' "
                        f"(raw: '{seq.raw_ground_truth_file}')"
                    )

            for obs in seq.observations:
                o_id = obs.observation_id
                if check_images:
                    if not obs.image_path.is_file():
                        missing_errors.append(
                            f"[{s_id} / {o_id}] Image file not found on disk: '{obs.image_path}' "
                            f"(raw: '{obs.raw_image_path}')"
                        )

                if check_annotations and obs.source_annotation_path is not None:
                    if not obs.source_annotation_path.is_file():
                        missing_errors.append(
                            f"[{s_id} / {o_id}] Observation annotation file not found on disk: "
                            f"'{obs.source_annotation_path}' (raw: '{obs.raw_source_annotation_path}')"
                        )

                if check_depth and obs.depth_path is not None:
                    if not obs.depth_path.is_file():
                        missing_errors.append(
                            f"[{s_id} / {o_id}] Depth map file not found on disk: '{obs.depth_path}' "
                            f"(raw: '{obs.raw_depth_path}')"
                        )

                if check_semantics and obs.semantic_path is not None:
                    if not obs.semantic_path.is_file():
                        missing_errors.append(
                            f"[{s_id} / {o_id}] Semantic mask file not found on disk: '{obs.semantic_path}' "
                            f"(raw: '{obs.raw_semantic_path}')"
                        )

        return missing_errors


class ManifestLoader:
    """
    Parser and validator for ObjectMemory evaluation manifests.
    Enforces path safety, timestamp monotonic ordering, identifier uniqueness,
    and schema version compatibility.
    """

    @classmethod
    def load(
        cls,
        manifest_path: Union[str, Path],
        dataset_root: Optional[Union[str, Path]] = None,
        check_files: bool = False,
    ) -> DatasetManifest:
        """
        Loads and validates a dataset manifest from a JSON file.

        Args:
            manifest_path: Path to the JSON manifest file.
            dataset_root: Optional base directory for relative paths. Takes precedence over manifest-declared root.
            check_files: If True, asserts physical on-disk existence of all referenced files.

        Raises:
            ManifestValidationError: If the manifest structure, version, timestamps, or paths are invalid.
            FileNotFoundError: If manifest_path itself does not exist.
        """
        path = Path(manifest_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Manifest file does not exist: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ManifestValidationError([f"Malformed JSON syntax: {exc}"], manifest_path=path) from exc

        return cls.parse_dict(raw_data, dataset_root=dataset_root, manifest_path=path, check_files=check_files)

    @classmethod
    def parse_dict(
        cls,
        data: Dict[str, Any],
        dataset_root: Optional[Union[str, Path]] = None,
        manifest_path: Optional[Path] = None,
        check_files: bool = False,
    ) -> DatasetManifest:
        """Parses and validates in-memory dictionary representation of a manifest."""
        errors: List[str] = []

        if not isinstance(data, dict):
            raise ManifestValidationError(["Manifest top-level content must be a JSON object."], manifest_path=manifest_path)

        # 1. Version Validation
        version = str(data.get("manifest_version") or data.get("schema_version") or "").strip()
        if not version:
            errors.append("Manifest lacks required version field ('manifest_version' or 'schema_version').")
        elif version not in SUPPORTED_MANIFEST_VERSIONS:
            errors.append(
                f"Unsupported manifest version '{version}'. Supported versions: {sorted(SUPPORTED_MANIFEST_VERSIONS)}"
            )

        # 2. Dataset Name
        dataset_name = str(data.get("dataset_name") or "").strip()
        if not dataset_name:
            errors.append("Manifest lacks required 'dataset_name' field.")

        # 3. Deterministic Dataset-Root Resolution
        root, root_err = cls._resolve_dataset_root(dataset_root, data.get("dataset_root"), manifest_path)
        if root_err:
            errors.append(root_err)
            # Use fallback to collect remaining errors rather than crashing immediately
            root = manifest_path.parent.resolve() if manifest_path else Path.cwd().resolve()

        # 4. Provenance Classification
        provenance = cls._determine_provenance(data)

        # 5. Extract sequences (support both multi-sequence v2.0 and single-sequence v1.0)
        sequences_raw: List[Dict[str, Any]] = []
        if "sequences" in data:
            if not isinstance(data["sequences"], list):
                errors.append("Field 'sequences' must be a list of sequence objects.")
            else:
                sequences_raw = data["sequences"]
        elif "observations" in data:
            # Single-sequence v1.0 schema: adapt to sequence container
            single_seq = {
                "sequence_id": data.get("sequence_id", "default_sequence"),
                "environment_id": data.get("environment_id", "default_env"),
                "dataset_name": dataset_name,
                "observations": data.get("observations", []),
                "coordinate_convention": data.get("coordinate_convention", "PIXEL_XMIN_YMIN_XMAX_YMAX"),
                "metadata": data.get("metadata", {}),
            }
            sequences_raw = [single_seq]
        else:
            errors.append("Manifest contains neither 'sequences' nor 'observations'.")

        if not sequences_raw and not errors:
            errors.append("Manifest contains zero sequences.")

        # 6. Parse and validate sequences
        parsed_sequences: List[SequenceEntry] = []
        seen_seq_ids: Set[str] = set()
        global_obs_ids: Set[str] = set()

        for s_idx, s_data in enumerate(sequences_raw):
            if not isinstance(s_data, dict):
                errors.append(f"Sequence at index {s_idx} must be a dictionary object.")
                continue

            seq_id = str(s_data.get("sequence_id") or "").strip()
            if not seq_id:
                errors.append(f"Sequence at index {s_idx} is missing required 'sequence_id'.")
                seq_id = f"unknown_seq_{s_idx}"
            elif seq_id in seen_seq_ids:
                errors.append(f"Duplicate sequence_id detected: '{seq_id}'. Sequence IDs must be unique.")
            seen_seq_ids.add(seq_id)

            env_id = str(s_data.get("environment_id") or data.get("environment_id") or "default_env").strip()
            coord_conv = str(s_data.get("coordinate_convention") or data.get("coordinate_convention") or "PIXEL_XMIN_YMIN_XMAX_YMAX")
            challenge = s_data.get("challenge_mode")
            split = str(s_data.get("split") or data.get("split") or "TEST").strip().upper()

            # Sequence-level relative path resolution
            ref_img, raw_ref_img, errs_ref = cls._resolve_path(s_data.get("reference_image"), root, f"[{seq_id}] reference_image")
            errors.extend(errs_ref)

            calib_file, raw_calib, errs_cal = cls._resolve_path(s_data.get("calibration_file"), root, f"[{seq_id}] calibration_file")
            errors.extend(errs_cal)

            gt_file, raw_gt, errs_gt = cls._resolve_path(s_data.get("ground_truth_file"), root, f"[{seq_id}] ground_truth_file")
            errors.extend(errs_gt)

            # Observations parsing
            obs_raw = s_data.get("observations", [])
            if not isinstance(obs_raw, list):
                errors.append(f"[{seq_id}] Field 'observations' must be a list.")
                obs_raw = []
            elif len(obs_raw) == 0:
                errors.append(f"[{seq_id}] Sequence contains 0 observations. Empty sequences are disallowed.")

            parsed_obs: List[ObservationEntry] = []
            seq_obs_ids: Set[str] = set()
            prev_dt: Optional[datetime] = None
            prev_obs_id: Optional[str] = None

            for o_idx, o_data in enumerate(obs_raw):
                if not isinstance(o_data, dict):
                    errors.append(f"[{seq_id}] Observation at index {o_idx} must be a dictionary object.")
                    continue

                o_id = str(o_data.get("observation_id") or "").strip()
                if not o_id:
                    errors.append(f"[{seq_id}] Observation at index {o_idx} lacks required 'observation_id'.")
                    o_id = f"unknown_obs_{o_idx}"
                else:
                    if o_id in seq_obs_ids:
                        errors.append(f"[{seq_id}] Duplicate observation_id detected: '{o_id}'.")
                    if o_id in global_obs_ids:
                        errors.append(f"Duplicate observation_id across dataset detected: '{o_id}'.")
                    seq_obs_ids.add(o_id)
                    global_obs_ids.add(o_id)

                # Mandatory image path validation
                raw_img = o_data.get("image_path")
                if raw_img is None or str(raw_img).strip() == "":
                    errors.append(f"[{seq_id} / {o_id}] Missing or empty required 'image_path'.")
                    img_path = None
                    raw_img_str = ""
                else:
                    img_path, raw_img_str, img_errs = cls._resolve_path(raw_img, root, f"[{seq_id} / {o_id}] image_path")
                    errors.extend(img_errs)

                # Timestamp validation and strict chronological ordering
                raw_ts = o_data.get("timestamp")
                dt: Optional[datetime] = None
                if raw_ts is None or str(raw_ts).strip() == "":
                    errors.append(f"[{seq_id} / {o_id}] Missing required 'timestamp'.")
                else:
                    dt = cls._parse_timestamp(str(raw_ts))
                    if dt is None:
                        errors.append(
                            f"[{seq_id} / {o_id}] Invalid timestamp '{raw_ts}'. "
                            f"Must be a valid ISO-8601 string or Unix epoch float/int."
                        )
                    else:
                        if prev_dt is not None and dt < prev_dt:
                            errors.append(
                                f"[{seq_id}] Invalid sequence ordering: observation '{o_id}' (timestamp {raw_ts}) "
                                f"strictly precedes '{prev_obs_id}' (timestamp {prev_dt.isoformat()}). "
                                f"Observations must not be non-chronological."
                            )
                        prev_dt = dt
                        prev_obs_id = o_id

                # Auxiliary paths
                depth_path, raw_depth, depth_errs = cls._resolve_path(o_data.get("depth_path"), root, f"[{seq_id} / {o_id}] depth_path")
                errors.extend(depth_errs)

                sem_path, raw_sem, sem_errs = cls._resolve_path(o_data.get("semantic_path"), root, f"[{seq_id} / {o_id}] semantic_path")
                errors.extend(sem_errs)

                ann_path, raw_ann, ann_errs = cls._resolve_path(o_data.get("source_annotation_path"), root, f"[{seq_id} / {o_id}] source_annotation_path")
                errors.extend(ann_errs)

                # Objects list validation
                parsed_objects: List[ObjectAnnotationEntry] = []
                objs_raw = o_data.get("objects", [])
                if isinstance(objs_raw, list):
                    for a_idx, obj_data in enumerate(objs_raw):
                        ann_entry, ann_errs = ObjectAnnotationEntry.from_dict(
                            obj_data, scope=f"[{seq_id} / {o_id} / obj_{a_idx}]"
                        )
                        errors.extend(ann_errs)
                        if ann_entry is not None:
                            parsed_objects.append(ann_entry)
                elif objs_raw is not None:
                    errors.append(f"[{seq_id} / {o_id}] Field 'objects' must be a list if provided.")

                obs_split = str(o_data.get("split") or split).strip().upper()
                reserved_obs = {
                    "observation_id", "sequence_id", "timestamp", "image_path",
                    "depth_path", "semantic_path", "camera_pose", "source_annotation_path",
                    "split", "objects",
                }
                obs_meta = {k: v for k, v in o_data.items() if k not in reserved_obs}

                # Construct ObservationEntry (placeholder datetime/path if malformed, preserved in sequence order)
                parsed_obs.append(
                    ObservationEntry(
                        observation_id=o_id,
                        sequence_id=seq_id,
                        timestamp=str(raw_ts) if raw_ts is not None else "",
                        timestamp_dt=dt if dt is not None else datetime.fromtimestamp(0, tz=timezone.utc),
                        image_path=img_path if img_path is not None else Path(root / "unresolved_image.png"),
                        raw_image_path=raw_img_str or "",
                        depth_path=depth_path,
                        raw_depth_path=raw_depth,
                        semantic_path=sem_path,
                        raw_semantic_path=raw_sem,
                        camera_pose=o_data.get("camera_pose"),
                        source_annotation_path=ann_path,
                        raw_source_annotation_path=raw_ann,
                        split=obs_split,
                        objects=parsed_objects,
                        metadata=obs_meta,
                    )
                )

            reserved_seq = {
                "sequence_id", "environment_id", "dataset_name", "observations",
                "challenge_mode", "reference_image", "calibration_file", "ground_truth_file",
                "coordinate_convention", "split", "metadata",
            }
            seq_meta = {k: v for k, v in s_data.items() if k not in reserved_seq}
            if "metadata" in s_data and isinstance(s_data["metadata"], dict):
                seq_meta.update(s_data["metadata"])

            parsed_sequences.append(
                SequenceEntry(
                    sequence_id=seq_id,
                    environment_id=env_id,
                    dataset_name=dataset_name,
                    observations=parsed_obs,
                    challenge_mode=str(challenge) if challenge is not None else None,
                    reference_image=ref_img,
                    raw_reference_image=raw_ref_img,
                    calibration_file=calib_file,
                    raw_calibration_file=raw_calib,
                    ground_truth_file=gt_file,
                    raw_ground_truth_file=raw_gt,
                    coordinate_convention=coord_conv,
                    split=split,
                    metadata=seq_meta,
                )
            )

        top_reserved = {
            "manifest_version", "schema_version", "dataset_name", "dataset_root",
            "sequences", "observations", "metadata", "provenance", "dataset_role",
        }
        top_meta = {k: v for k, v in data.items() if k not in top_reserved}
        if "metadata" in data and isinstance(data["metadata"], dict):
            top_meta.update(data["metadata"])

        manifest = DatasetManifest(
            manifest_version=version,
            dataset_name=dataset_name,
            dataset_root=root,
            sequences=parsed_sequences,
            provenance=provenance,
            manifest_path=manifest_path,
            metadata=top_meta,
        )

        if errors:
            raise ManifestValidationError(errors, manifest_path=manifest_path)

        if check_files:
            file_errors = manifest.validate_files()
            if file_errors:
                raise ManifestValidationError(file_errors, manifest_path=manifest_path)

        return manifest

    @classmethod
    def _resolve_dataset_root(
        cls,
        explicit_root: Optional[Union[str, Path]],
        manifest_root: Optional[Any],
        manifest_path: Optional[Path],
    ) -> Tuple[Path, Optional[str]]:
        """
        Deterministically resolves the dataset root following strict precedence:
        1. Explicit argument takes precedence.
        2. Manifest-declared root.
        3. Manifest file parent directory.
        4. Current working directory.
        """
        # Rule 1: Explicit argument takes absolute precedence
        if explicit_root is not None:
            p = Path(explicit_root).resolve()
            if not p.is_dir():
                return p, f"Explicit dataset_root does not exist or is not a directory: '{explicit_root}' (resolved to '{p}')."
            return p, None

        # Rule 2: Manifest-declared root
        if manifest_root is not None and str(manifest_root).strip() != "":
            raw_decl = Path(str(manifest_root).strip())
            if raw_decl.is_absolute():
                candidate = raw_decl.resolve()
            else:
                base = manifest_path.parent if manifest_path is not None else Path.cwd()
                candidate = (base / raw_decl).resolve()

            if not candidate.is_dir():
                return candidate, f"Manifest declared dataset_root does not exist or is not a directory: '{manifest_root}' (resolved to '{candidate}')."
            return candidate, None

        # Rule 3: Manifest file parent directory
        if manifest_path is not None:
            return manifest_path.parent.resolve(), None

        # Rule 4: Fallback to current working directory
        return Path.cwd().resolve(), None

    @classmethod
    def _determine_provenance(cls, data: Dict[str, Any]) -> DatasetProvenance:
        """Determines explicit dataset provenance without fragile substring guessing."""
        raw_prov = data.get("provenance") or data.get("dataset_role")
        if raw_prov is None and isinstance(data.get("metadata"), dict):
            raw_prov = data["metadata"].get("provenance") or data["metadata"].get("dataset_role")

        if raw_prov is not None:
            p_str = str(raw_prov).strip().upper()
            try:
                return DatasetProvenance(p_str)
            except ValueError:
                if "PHYSICAL" in p_str:
                    return DatasetProvenance.GENUINE_PHYSICAL
                if "SYNTHETIC" in p_str or "GENERATED" in p_str:
                    return DatasetProvenance.SYNTHETIC_GENERATED
                if "PILOT" in p_str:
                    return DatasetProvenance.EXPLORATORY_PILOT
                if "PUBLIC" in p_str or "BENCHMARK" in p_str:
                    return DatasetProvenance.PUBLIC_BENCHMARK

        # Unknown provenance must not automatically be represented as verified physical-world data
        return DatasetProvenance.UNKNOWN

    @staticmethod
    def _parse_timestamp(ts_str: str) -> Optional[datetime]:
        """
        Parses ISO-8601 timestamp string or Unix epoch numeric value into UTC-normalized datetime.
        """
        ts_clean = ts_str.strip()
        if ts_clean.endswith("Z"):
            ts_clean = ts_clean[:-1] + "+00:00"

        # Try ISO-8601
        try:
            dt = datetime.fromisoformat(ts_clean)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            pass

        # Try Unix epoch float/integer (documented accepted alternative)
        try:
            val = float(ts_clean)
            return datetime.fromtimestamp(val, tz=timezone.utc)
        except (ValueError, OverflowError, OSError):
            return None

    @classmethod
    def _resolve_path(
        cls,
        rel_str: Optional[Any],
        root: Path,
        context: str,
    ) -> Tuple[Optional[Path], Optional[str], List[str]]:
        """
        Safely resolves a relative path against dataset root.
        Detects directory traversal (e.g. escaping root) and empty paths.
        Also inspects symlinks to ensure targets resolve within dataset root.
        """
        if rel_str is None:
            return None, None, []

        s = str(rel_str).strip()
        if not s:
            return None, None, []

        errors: List[str] = []
        try:
            raw_path = Path(s)
            if raw_path.is_absolute():
                resolved = raw_path.resolve()
            else:
                resolved = (root / raw_path).resolve()

            # Prevent directory traversal escaping dataset root (including symlink targets)
            try:
                resolved.relative_to(root.resolve())
            except ValueError:
                errors.append(
                    f"{context}: Path traversal violation detected: '{s}' resolves outside dataset root '{root}'."
                )
                return None, s, errors

            return resolved, s, errors
        except Exception as exc:
            errors.append(f"{context}: Malformed path '{s}': {exc}")
            return None, s, errors
