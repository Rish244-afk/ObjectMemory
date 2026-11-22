"""
Physical Workbench Benchmark Dataset Model, Validator, and Loader.

Defines the data model, validation suite, and ingestion harness for real photographic
workbench sequences in ObjectMemory Phase 14C.

Scientific Integrity Principles:
1. Does not fabricate physical photographs, bounding boxes, or detector outputs.
2. If genuine photographs are not present on disk, status is marked AWAITING_PHYSICAL_DATA.
3. Explicitly decouples physical visibility (human observation) from epistemic belief state.
4. Distinguishes three distinct absence modes:
   - NOT_VISIBLE_OCCLUDED (object is physically present but blocked by an occluder)
   - INSUFFICIENT_EVIDENCE (object cannot be reliably verified from current viewpoint)
   - CONFIRMED_REMOVED (object was physically taken away from the workspace)
5. Separates fiducial marker calibration (ArUco planar homography) from object detection.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import cv2
import numpy as np

from objectmemory.domain.models import BoundingBox
from objectmemory.domain.states import EpistemicState


class PhysicalVisibility(str, Enum):
    """Physical visibility of an object in a captured photographic frame."""
    VISIBLE = "VISIBLE"
    PARTIALLY_OCCLUDED = "PARTIALLY_OCCLUDED"
    FULLY_OCCLUDED = "FULLY_OCCLUDED"
    ABSENT = "ABSENT"


class AbsenceReason(str, Enum):
    """Specific cause for an object not being fully observable."""
    NONE = "NONE"
    OCCLUSION = "OCCLUSION"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONFIRMED_REMOVED = "CONFIRMED_REMOVED"
    OUT_OF_FRAME = "OUT_OF_FRAME"


class DatasetCaptureStatus(str, Enum):
    """Physical capture status of the dataset or sequence."""
    AWAITING_PHYSICAL_DATA = "AWAITING_PHYSICAL_DATA"
    PARTIAL_CAPTURE = "PARTIAL_CAPTURE"
    COMPLETE = "COMPLETE"


@dataclass
class PhysicalWorkbenchAnnotation:
    """Ground-truth annotation for a single physical object in a visit image."""
    sequence_id: str
    visit_id: str
    object_id: str
    class_label: str
    bbox: Optional[BoundingBox]
    visibility: PhysicalVisibility = PhysicalVisibility.VISIBLE
    absence_reason: AbsenceReason = AbsenceReason.NONE
    expected_state: Optional[EpistemicState] = None
    expected_event: Optional[str] = None
    notes: str = ""
    uncertainty_px: float = 0.0
    malformed_bbox_raw: Optional[List[Any]] = None

    def __post_init__(self) -> None:
        if isinstance(self.visibility, str):
            self.visibility = PhysicalVisibility(self.visibility)
        if isinstance(self.absence_reason, str):
            self.absence_reason = AbsenceReason(self.absence_reason)
        if isinstance(self.expected_state, str):
            self.expected_state = EpistemicState(self.expected_state)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "visit_id": self.visit_id,
            "object_id": self.object_id,
            "class_label": self.class_label,
            "bbox_xyxy": [self.bbox.x_min, self.bbox.y_min, self.bbox.x_max, self.bbox.y_max] if self.bbox else self.malformed_bbox_raw,
            "visibility": self.visibility.value,
            "absence_reason": self.absence_reason.value,
            "expected_state": self.expected_state.value if self.expected_state else None,
            "expected_event": self.expected_event,
            "notes": self.notes,
            "uncertainty_px": self.uncertainty_px,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> PhysicalWorkbenchAnnotation:
        raw_box = d.get("bbox_xyxy") or d.get("bbox")
        bbox = None
        malformed_bbox_raw = None
        if raw_box is not None:
            if isinstance(raw_box, (list, tuple)) and len(raw_box) == 4:
                try:
                    bbox = BoundingBox(
                        x_min=float(raw_box[0]),
                        y_min=float(raw_box[1]),
                        x_max=float(raw_box[2]),
                        y_max=float(raw_box[3]),
                    )
                except (ValueError, TypeError):
                    malformed_bbox_raw = list(raw_box)
            else:
                malformed_bbox_raw = list(raw_box) if isinstance(raw_box, (list, tuple)) else [raw_box]
        return cls(
            sequence_id=d.get("sequence_id", ""),
            visit_id=d.get("visit_id", ""),
            object_id=d.get("object_id", ""),
            class_label=d.get("class_label") or d.get("class", ""),
            bbox=bbox,
            visibility=PhysicalVisibility(d.get("visibility", "VISIBLE")),
            absence_reason=AbsenceReason(d.get("absence_reason", "NONE")),
            expected_state=EpistemicState(d["expected_state"]) if d.get("expected_state") else None,
            expected_event=d.get("expected_event"),
            notes=d.get("notes", ""),
            uncertainty_px=float(d.get("uncertainty_px", 0.0)),
            malformed_bbox_raw=malformed_bbox_raw,
        )


@dataclass
class PhysicalWorkbenchVisit:
    """A single observation visit in a physical workbench scenario."""
    visit_id: str
    visit_index: int
    image_relative_path: str
    image_absolute_path: str
    expected_dimensions: Tuple[int, int]  # (width, height)
    timestamp: Optional[str] = None
    camera_notes: str = ""
    scenario_description: str = ""
    expected_event: Optional[str] = None
    expected_events: List[str] = field(default_factory=list)
    annotations: List[PhysicalWorkbenchAnnotation] = field(default_factory=list)
    aruco_markers_expected: bool = False

    def is_image_present(self) -> bool:
        return os.path.exists(self.image_absolute_path) and os.path.isfile(self.image_absolute_path)

    def load_image(self) -> Optional[np.ndarray]:
        if not self.is_image_present():
            return None
        img = cv2.imread(self.image_absolute_path)
        return img

    def to_dict(self) -> Dict[str, Any]:
        return {
            "visit_id": self.visit_id,
            "visit_index": self.visit_index,
            "image": self.image_relative_path,
            "image_dimensions": [self.expected_dimensions[0], self.expected_dimensions[1]],
            "timestamp": self.timestamp,
            "camera_notes": self.camera_notes,
            "scenario_description": self.scenario_description,
            "expected_event": self.expected_event,
            "expected_events": list(self.expected_events),
            "aruco_markers_expected": self.aruco_markers_expected,
            "annotations": [a.to_dict() for a in self.annotations],
        }


@dataclass
class PhysicalWorkbenchSequence:
    """A 5-visit physical scenario testing a specific visual memory hypothesis."""
    sequence_id: str
    scenario_name: str
    description: str
    visits: List[PhysicalWorkbenchVisit] = field(default_factory=list)

    def get_visit(self, visit_id_or_index: Any) -> PhysicalWorkbenchVisit:
        for v in self.visits:
            if v.visit_id == str(visit_id_or_index) or v.visit_index == visit_id_or_index:
                return v
        raise KeyError(f"Visit '{visit_id_or_index}' not found in sequence {self.sequence_id}")

    def is_complete(self) -> bool:
        return len(self.visits) == 5 and all(v.is_image_present() for v in self.visits)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "scenario_name": self.scenario_name,
            "description": self.description,
            "visits": [v.to_dict() for v in self.visits],
        }


class PhysicalWorkbenchDataset:
    """
    Loader, validator, and evaluator interface for Phase 14C Physical Workbench data.
    """

    CANONICAL_CLASSES: Set[str] = {"cup", "book", "scissors", "bottle", "cell phone"}
    CANONICAL_SCENARIOS: List[str] = [
        "S01_stationary_baseline",
        "S02_temporary_movement_return",
        "S03_persistent_relocation",
        "S04_permanent_removal",
        "S05_partial_occlusion",
        "S06_illumination_stress",
        "S07_viewpoint_perturbation",
    ]

    def __init__(
        self,
        dataset_dir: str,
        manifest_data: Dict[str, Any],
        sequences: Dict[str, PhysicalWorkbenchSequence],
    ) -> None:
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.manifest_data = manifest_data
        self.sequences = sequences

    @property
    def capture_status(self) -> DatasetCaptureStatus:
        total_visits = sum(len(seq.visits) for seq in self.sequences.values())
        if total_visits == 0:
            return DatasetCaptureStatus.AWAITING_PHYSICAL_DATA
        present_images = sum(
            1 for seq in self.sequences.values() for v in seq.visits if v.is_image_present()
        )
        if present_images == 0:
            return DatasetCaptureStatus.AWAITING_PHYSICAL_DATA
        if present_images < total_visits:
            return DatasetCaptureStatus.PARTIAL_CAPTURE
        return DatasetCaptureStatus.COMPLETE

    def get_all_annotated_classes(self) -> Set[str]:
        """Returns the set of all unique class labels annotated across all visits."""
        classes = set()
        for seq in self.sequences.values():
            for v in seq.visits:
                for a in v.annotations:
                    if a.class_label:
                        classes.add(a.class_label)
        return classes

    @classmethod
    def from_real_workbench_dataset(
        cls,
        real_dataset: Any,
    ) -> PhysicalWorkbenchDataset:
        """
        Explicit, non-mutating adapter that converts a RealWorkbenchDataset (e.g. desk_pilot_01)
        into a PhysicalWorkbenchDataset instance for evaluation in Phase 14C.
        Preserves original files, schemas, and annotations completely untouched.
        """
        from objectmemory.datasets.real_dataset import AnnotatedEvent, PhysicalObservation

        seq_id = real_dataset.dataset_name or "desk_pilot_01"
        visits: List[PhysicalWorkbenchVisit] = []

        prior_entity_states: Dict[str, EpistemicState] = {}
        for v_idx, v in enumerate(real_dataset.visits, start=1):
            rel_img = str(v.image_path.relative_to(real_dataset.root_dir)).replace("\\", "/")
            abs_img = str(v.image_path.resolve())

            # Read dimensions from loaded image if available
            img = None
            if v.image_path.exists():
                img = v.load_image()
            dims = (img.shape[1], img.shape[0]) if img is not None else (780, 636)

            # Determine individual entity events and visit-level active events
            visit_entity_events: List[Tuple[str, str]] = []  # (object_id, event_type)
            active_events_set: List[str] = []

            for a in v.annotations:
                ent_id = a.entity_id or a.entity_label
                prior_st = prior_entity_states.get(ent_id)
                curr_st = a.expected_state
                ent_event = "unchanged"

                if a.annotated_event == AnnotatedEvent.RELOCATED:
                    if curr_st == EpistemicState.CONFIRMED_PRESENT or "return" in (a.notes or "").lower():
                        ent_event = "return_to_baseline"
                    else:
                        ent_event = "object_moved"
                elif a.annotated_event == AnnotatedEvent.NEWLY_APPEARED:
                    if prior_st is not None:
                        ent_event = "reappeared"
                    else:
                        ent_event = "newly_appeared"
                elif a.physical_observation == PhysicalObservation.ABSENT:
                    if prior_st == EpistemicState.CONFIRMED_REMOVED and curr_st == EpistemicState.CONFIRMED_REMOVED:
                        ent_event = "continued_absence"
                    elif curr_st == EpistemicState.CONFIRMED_REMOVED:
                        ent_event = "object_removed"
                    elif curr_st == EpistemicState.PENDING_REMOVAL:
                        ent_event = "object_pending_removal"
                    else:
                        ent_event = "object_pending_removal"
                elif a.annotated_event == AnnotatedEvent.UNCHANGED:
                    ent_event = "unchanged"

                visit_entity_events.append((ent_id, ent_event))
                if ent_event not in ("unchanged", "continued_absence"):
                    if ent_event not in active_events_set:
                        active_events_set.append(ent_event)

                # Update prior state tracker for this entity
                if curr_st is not None:
                    prior_entity_states[ent_id] = curr_st

            # Deterministic primary event selection by priority hierarchy
            # Dynamic physical changes take absolute priority over passive absence
            EVENT_PRIORITY = {
                "return_to_baseline": 1,
                "object_moved": 2,
                "reappeared": 3,
                "newly_appeared": 4,
                "object_removed": 5,
                "object_pending_removal": 6,
            }

            if v_idx == 1:
                primary_event = "baseline_established"
            elif not active_events_set:
                primary_event = "baseline_stable"
            else:
                primary_event = min(active_events_set, key=lambda ev: EVENT_PRIORITY.get(ev, 99))

            annotations: List[PhysicalWorkbenchAnnotation] = []
            for a, (ent_id, ent_event) in zip(v.annotations, visit_entity_events):
                vis = PhysicalVisibility(a.physical_observation.value)
                reason = AbsenceReason.CONFIRMED_REMOVED if vis == PhysicalVisibility.ABSENT else AbsenceReason.NONE
                annotations.append(
                    PhysicalWorkbenchAnnotation(
                        sequence_id=seq_id,
                        visit_id=v.visit_id,
                        object_id=ent_id,
                        class_label=a.entity_label,
                        bbox=a.bbox,
                        visibility=vis,
                        absence_reason=reason,
                        expected_state=a.expected_state,
                        expected_event=ent_event,
                        notes=a.notes,
                        uncertainty_px=15.0,
                    )
                )

            visits.append(
                PhysicalWorkbenchVisit(
                    visit_id=v.visit_id,
                    visit_index=v_idx,
                    image_relative_path=rel_img,
                    image_absolute_path=abs_img,
                    expected_dimensions=dims,
                    timestamp=v.timestamp,
                    scenario_description=v.session_name,
                    expected_event=primary_event,
                    expected_events=list(active_events_set),
                    annotations=annotations,
                    aruco_markers_expected=False,
                )
            )

        seq = PhysicalWorkbenchSequence(
            sequence_id=seq_id,
            scenario_name=f"Real Pilot Workbench ({len(visits)} Visits)",
            description=real_dataset.description,
            visits=visits,
        )

        manifest_data = {
            "dataset_name": real_dataset.dataset_name,
            "version": "1.0.0-pilot",
            "provenance": real_dataset.metadata.get("image_provenance", "Physical smartphone photography"),
            "dataset_classification": "PILOT_VALIDATION_SEQUENCE",
            "resolution_groups": real_dataset.metadata.get("resolution_groups", {}),
            "sequences": [seq.to_dict()],
        }

        return cls(
            dataset_dir=str(real_dataset.root_dir),
            manifest_data=manifest_data,
            sequences={seq_id: seq},
        )

    @classmethod
    def load(cls, dataset_dir: str, manifest_filename: str = "manifest.json") -> PhysicalWorkbenchDataset:
        """Loads physical workbench dataset from directory with manifest or directly from manifest path."""
        abs_target = os.path.abspath(dataset_dir)
        if os.path.isfile(abs_target):
            manifest_path = abs_target
            abs_dir = os.path.dirname(abs_target)
        else:
            abs_dir = abs_target
            manifest_path = os.path.join(abs_dir, manifest_filename)
            if not os.path.exists(manifest_path):
                manifest_path = os.path.join(abs_dir, "annotations", manifest_filename)
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found in {dataset_dir} (expected {manifest_filename} or annotations/{manifest_filename})")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        # Explicit Adapter for legacy RealWorkbenchDataset format (e.g. data/real_workbench/manifest.json)
        if "visits" in manifest_data and "sequences" not in manifest_data:
            from objectmemory.datasets.real_dataset import RealWorkbenchDataset
            real_ds = RealWorkbenchDataset.load_from_directory(Path(abs_dir))
            return cls.from_real_workbench_dataset(real_ds)

        default_dims = tuple(manifest_data.get("default_image_dimensions", [1920, 1080]))
        sequences: Dict[str, PhysicalWorkbenchSequence] = {}

        for s_dict in manifest_data.get("sequences", []):
            s_id = s_dict["sequence_id"]
            s_name = s_dict.get("scenario_name", s_id)
            s_desc = s_dict.get("description", "")
            visits: List[PhysicalWorkbenchVisit] = []

            for v_dict in s_dict.get("visits", []):
                v_id = v_dict["visit_id"]
                v_idx = int(v_dict["visit_index"])
                rel_img = v_dict.get("image", f"images/{s_id}_{v_id}.jpg")
                abs_img = os.path.join(abs_dir, rel_img.replace("/", os.sep))
                dims = tuple(v_dict.get("image_dimensions", default_dims))

                annotations: List[PhysicalWorkbenchAnnotation] = []
                for ann_dict in v_dict.get("annotations", []):
                    ann_dict_copy = dict(ann_dict)
                    ann_dict_copy["sequence_id"] = s_id
                    ann_dict_copy["visit_id"] = v_id
                    annotations.append(PhysicalWorkbenchAnnotation.from_dict(ann_dict_copy))

                visits.append(
                    PhysicalWorkbenchVisit(
                        visit_id=v_id,
                        visit_index=v_idx,
                        image_relative_path=rel_img,
                        image_absolute_path=abs_img,
                        expected_dimensions=(int(dims[0]), int(dims[1])),
                        timestamp=v_dict.get("timestamp"),
                        camera_notes=v_dict.get("camera_notes", ""),
                        scenario_description=v_dict.get("scenario_description", ""),
                        expected_event=v_dict.get("expected_event"),
                        expected_events=v_dict.get("expected_events", [v_dict["expected_event"]] if v_dict.get("expected_event") else []),
                        annotations=annotations,
                        aruco_markers_expected=bool(v_dict.get("aruco_markers_expected", False)),
                    )
                )

            sequences[s_id] = PhysicalWorkbenchSequence(
                sequence_id=s_id,
                scenario_name=s_name,
                description=s_desc,
                visits=visits,
            )

        return cls(dataset_dir=abs_dir, manifest_data=manifest_data, sequences=sequences)

    def validate(self) -> Dict[str, Any]:
        """
        Validates all sequences, visits, annotations, and images against strict scientific rules.

        Returns:
            Structured validation dictionary with status, errors, warnings, and counts.
        """
        errors: List[str] = []
        warnings: List[str] = []

        total_sequences = len(self.sequences)
        total_visits = 0
        total_annotations = 0
        images_found = 0
        images_missing = 0

        # Check required scenarios
        for expected_seq in self.CANONICAL_SCENARIOS:
            if expected_seq not in self.sequences:
                warnings.append(f"Canonical scenario '{expected_seq}' not defined in manifest.")

        for seq_id, seq in self.sequences.items():
            visit_indices = set()
            visit_ids = set()

            if len(seq.visits) != 5:
                warnings.append(
                    f"Sequence '{seq_id}' has {len(seq.visits)} visits; standard protocol expects 5 visits."
                )

            last_idx = -1
            for visit in seq.visits:
                total_visits += 1

                # 1. Non-chronological or duplicate check
                if visit.visit_index in visit_indices:
                    errors.append(f"Sequence '{seq_id}' has duplicate visit_index: {visit.visit_index}")
                if visit.visit_id in visit_ids:
                    errors.append(f"Sequence '{seq_id}' has duplicate visit_id: '{visit.visit_id}'")
                if visit.visit_index <= last_idx:
                    errors.append(
                        f"Non-chronological visit order in '{seq_id}': index {visit.visit_index} follows {last_idx}"
                    )
                last_idx = visit.visit_index
                visit_indices.add(visit.visit_index)
                visit_ids.add(visit.visit_id)

                # 2. Image existence and dimension verification
                if not visit.is_image_present():
                    images_missing += 1
                else:
                    images_found += 1
                    img = visit.load_image()
                    if img is None:
                        errors.append(f"Unreadable image file in '{seq_id}' visit '{visit.visit_id}': {visit.image_absolute_path}")
                    else:
                        h, w = img.shape[:2]
                        exp_w, exp_h = visit.expected_dimensions
                        if (w, h) != (exp_w, exp_h):
                            errors.append(
                                f"Dimension mismatch in '{seq_id}' visit '{visit.visit_id}': image is ({w}, {h}), expected ({exp_w}, {exp_h})"
                            )

                # 3. Annotation checks
                for ann in visit.annotations:
                    total_annotations += 1

                    # Canonical class verification
                    if ann.class_label not in self.CANONICAL_CLASSES:
                        warnings.append(
                            f"Non-canonical class '{ann.class_label}' in '{seq_id}' visit '{visit.visit_id}'. Standard classes: {sorted(list(self.CANONICAL_CLASSES))}"
                        )

                    # Bounding box bounds check
                    if ann.malformed_bbox_raw is not None:
                        errors.append(
                            f"Malformed inverted bbox for '{ann.object_id}' in '{seq_id}' visit '{visit.visit_id}': "
                            f"{ann.malformed_bbox_raw}"
                        )
                    elif ann.bbox is not None:
                        b = ann.bbox
                        exp_w, exp_h = visit.expected_dimensions
                        if b.x_min < 0 or b.y_min < 0 or b.x_max > exp_w or b.y_max > exp_h:
                            errors.append(
                                f"Out-of-bounds bbox for '{ann.object_id}' in '{seq_id}' visit '{visit.visit_id}': "
                                f"[{b.x_min}, {b.y_min}, {b.x_max}, {b.y_max}] exceeds bounds ({exp_w}, {exp_h})"
                            )
                        if b.x_min >= b.x_max or b.y_min >= b.y_max:
                            errors.append(
                                f"Malformed inverted bbox for '{ann.object_id}' in '{seq_id}' visit '{visit.visit_id}': "
                                f"[{b.x_min}, {b.y_min}, {b.x_max}, {b.y_max}]"
                            )

                    # Contradiction check: physical visibility vs absence reason
                    if ann.visibility == PhysicalVisibility.VISIBLE and ann.absence_reason != AbsenceReason.NONE:
                        errors.append(
                            f"Contradiction in '{seq_id}' visit '{visit.visit_id}' for '{ann.object_id}': "
                            f"visibility=VISIBLE but absence_reason={ann.absence_reason.value}"
                        )

                    # Contradiction check: ABSENT vs bbox present
                    if ann.visibility == PhysicalVisibility.ABSENT and ann.bbox is not None:
                        errors.append(
                            f"Contradiction in '{seq_id}' visit '{visit.visit_id}' for '{ann.object_id}': "
                            f"visibility=ABSENT but non-null bounding box specified"
                        )

                    # Contradiction check: VISIBLE with expected_state CONFIRMED_REMOVED
                    if ann.visibility == PhysicalVisibility.VISIBLE and ann.expected_state == EpistemicState.CONFIRMED_REMOVED:
                        errors.append(
                            f"Contradiction in '{seq_id}' visit '{visit.visit_id}' for '{ann.object_id}': "
                            f"visibility=VISIBLE but expected_state=CONFIRMED_REMOVED"
                        )

        # Status resolution
        overall_status = "PASSED" if not errors else "FAILED"
        capture_state = self.capture_status

        return {
            "dataset_name": self.manifest_data.get("dataset_name", "Physical Workbench Benchmark v1"),
            "dataset_version": self.manifest_data.get("version", "1.0.0"),
            "status": overall_status,
            "capture_status": capture_state.value,
            "provenance": self.manifest_data.get("provenance", "Physical laboratory capture"),
            "total_sequences": total_sequences,
            "total_visits": total_visits,
            "images_found": images_found,
            "images_missing": images_missing,
            "total_annotations": total_annotations,
            "errors": errors,
            "warnings": warnings,
        }

    def save_validation_report(self, output_path: Optional[str] = None) -> str:
        rep = self.validate()
        if output_path is None:
            output_path = os.path.join(self.dataset_dir, "validation_report.json")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2)
        return output_path


# --- ArUco Marker Calibration Utility ---

def detect_aruco_corner_markers(
    image: np.ndarray,
    dictionary_id: int = cv2.aruco.DICT_4X4_50 if hasattr(cv2, "aruco") else 0,
) -> Dict[str, Any]:
    """
    Detects standard ArUco corner fiducials to verify physical calibration.
    Returns detected marker IDs and their pixel centers/corners.
    """
    if not hasattr(cv2, "aruco"):
        return {
            "supported": False,
            "markers_detected": 0,
            "marker_ids": [],
            "error": "OpenCV ArUco module not available in current environment",
        }

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    corners, ids, rejected = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return {
            "supported": True,
            "markers_detected": 0,
            "marker_ids": [],
            "corners": [],
        }

    detected_ids = [int(i[0]) for i in ids]
    parsed_corners = [c[0].tolist() for c in corners]

    return {
        "supported": True,
        "markers_detected": len(detected_ids),
        "marker_ids": detected_ids,
        "corners": parsed_corners,
    }
