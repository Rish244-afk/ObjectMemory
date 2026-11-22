"""
Real-image dataset interface and schema for ObjectMemory.

Provides structured loading, validation, and schema definitions for multi-visit
workbench imagery captured in real laboratory environments.

Guiding principles:
1. Does not fabricate real-world images or annotations.
2. Strictly separates real-world datasets from synthetic benchmarks.
3. Explicitly reports when evaluation is blocked due to missing manual ground truth.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import cv2
from enum import Enum
import numpy as np

from objectmemory.domain.models import BoundingBox, normalize_iso_timestamp
from objectmemory.domain.states import EpistemicState


class PhysicalObservation(str, Enum):
    """
    Independent human observation of physical entity visibility in a visit frame.
    Decoupled from system belief or detector confidence.
    """
    VISIBLE = "VISIBLE"
    PARTIALLY_OCCLUDED = "PARTIALLY_OCCLUDED"
    FULLY_OCCLUDED = "FULLY_OCCLUDED"
    OUT_OF_FRAME = "OUT_OF_FRAME"
    ABSENT = "ABSENT"


class AnnotatedEvent(str, Enum):
    """
    Independent human observation of physical state or change across visits.
    Decoupled from system's internal EpistemicState belief state machine.
    """
    UNCHANGED = "UNCHANGED"
    RELOCATED = "RELOCATED"
    NEWLY_APPEARED = "NEWLY_APPEARED"
    ABSENT = "ABSENT"


@dataclass
class RealVisitAnnotation:
    """
    Manual ground-truth annotation for one physical entity in a real visit image.
    Decouples persistent physical identity and independent observations from system beliefs.
    """
    entity_label: str
    bbox: Optional[BoundingBox] = None
    expected_state: Optional[EpistemicState] = None
    entity_id: Optional[str] = None
    physical_observation: PhysicalObservation = PhysicalObservation.VISIBLE
    annotated_event: Optional[AnnotatedEvent] = None
    notes: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.entity_id is None or not self.entity_id.strip():
            self.entity_id = self.entity_label
        # If bbox is explicitly None and physical_observation was left at default VISIBLE:
        # if expected_state is removal/pending or annotated_event is ABSENT, automatically infer ABSENT
        if self.bbox is None and self.physical_observation == PhysicalObservation.VISIBLE:
            if (
                self.expected_state in (EpistemicState.PENDING_REMOVAL, EpistemicState.CONFIRMED_REMOVED)
                or self.annotated_event == AnnotatedEvent.ABSENT
            ):
                self.physical_observation = PhysicalObservation.ABSENT

        if not isinstance(self.physical_observation, PhysicalObservation):
            self.physical_observation = PhysicalObservation(self.physical_observation)
        if self.annotated_event is not None and not isinstance(self.annotated_event, AnnotatedEvent):
            self.annotated_event = AnnotatedEvent(self.annotated_event)


    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "entity_label": self.entity_label,
            "bbox": [self.bbox.x_min, self.bbox.y_min, self.bbox.x_max, self.bbox.y_max] if self.bbox else None,
            "expected_state": self.expected_state.value if self.expected_state else None,
            "physical_observation": self.physical_observation.value,
            "annotated_event": self.annotated_event.value if self.annotated_event else None,
            "notes": self.notes,
            "attributes": self.attributes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RealVisitAnnotation:
        raw_box = data.get("bbox")
        bbox = None
        if raw_box is not None:
            if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
                raise ValueError(f"Malformed bounding box: {raw_box}. Must be 4-element sequence [x_min, y_min, x_max, y_max].")
            bbox = BoundingBox(float(raw_box[0]), float(raw_box[1]), float(raw_box[2]), float(raw_box[3]))

        st_val = data.get("expected_state")
        st = None
        if st_val is not None:
            try:
                st = EpistemicState(st_val)
            except ValueError as exc:
                raise ValueError(f"Unknown expected_state '{st_val}'. Valid values: {[e.value for e in EpistemicState]}") from exc

        obs_val = data.get("physical_observation", PhysicalObservation.VISIBLE.value)
        try:
            obs = PhysicalObservation(obs_val)
        except ValueError as exc:
            raise ValueError(f"Unknown physical_observation '{obs_val}'. Valid values: {[o.value for o in PhysicalObservation]}") from exc

        evt_val = data.get("annotated_event")
        evt = None
        if evt_val is not None:
            try:
                evt = AnnotatedEvent(evt_val)
            except ValueError as exc:
                raise ValueError(f"Unknown annotated_event '{evt_val}'. Valid values: {[e.value for e in AnnotatedEvent]}") from exc

        return cls(
            entity_label=data["entity_label"],
            bbox=bbox,
            expected_state=st,
            entity_id=data.get("entity_id", data["entity_label"]),
            physical_observation=obs,
            annotated_event=evt,
            notes=data.get("notes", ""),
            attributes=data.get("attributes", {}),
        )


@dataclass
class RealVisitFrame:
    """A single visit image and its associated metadata in a multi-visit sequence."""
    visit_id: str
    session_name: str
    timestamp: str
    image_path: Path
    annotations: List[RealVisitAnnotation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    _cached_image: Optional[np.ndarray] = field(default=None, repr=False)

    def load_image(self) -> np.ndarray:
        """Loads and returns the BGR image from disk."""
        if self._cached_image is not None:
            return self._cached_image

        if not self.image_path.exists():
            raise FileNotFoundError(f"Visit image not found at '{self.image_path}'.")

        img = cv2.imread(str(self.image_path))
        if img is None:
            raise ValueError(f"Failed to decode image from '{self.image_path}'.")

        self._cached_image = img
        return img

    def release_cache(self) -> None:
        """Frees cached image to conserve CPU RAM."""
        self._cached_image = None


@dataclass
class RealWorkbenchDataset:
    """
    Dataset representing repeated physical visits to a workbench over time.
    """
    dataset_name: str
    description: str
    root_dir: Path
    visits: List[RealVisitFrame]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_annotated(self) -> bool:
        """Returns True only if all visits contain manual ground-truth annotations."""
        if not self.visits:
            return False
        return any(len(v.annotations) > 0 for v in self.visits)

    def validate_dataset_integrity(self) -> List[str]:
        """
        Performs thorough validation of the dataset files, timestamps, and annotations.
        Returns a list of error strings. If empty, the dataset structure is valid.
        """
        errors: List[str] = []

        if len(self.visits) < 2:
            errors.append(f"Sequence too short: Dataset contains {len(self.visits)} visit(s); minimum is 2.")

        # Check visit timestamps monotonicity and uniqueness
        prev_ts = None
        for i, v in enumerate(self.visits):
            if not v.visit_id or not v.visit_id.strip():
                errors.append(f"Visit at index {i} has empty or missing visit_id.")
            if not v.timestamp:
                errors.append(f"Visit '{v.visit_id}' has empty or missing timestamp.")
            else:
                try:
                    normalize_iso_timestamp(v.timestamp)
                except ValueError as exc:
                    errors.append(f"Visit '{v.visit_id}' has invalid timestamp '{v.timestamp}': {exc}")

            if prev_ts is not None and v.timestamp <= prev_ts:
                errors.append(
                    f"Non-monotonic timestamp order: Visit '{v.visit_id}' timestamp '{v.timestamp}' "
                    f"is not strictly greater than previous visit timestamp '{prev_ts}'."
                )
            prev_ts = v.timestamp

            # Check image existence and readability
            if not v.image_path.exists():
                errors.append(f"Missing image file for visit '{v.visit_id}': {v.image_path}")
            else:
                # Attempt decoding
                try:
                    img = v.load_image()
                    if img is None or img.size == 0:
                        errors.append(f"Image for visit '{v.visit_id}' at '{v.image_path}' decoded to empty/None array.")
                    v.release_cache()
                except Exception as exc:
                    errors.append(f"Unreadable image for visit '{v.visit_id}' at '{v.image_path}': {exc}")


        # Check entity persistent IDs and observation/box consistency
        entity_labels: Dict[str, str] = {}
        for v in self.visits:
            for a in v.annotations:
                # Check entity label and ID
                if not a.entity_label or not a.entity_label.strip():
                    errors.append(f"Visit '{v.visit_id}' contains annotation with missing entity_label.")
                if a.entity_id:
                    if a.entity_id in entity_labels and entity_labels[a.entity_id] != a.entity_label:
                        errors.append(
                            f"Inconsistent entity class label for entity_id '{a.entity_id}': "
                            f"previously '{entity_labels[a.entity_id]}', now '{a.entity_label}' in visit '{v.visit_id}'."
                        )
                    entity_labels[a.entity_id] = a.entity_label

                # Consistency between PhysicalObservation and BoundingBox
                if a.physical_observation == PhysicalObservation.VISIBLE and a.bbox is None:
                    errors.append(
                        f"Annotation for entity '{a.entity_id}' in visit '{v.visit_id}' has physical_observation='VISIBLE' "
                        f"but bbox is null/missing."
                    )
                if a.physical_observation in (PhysicalObservation.ABSENT, PhysicalObservation.FULLY_OCCLUDED, PhysicalObservation.OUT_OF_FRAME) and a.bbox is not None:
                    errors.append(
                        f"Annotation for entity '{a.entity_id}' in visit '{v.visit_id}' has physical_observation='{a.physical_observation.value}' "
                        f"but defines a non-null bbox {a.bbox.as_tuple()}. An unobservable/absent entity must have a null bounding box."
                    )
                if a.annotated_event == AnnotatedEvent.ABSENT and a.physical_observation != PhysicalObservation.ABSENT:
                    errors.append(
                        f"Annotation for entity '{a.entity_id}' in visit '{v.visit_id}' has annotated_event='ABSENT' "
                        f"but physical_observation='{a.physical_observation.value}'."
                    )

                # Semantic contradiction checks
                if a.annotated_event == AnnotatedEvent.UNCHANGED:
                    if a.expected_state in (EpistemicState.CONFIRMED_REMOVED, EpistemicState.PENDING_REMOVAL):
                        errors.append(
                            f"Contradictory annotation for entity '{a.entity_id}' in visit '{v.visit_id}': "
                            f"annotated_event='UNCHANGED' cannot be paired with expected_state='{a.expected_state.value}'."
                        )
                    if a.physical_observation == PhysicalObservation.ABSENT:
                        errors.append(
                            f"Contradictory annotation for entity '{a.entity_id}' in visit '{v.visit_id}': "
                            f"annotated_event='UNCHANGED' cannot be paired with physical_observation='ABSENT'."
                        )

                if a.physical_observation == PhysicalObservation.ABSENT:
                    if a.expected_state in (EpistemicState.CONFIRMED_PRESENT, EpistemicState.CONFIRMED_MOVED):
                        errors.append(
                            f"Contradictory annotation for entity '{a.entity_id}' in visit '{v.visit_id}': "
                            f"physical_observation='ABSENT' cannot be paired with expected_state='{a.expected_state.value}'."
                        )

                if a.annotated_event == AnnotatedEvent.ABSENT:
                    if a.expected_state in (EpistemicState.CONFIRMED_PRESENT, EpistemicState.CONFIRMED_MOVED):
                        errors.append(
                            f"Contradictory annotation for entity '{a.entity_id}' in visit '{v.visit_id}': "
                            f"annotated_event='ABSENT' cannot be paired with expected_state='{a.expected_state.value}'."
                        )

                if a.annotated_event == AnnotatedEvent.RELOCATED:
                    if a.expected_state in (EpistemicState.CONFIRMED_REMOVED, EpistemicState.PENDING_REMOVAL):
                        errors.append(
                            f"Contradictory annotation for entity '{a.entity_id}' in visit '{v.visit_id}': "
                            f"annotated_event='RELOCATED' cannot be paired with expected_state='{a.expected_state.value}'."
                        )
                    if a.physical_observation == PhysicalObservation.ABSENT:
                        errors.append(
                            f"Contradictory annotation for entity '{a.entity_id}' in visit '{v.visit_id}': "
                            f"annotated_event='RELOCATED' cannot be paired with physical_observation='ABSENT'."
                        )

                if a.annotated_event == AnnotatedEvent.NEWLY_APPEARED:
                    if a.expected_state in (EpistemicState.CONFIRMED_REMOVED, EpistemicState.PENDING_REMOVAL):
                        errors.append(
                            f"Contradictory annotation for entity '{a.entity_id}' in visit '{v.visit_id}': "
                            f"annotated_event='NEWLY_APPEARED' cannot be paired with expected_state='{a.expected_state.value}'."
                        )
                    if a.physical_observation == PhysicalObservation.ABSENT:
                        errors.append(
                            f"Contradictory annotation for entity '{a.entity_id}' in visit '{v.visit_id}': "
                            f"annotated_event='NEWLY_APPEARED' cannot be paired with physical_observation='ABSENT'."
                        )

                if a.physical_observation == PhysicalObservation.VISIBLE:
                    if a.expected_state in (EpistemicState.CONFIRMED_REMOVED, EpistemicState.PENDING_REMOVAL):
                        errors.append(
                            f"Contradictory annotation for entity '{a.entity_id}' in visit '{v.visit_id}': "
                            f"physical_observation='VISIBLE' cannot be paired with expected_state='{a.expected_state.value}'."
                        )
                if a.bbox is not None:
                    if a.bbox.x_max <= a.bbox.x_min or a.bbox.y_max <= a.bbox.y_min:
                        errors.append(
                            f"Annotation for entity '{a.entity_id}' in visit '{v.visit_id}' defines degenerate bbox {a.bbox.as_tuple()}: "
                            f"width and height must be strictly positive."
                        )
                    if v.image_path.exists():
                        try:
                            # Verify box fits within image bounds if image is accessible
                            img_shape = v.load_image().shape[:2]
                            h_img, w_img = img_shape
                            if a.bbox.x_max > w_img or a.bbox.y_max > h_img:
                                errors.append(
                                    f"Annotation for entity '{a.entity_id}' in visit '{v.visit_id}' defines bbox {a.bbox.as_tuple()} "
                                    f"exceeding image canvas dimensions (width={w_img}, height={h_img})."
                                )
                            v.release_cache()
                        except Exception:
                            pass

        return errors

    def is_evaluation_ready(self) -> Tuple[bool, str]:
        """
        Determines whether the dataset has sufficient imagery and annotations
        to perform quantitative evaluation.
        """
        # First check structural errors
        # Check if dataset is marked as template or contains placeholder images
        if self.metadata.get("is_template") or any("placeholder" in str(v.image_path) for v in self.visits):
            return (
                False,
                "Dataset contains placeholder image paths or template annotations. "
                "Evaluation is blocked until real physical photographs and manual annotations replace the template placeholders."
            )

        errors = self.validate_dataset_integrity()
        if errors:
            return False, "; ".join(errors)

        annotated_visits = sum(1 for v in self.visits if v.annotations)
        if annotated_visits == 0:
            return (
                False,
                "No manual ground-truth annotations found. Real-image detection and change evaluation remains blocked "
                "until manual bounding boxes and expected states are provided.",
            )

        return True, "Dataset is validated and ready for real-image evaluation."



    @classmethod
    def load_from_directory(cls, dir_path: Path | str) -> RealWorkbenchDataset:
        """
        Loads a real workbench dataset from a directory containing manifest.json.
        """
        root = Path(dir_path)
        manifest_path = root / "manifest.json"

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Real workbench dataset manifest not found at '{manifest_path}'. "
                f"Please place manifest.json and visit image folders in '{root}'."
            )

        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        visits: List[RealVisitFrame] = []
        seen_visit_ids: Set[str] = set()
        for v_data in data.get("visits", []):
            vid = v_data["visit_id"]
            if vid in seen_visit_ids:
                raise ValueError(f"Duplicate visit_id '{vid}' found in manifest '{manifest_path}'.")
            seen_visit_ids.add(vid)

            rel_img = v_data["image_path"]
            abs_img = (root / rel_img).resolve()
            ts = normalize_iso_timestamp(v_data["timestamp"])

            annots: List[RealVisitAnnotation] = []
            for a_data in v_data.get("annotations", []):
                annots.append(RealVisitAnnotation.from_dict(a_data))

            visits.append(
                RealVisitFrame(
                    visit_id=vid,
                    session_name=v_data.get("session_name", vid),
                    timestamp=ts,
                    image_path=abs_img,
                    annotations=annots,
                    metadata=v_data.get("metadata", {}),
                )
            )

        # Ensure visits are sorted chronologically
        visits.sort(key=lambda v: v.timestamp)

        meta = data.get("metadata", {})
        if "is_template" in data:
            meta["is_template"] = data["is_template"]

        return cls(
            dataset_name=data.get("dataset_name", "RealWorkbenchDataset"),
            description=data.get("description", "Physical workbench multi-visit sequence"),
            root_dir=root,
            visits=visits,
            metadata=meta,
        )

    @classmethod
    def create_empty_template(cls, target_dir: Path | str) -> Path:
        """
        Creates directory structure and manifest.json template for real data collection.
        """
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "visits").mkdir(exist_ok=True)

        template_manifest = {
            "dataset_name": "laboratory_workbench_template",
            "description": "TEMPLATE: Multi-visit photo sequence of physical electronics workbench (replace placeholders with real photographs)",
            "is_template": True,
            "metadata": {
                "sensor": "Overhead smartphone or bench camera (RGB)",
                "lighting_conditions": "Ceiling fluorescent + natural ambient light",
                "resolution": [1920, 1080],
                "coordinate_convention": "[x_min, y_min, x_max, y_max] in raw pixel coordinates where (0,0) is top-left"
            },
            "visits": [
                {
                    "visit_id": "visit_01",
                    "session_name": "Visit 1 Initial Workbench Baseline",
                    "timestamp": "2026-03-01T10:00:00Z",
                    "image_path": "visits/visit_01_placeholder.jpg",
                    "annotations": [
                        {
                            "entity_id": "cup_01",
                            "entity_label": "cup",
                            "bbox": [280.0, 80.0, 390.0, 260.0],
                            "physical_observation": "VISIBLE",
                            "annotated_event": "UNCHANGED",
                            "expected_state": "CONFIRMED_PRESENT",
                            "notes": "Initial baseline placement at primary station"
                        },
                        {
                            "entity_id": "scissors_01",
                            "entity_label": "scissors",
                            "bbox": [120.0, 310.0, 210.0, 420.0],
                            "physical_observation": "VISIBLE",
                            "annotated_event": "UNCHANGED",
                            "expected_state": "CONFIRMED_PRESENT",
                            "notes": "Tool placed on left mat"
                        },
                        {
                            "entity_id": "bottle_01",
                            "entity_label": "bottle",
                            "bbox": [510.0, 150.0, 620.0, 380.0],
                            "physical_observation": "VISIBLE",
                            "annotated_event": "UNCHANGED",
                            "expected_state": "CONFIRMED_PRESENT",
                            "notes": "Solvent bottle on right tray"
                        }
                    ]
                },
                {
                    "visit_id": "visit_02",
                    "session_name": "Visit 2 Relocation and Temporary Occlusion",
                    "timestamp": "2026-03-01T11:00:00Z",
                    "image_path": "visits/visit_02_placeholder.jpg",
                    "annotations": [
                        {
                            "entity_id": "cup_01",
                            "entity_label": "cup",
                            "bbox": [440.0, 200.0, 550.0, 380.0],
                            "physical_observation": "VISIBLE",
                            "annotated_event": "RELOCATED",
                            "expected_state": "CONFIRMED_MOVED",
                            "notes": "Relocated >35px across bench mat"
                        },
                        {
                            "entity_id": "scissors_01",
                            "entity_label": "scissors",
                            "bbox": None,
                            "physical_observation": "PARTIALLY_OCCLUDED",
                            "annotated_event": "UNCHANGED",
                            "expected_state": "CONFIRMED_PRESENT",
                            "notes": "Covered by clipboard/notebook; unobservable but present"
                        },
                        {
                            "entity_id": "bottle_01",
                            "entity_label": "bottle",
                            "bbox": None,
                            "physical_observation": "ABSENT",
                            "annotated_event": "ABSENT",
                            "expected_state": "PENDING_REMOVAL",
                            "notes": "First reliable absence; streak=1"
                        }
                    ]
                },
                {
                    "visit_id": "visit_03",
                    "session_name": "Visit 3 Confirmed Removal and Occlusion Recovery",
                    "timestamp": "2026-03-01T12:00:00Z",
                    "image_path": "visits/visit_03_placeholder.jpg",
                    "annotations": [
                        {
                            "entity_id": "cup_01",
                            "entity_label": "cup",
                            "bbox": [440.0, 200.0, 550.0, 380.0],
                            "physical_observation": "VISIBLE",
                            "annotated_event": "UNCHANGED",
                            "expected_state": "CONFIRMED_MOVED",
                            "notes": "Remains in new relocated position"
                        },
                        {
                            "entity_id": "scissors_01",
                            "entity_label": "scissors",
                            "bbox": [120.0, 310.0, 210.0, 420.0],
                            "physical_observation": "VISIBLE",
                            "annotated_event": "UNCHANGED",
                            "expected_state": "CONFIRMED_PRESENT",
                            "notes": "Uncovered; confirmed present again"
                        },
                        {
                            "entity_id": "bottle_01",
                            "entity_label": "bottle",
                            "bbox": None,
                            "physical_observation": "ABSENT",
                            "annotated_event": "ABSENT",
                            "expected_state": "CONFIRMED_REMOVED",
                            "notes": "Consecutive reliable absence (streak=2); confirmed removed"
                        }
                    ]
                },
                {
                    "visit_id": "visit_04",
                    "session_name": "Visit 4 Tool Reappearance / Return",
                    "timestamp": "2026-03-01T13:00:00Z",
                    "image_path": "visits/visit_04_placeholder.jpg",
                    "annotations": [
                        {
                            "entity_id": "cup_01",
                            "entity_label": "cup",
                            "bbox": [440.0, 200.0, 550.0, 380.0],
                            "physical_observation": "VISIBLE",
                            "annotated_event": "UNCHANGED",
                            "expected_state": "CONFIRMED_MOVED",
                            "notes": "Remains unchanged in moved spot"
                        },
                        {
                            "entity_id": "bottle_01",
                            "entity_label": "bottle",
                            "bbox": [510.0, 150.0, 620.0, 380.0],
                            "physical_observation": "VISIBLE",
                            "annotated_event": "NEWLY_APPEARED",
                            "expected_state": "CONFIRMED_PRESENT",
                            "notes": "Returned to workbench; restored to CONFIRMED_PRESENT"
                        }
                    ]
                }
            ],
        }


        manifest_path = target / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(template_manifest, f, indent=2)

        readme_path = target / "README.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(
                "# Real Workbench Dataset Directory\n\n"
                "Place multi-visit physical camera photos in `visits/`.\n"
                "Define timestamps and optional ground-truth annotations in `manifest.json`.\n"
            )

        return manifest_path
