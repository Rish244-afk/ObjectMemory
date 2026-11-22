"""
Controlled Temporal Dataset loader and validation infrastructure.

Provides structured access and strict integrity validation for the procedurally
generated 6-sequence, 30-image controlled benchmark dataset.

Scientific Boundaries:
1. Images are procedurally generated illustrations, NOT natural photographs.
2. Renderer-derived bounding boxes provide exact programmatic ground truth.
3. This dataset evaluates controlled temporal-reasoning logic, registration
   under specific geometric/lighting perturbations, and state machine transitions.
   It must NEVER be cited as a natural-image detection benchmark.
"""

from __future__ import annotations
import csv
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from objectmemory.domain.models import BoundingBox


@dataclass
class ControlledAnnotation:
    """Ground truth annotation for a single object in a visit frame."""
    object_id: str
    class_name: str
    bbox: BoundingBox
    visibility: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_id": self.object_id,
            "class": self.class_name,
            "bbox_xyxy": [self.bbox.x_min, self.bbox.y_min, self.bbox.x_max, self.bbox.y_max],
            "visibility": self.visibility,
        }


@dataclass
class ControlledVisit:
    """A single observation visit within a controlled temporal sequence."""
    visit_index: int
    image_relative_path: str
    image_absolute_path: str
    nuisance_tags: List[str]
    annotations: List[ControlledAnnotation]
    image_dimensions: Tuple[int, int]  # (width, height)

    def load_image(self) -> np.ndarray:
        if not os.path.exists(self.image_absolute_path):
            raise FileNotFoundError(f"Image not found at: {self.image_absolute_path}")
        img = cv2.imread(self.image_absolute_path)
        if img is None:
            raise ValueError(f"Failed to decode image at: {self.image_absolute_path}")
        return img


@dataclass
class ControlledSequence:
    """A multi-visit temporal sequence testing a specific environmental hypothesis."""
    sequence_id: str
    description: str
    visits: List[ControlledVisit]

    def get_visit(self, visit_index: int) -> ControlledVisit:
        for v in self.visits:
            if v.visit_index == visit_index:
                return v
        raise KeyError(f"Visit index {visit_index} not found in sequence {self.sequence_id}")


@dataclass
class ControlledExpectedEvent:
    """Expected temporal state machine transition or nuisance invariance event."""
    sequence_id: str
    visit_index: int
    event: str
    target: Optional[str] = None
    object_id: Optional[str] = None
    should_create_object_movement: Optional[bool] = None
    should_create_removal: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "sequence_id": self.sequence_id,
            "visit_index": self.visit_index,
            "event": self.event,
        }
        if self.target is not None:
            d["target"] = self.target
        if self.object_id is not None:
            d["object_id"] = self.object_id
        if self.should_create_object_movement is not None:
            d["should_create_object_movement"] = self.should_create_object_movement
        if self.should_create_removal is not None:
            d["should_create_removal"] = self.should_create_removal
        return d


class ControlledDataset:
    """
    Reader, validator, and accessor for the controlled temporal benchmark dataset.
    """

    def __init__(
        self,
        dataset_dir: str,
        manifest_data: Dict[str, Any],
        expected_events: List[ControlledExpectedEvent],
        sequences: Dict[str, ControlledSequence],
    ):
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.manifest_data = manifest_data
        self.expected_events = expected_events
        self.sequences = sequences

    @classmethod
    def load(cls, dataset_dir: str) -> ControlledDataset:
        """Loads and parses the dataset from disk without altering original files."""
        abs_dir = os.path.abspath(dataset_dir)
        manifest_path = os.path.join(abs_dir, "annotations", "manifest.json")
        events_path = os.path.join(abs_dir, "annotations", "expected_events.json")

        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        if not os.path.exists(events_path):
            raise FileNotFoundError(f"Expected events not found: {events_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        with open(events_path, "r", encoding="utf-8") as f:
            raw_events = json.load(f)

        expected_events = [
            ControlledExpectedEvent(
                sequence_id=e["sequence_id"],
                visit_index=int(e["visit_index"]),
                event=e["event"],
                target=e.get("target"),
                object_id=e.get("object_id"),
                should_create_object_movement=e.get("should_create_object_movement"),
                should_create_removal=e.get("should_create_removal"),
            )
            for e in raw_events
        ]

        sequences: Dict[str, ControlledSequence] = {}
        for seq_dict in manifest_data.get("sequences", []):
            s_id = seq_dict["sequence_id"]
            desc = seq_dict.get("description", "")
            visits: List[ControlledVisit] = []

            for v_dict in seq_dict.get("visits", []):
                v_idx = int(v_dict["visit_index"])
                rel_img = v_dict["image"]
                abs_img = os.path.join(abs_dir, rel_img.replace("/", os.sep))
                nuisance = list(v_dict.get("nuisance_tags", []))

                annotations: List[ControlledAnnotation] = []
                for ann in v_dict.get("annotations", []):
                    bbox_raw = ann["bbox_xyxy"]
                    box = BoundingBox(
                        x_min=float(bbox_raw[0]),
                        y_min=float(bbox_raw[1]),
                        x_max=float(bbox_raw[2]),
                        y_max=float(bbox_raw[3]),
                    )
                    annotations.append(
                        ControlledAnnotation(
                            object_id=ann["object_id"],
                            class_name=ann["class"],
                            bbox=box,
                            visibility=ann.get("visibility", "visible"),
                        )
                    )

                dims_raw = manifest_data.get("image_dimensions", [640, 480])
                if isinstance(dims_raw, (list, tuple)) and len(dims_raw) >= 2:
                    expected_dims = (int(dims_raw[0]), int(dims_raw[1]))
                elif isinstance(dims_raw, dict):
                    expected_dims = (int(dims_raw.get("width", 640)), int(dims_raw.get("height", 480)))
                else:
                    expected_dims = (640, 480)

                visits.append(
                    ControlledVisit(
                        visit_index=v_idx,
                        image_relative_path=rel_img,
                        image_absolute_path=abs_img,
                        nuisance_tags=nuisance,
                        annotations=annotations,
                        image_dimensions=expected_dims,
                    )
                )

            sequences[s_id] = ControlledSequence(
                sequence_id=s_id,
                description=desc,
                visits=visits,
            )

        return cls(
            dataset_dir=abs_dir,
            manifest_data=manifest_data,
            expected_events=expected_events,
            sequences=sequences,
        )

    def validate(self) -> Dict[str, Any]:
        """
        Validates every image path, image dimension, annotation bounding box,
        object ID, visit ID, and expected event in the dataset.

        Returns a detailed machine-readable validation report dictionary.
        """
        errors: List[str] = []
        warnings: List[str] = []
        images_validated = 0
        annotations_validated = 0
        events_validated = 0

        dims_raw = self.manifest_data.get("image_dimensions", [640, 480])
        if isinstance(dims_raw, (list, tuple)) and len(dims_raw) >= 2:
            manifest_dims = (int(dims_raw[0]), int(dims_raw[1]))
        elif isinstance(dims_raw, dict):
            manifest_dims = (int(dims_raw.get("width", 640)), int(dims_raw.get("height", 480)))
        else:
            manifest_dims = (640, 480)

        # 1. Validate images and annotations
        for seq_id, seq in self.sequences.items():
            for visit in seq.visits:
                img_path = visit.image_absolute_path
                if not os.path.exists(img_path):
                    errors.append(f"Image missing for {seq_id} visit {visit.visit_index}: {img_path}")
                    continue

                img = cv2.imread(img_path)
                if img is None:
                    errors.append(f"Image could not be decoded: {img_path}")
                    continue

                images_validated += 1
                h, w = img.shape[:2]
                if (w, h) != manifest_dims:
                    errors.append(
                        f"Dimension mismatch in {visit.image_relative_path}: expected {manifest_dims}, got ({w}, {h})"
                    )

                for ann in visit.annotations:
                    annotations_validated += 1
                    b = ann.bbox
                    if not (0 <= b.x_min < b.x_max <= w and 0 <= b.y_min < b.y_max <= h):
                        errors.append(
                            f"Invalid bounding box in {seq_id} v{visit.visit_index} for {ann.object_id}: {b.as_tuple()}"
                        )

        # 2. Validate expected events reference valid sequences and visits
        for ev in self.expected_events:
            events_validated += 1
            if ev.sequence_id not in self.sequences:
                errors.append(f"Expected event references unknown sequence: {ev.sequence_id}")
                continue
            seq = self.sequences[ev.sequence_id]
            visit_indices = [v.visit_index for v in seq.visits]
            if ev.visit_index not in visit_indices:
                errors.append(
                    f"Expected event references invalid visit {ev.visit_index} in sequence {ev.sequence_id}"
                )

        # 3. Validate CSV consistency against manifest
        csv_path = os.path.join(self.dataset_dir, "annotations", "object_annotations.csv")
        csv_rows = 0
        if os.path.exists(csv_path):
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for _ in reader:
                    csv_rows += 1
            if csv_rows != annotations_validated:
                errors.append(
                    f"CSV row count ({csv_rows}) does not match manifest annotations count ({annotations_validated})"
                )
        else:
            warnings.append(f"Optional CSV file missing: {csv_path}")

        status = "PASSED" if not errors else "FAILED"
        report = {
            "dataset_name": self.manifest_data.get("dataset_name", "Controlled Temporal Benchmark"),
            "status": status,
            "validation_timestamp": self.manifest_data.get("created_date", ""),
            "images_validated": images_validated,
            "annotations_validated": annotations_validated,
            "expected_events_validated": events_validated,
            "sequences_count": len(self.sequences),
            "errors": errors,
            "warnings": warnings,
            "scientific_classification": "SYNTHETIC_PROCEDURAL_BENCHMARK",
            "provenance": "CC0-1.0 procedural rendering, non-photographic",
        }
        return report

    def save_validation_report(self, output_path: Optional[str] = None) -> str:
        """Generates and writes a machine-readable validation report."""
        report = self.validate()
        if output_path is None:
            output_path = os.path.join(self.dataset_dir, "validation_report.json")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return output_path
