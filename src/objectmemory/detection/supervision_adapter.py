"""
Roboflow Supervision Adapter for ObjectMemory.

Provides interoperability between ObjectMemory detection domain models
and Supervision data structures without replacing ObjectMemory's custom
temporal reasoning, epistemic state machine, or persistent spatial memory.

Guarantees:
1. Model-specific output conversion is isolated behind adapters.
2. Original detector labels, coordinates, and confidence values are preserved exactly.
3. No confidence values are manufactured or fabricated.
4. Supervision is treated strictly as an interoperability and evaluation utility,
   NOT as a detector or long-term identity tracker.
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

import supervision as sv

from objectmemory.domain.models import BoundingBox
from objectmemory.detection.models import Detection, DetectionBatch


def to_supervision_detections(
    detections: Union[DetectionBatch, Sequence[Detection]],
    class_to_id: Optional[Dict[str, int]] = None,
    strict_box_validation: bool = True,
) -> sv.Detections:
    """
    Converts ObjectMemory detections into a Supervision Detections object.

    Args:
        detections: DetectionBatch or Sequence of Detection domain models.
        class_to_id: Optional mapping from label strings to integer class IDs.
        strict_box_validation: If True, raises ValueError on invalid boxes.

    Returns:
        sv.Detections with xyxy, confidence, class_id, and data payload.
    """
    det_list: Sequence[Detection]
    if isinstance(detections, DetectionBatch):
        det_list = detections.detections
    else:
        det_list = detections

    if not det_list:
        return sv.Detections.empty()

    xyxy_list: List[List[float]] = []
    conf_list: List[float] = []
    labels: List[str] = []
    canonical_xyxy_list: List[Optional[List[float]]] = []

    # Build unique class list if mapping not provided
    if class_to_id is None:
        unique_labels = sorted(list({d.label for d in det_list}))
        class_to_id = {lbl: idx for idx, lbl in enumerate(unique_labels)}

    class_ids: List[int] = []

    for d in det_list:
        bbox = d.raw_bbox
        x1, y1, x2, y2 = bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max

        # Validate coordinate finiteness
        if not (math.isfinite(x1) and math.isfinite(y1) and math.isfinite(x2) and math.isfinite(y2)):
            if strict_box_validation:
                raise ValueError(f"Detection has non-finite coordinates: ({x1}, {y1}, {x2}, {y2})")
            continue

        # Validate non-inversion
        if x2 < x1 or y2 < y1:
            if strict_box_validation:
                raise ValueError(f"Detection has inverted bounding box coordinates: ({x1}, {y1}, {x2}, {y2})")
            continue

        # Validate confidence score in [0.0, 1.0]
        if not math.isfinite(d.confidence) or not (0.0 <= d.confidence <= 1.0):
            if strict_box_validation:
                raise ValueError(f"Detection confidence must be in [0.0, 1.0], got {d.confidence}")
            continue

        xyxy_list.append([float(x1), float(y1), float(x2), float(y2)])
        conf_list.append(float(d.confidence))
        labels.append(d.label)
        cid = class_to_id.get(d.label, -1)
        class_ids.append(cid)

        if d.canonical_bbox is not None:
            cb = d.canonical_bbox
            canonical_xyxy_list.append([float(cb.x_min), float(cb.y_min), float(cb.x_max), float(cb.y_max)])
        else:
            canonical_xyxy_list.append(None)

    if not xyxy_list:
        return sv.Detections.empty()

    xyxy_array = np.array(xyxy_list, dtype=np.float32)
    conf_array = np.array(conf_list, dtype=np.float32)
    class_id_array = np.array(class_ids, dtype=int)
    class_name_array = np.array(labels, dtype=object)

    custom_data: Dict[str, Any] = {
        "class_name": class_name_array,
    }

    # Include canonical bounding boxes if at least one detection has it
    if any(c is not None for c in canonical_xyxy_list):
        custom_data["canonical_xyxy"] = canonical_xyxy_list

    return sv.Detections(
        xyxy=xyxy_array,
        confidence=conf_array,
        class_id=class_id_array,
        data=custom_data,
    )


def from_supervision_detections(
    sv_detections: sv.Detections,
    detector_name: str = "supervision_converted",
    image_size: Tuple[int, int] = (640, 480),
    id_to_class: Optional[Dict[int, str]] = None,
    latency_ms: float = 0.0,
    is_synthetic: bool = False,
) -> DetectionBatch:
    """
    Converts Supervision Detections into an ObjectMemory DetectionBatch.

    Args:
        sv_detections: Supervision Detections instance.
        detector_name: Detector name to record in DetectionBatch.
        image_size: (width, height) of image.
        id_to_class: Optional mapping from class_id to label string.
        latency_ms: Associated detector latency.
        is_synthetic: Flag indicating whether input is synthetic.

    Returns:
        DetectionBatch containing ObjectMemory Detection instances.
    """
    if len(sv_detections) == 0:
        return DetectionBatch(
            detections=[],
            image_size=image_size,
            latency_ms=latency_ms,
            detector_name=detector_name,
            is_synthetic=is_synthetic,
        )

    detections: List[Detection] = []
    class_names = sv_detections.data.get("class_name")
    canonical_boxes = sv_detections.data.get("canonical_xyxy")

    for i in range(len(sv_detections)):
        x1, y1, x2, y2 = sv_detections.xyxy[i]
        raw_box = BoundingBox(
            x_min=float(x1),
            y_min=float(y1),
            x_max=float(x2),
            y_max=float(y2),
        )

        conf: float = 1.0
        if sv_detections.confidence is not None and i < len(sv_detections.confidence):
            c_val = float(sv_detections.confidence[i])
            conf = max(0.0, min(1.0, c_val))

        label: str = "unknown"
        if class_names is not None and i < len(class_names) and class_names[i] is not None:
            label = str(class_names[i])
        elif sv_detections.class_id is not None and i < len(sv_detections.class_id):
            cid = int(sv_detections.class_id[i])
            if id_to_class and cid in id_to_class:
                label = id_to_class[cid]
            else:
                label = f"class_{cid}"

        canonical_bbox: Optional[BoundingBox] = None
        if canonical_boxes is not None and i < len(canonical_boxes) and canonical_boxes[i] is not None:
            cx1, cy1, cx2, cy2 = canonical_boxes[i]
            canonical_bbox = BoundingBox(
                x_min=float(cx1),
                y_min=float(cy1),
                x_max=float(cx2),
                y_max=float(cy2),
            )

        detections.append(
            Detection(
                label=label,
                confidence=conf,
                raw_bbox=raw_box,
                canonical_bbox=canonical_bbox,
            )
        )

    return DetectionBatch(
        detections=detections,
        image_size=image_size,
        latency_ms=latency_ms,
        detector_name=detector_name,
        is_synthetic=is_synthetic,
    )


def annotate_with_supervision(
    image: np.ndarray,
    detections: Union[DetectionBatch, Sequence[Detection], sv.Detections],
    box_annotator: Optional[sv.BoxAnnotator] = None,
    label_annotator: Optional[sv.LabelAnnotator] = None,
    custom_labels: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Renders bounding boxes and labels onto an image using Supervision annotators.

    Args:
        image: BGR or RGB uint8 numpy image array.
        detections: ObjectMemory detections or Supervision Detections.
        box_annotator: Optional custom BoxAnnotator.
        label_annotator: Optional custom LabelAnnotator.
        custom_labels: Optional list of formatted label strings.

    Returns:
        Annotated copy of the image.
    """
    sv_dets: sv.Detections
    if isinstance(detections, sv.Detections):
        sv_dets = detections
    else:
        sv_dets = to_supervision_detections(detections)

    annotated = image.copy()
    if len(sv_dets) == 0:
        return annotated

    ba = box_annotator or sv.BoxAnnotator(thickness=2)
    la = label_annotator or sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    annotated = ba.annotate(scene=annotated, detections=sv_dets)

    if custom_labels is None:
        if "class_name" in sv_dets.data and sv_dets.confidence is not None:
            labels = [
                f"{name} {conf:.2f}"
                for name, conf in zip(sv_dets.data["class_name"], sv_dets.confidence)
            ]
        elif sv_dets.class_id is not None and sv_dets.confidence is not None:
            labels = [
                f"{cid} {conf:.2f}"
                for cid, conf in zip(sv_dets.class_id, sv_dets.confidence)
            ]
        else:
            labels = [f"obj {i}" for i in range(len(sv_dets))]
    else:
        labels = custom_labels

    annotated = la.annotate(scene=annotated, detections=sv_dets, labels=labels)
    return annotated


def evaluate_detection_map(
    predictions: Sequence[Union[DetectionBatch, sv.Detections]],
    targets: Sequence[Union[DetectionBatch, sv.Detections]],
    class_agnostic: bool = False,
) -> Dict[str, Any]:
    """
    Calculates COCO-style Mean Average Precision (mAP) using Supervision.

    Args:
        predictions: List of predicted detections per image.
        targets: List of ground-truth detections per image.
        class_agnostic: Whether to evaluate class-agnostic mAP.

    Returns:
        Dictionary with map50_95, map50, map75, and per-class AP summaries.
    """
    if len(predictions) != len(targets):
        raise ValueError(
            f"Number of prediction frames ({len(predictions)}) does not match target frames ({len(targets)})"
        )

    sv_preds: List[sv.Detections] = []
    sv_targets: List[sv.Detections] = []

    # Determine unified class mapping across all frames
    all_classes = set()
    for item in list(predictions) + list(targets):
        if isinstance(item, DetectionBatch):
            all_classes.update(d.label for d in item.detections)
        elif isinstance(item, sv.Detections) and "class_name" in item.data:
            all_classes.update(str(name) for name in item.data["class_name"])

    class_to_id = {name: idx for idx, name in enumerate(sorted(all_classes))}

    for p in predictions:
        if isinstance(p, sv.Detections):
            sv_preds.append(p)
        else:
            sv_preds.append(to_supervision_detections(p, class_to_id=class_to_id))

    for t in targets:
        if isinstance(t, sv.Detections):
            sv_targets.append(t)
        else:
            sv_targets.append(to_supervision_detections(t, class_to_id=class_to_id))

    metric = sv.metrics.MeanAveragePrecision(class_agnostic=class_agnostic)
    metric.update(sv_preds, sv_targets)
    result = metric.compute()

    return {
        "map50_95": float(result.map50_95),
        "map50": float(result.map50),
        "map75": float(result.map75),
        "classes_evaluated": list(sorted(all_classes)),
        "class_agnostic": class_agnostic,
        "total_images": len(predictions),
    }
