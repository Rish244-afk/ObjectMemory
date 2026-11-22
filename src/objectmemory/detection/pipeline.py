"""
Detection pipeline coordinating object detection, signature extraction,
and coordinate mapping into the canonical reference frame.

Safeguards:
1. Maps bounding box corners via homography H into canonical reference coordinates.
2. If alignment failed (H is None), canonical_bbox is set to None (no fabricated coordinates).
3. Converts detections to Domain Observation instances without manufacturing match scores.
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import cv2
import numpy as np

from objectmemory.detection.detector import ObjectDetector
from objectmemory.detection.models import (
    Detection,
    DetectionBatch,
    DetectorConfiguration,
)
from objectmemory.detection.signature import VisualSignatureExtractor
from objectmemory.domain.models import BoundingBox, Observation
from objectmemory.domain.states import VisibilityStatus


def transform_bbox_to_canonical(
    bbox: BoundingBox,
    homography: np.ndarray,
    ref_size: Tuple[int, int],
) -> Optional[BoundingBox]:
    """
    Transforms a 2D bounding box from query coordinates to reference coordinates
    using homography H (where p_ref ~ H @ p_query).

    Returns an axis-aligned BoundingBox in reference coordinates clamped to ref_size,
    or None if the transformed box is invalid or out-of-bounds.
    """
    if not np.all(np.isfinite(homography)):
        return None

    ref_w, ref_h = ref_size
    corners = np.array(
        [
            [bbox.x_min, bbox.y_min],
            [bbox.x_max, bbox.y_min],
            [bbox.x_max, bbox.y_max],
            [bbox.x_min, bbox.y_max],
        ],
        dtype=np.float64,
    ).reshape(-1, 1, 2)

    try:
        projected = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
    except cv2.error:
        return None

    if not np.all(np.isfinite(projected)):
        return None

    x_min = float(np.min(projected[:, 0]))
    y_min = float(np.min(projected[:, 1]))
    x_max = float(np.max(projected[:, 0]))
    y_max = float(np.max(projected[:, 1]))

    # Clamp to reference canvas
    c_xmin = max(0.0, min(float(ref_w), x_min))
    c_ymin = max(0.0, min(float(ref_h), y_min))
    c_xmax = max(0.0, min(float(ref_w), x_max))
    c_ymax = max(0.0, min(float(ref_h), y_max))

    if c_xmax <= c_xmin or c_ymax <= c_ymin:
        # Bounding box projected entirely outside of canvas or collapsed
        return None

    return BoundingBox(x_min=c_xmin, y_min=c_ymin, x_max=c_xmax, y_max=c_ymax)


def is_bbox_boundary_clipped(
    bbox: BoundingBox,
    image_size: Tuple[int, int],
    margin: float = 8.0,
) -> bool:
    """
    Checks if a 2D bounding box touches or crosses the image boundary within the given margin.
    image_size is (width, height).
    """
    w, h = image_size
    return (
        bbox.x_min <= margin
        or bbox.y_min <= margin
        or bbox.x_max >= float(w) - margin
        or bbox.y_max >= float(h) - margin
    )


class DetectionPipeline:
    """
    Coordinates object detection, visual signature extraction, and spatial transformation.
    """

    def __init__(
        self,
        detector: ObjectDetector,
        signature_extractor: Optional[VisualSignatureExtractor] = None,
    ) -> None:
        self.detector = detector
        self.signature_extractor = signature_extractor or VisualSignatureExtractor()

    def process_frame(
        self,
        image: np.ndarray,
        homography: Optional[np.ndarray] = None,
        ref_size: Optional[Tuple[int, int]] = None,
        confidence_threshold: float = 0.4,
        config: Optional[DetectorConfiguration] = None,
    ) -> DetectionBatch:
        """
        Runs detection, visual signature extraction, and optional homography coordinate transformation.
        """
        if config is not None:
            batch = self.detector.detect(image, config=config)
        else:
            batch = self.detector.detect(image, confidence_threshold=confidence_threshold)

        h_img, w_img = image.shape[:2]
        for det in batch.detections:
            # 1. Extract visual appearance signature
            det.signature = self.signature_extractor.extract(image, det.raw_bbox)

            # 2. Check boundary clipping
            det.metadata["is_clipped"] = is_bbox_boundary_clipped(det.raw_bbox, (w_img, h_img))

            # 3. Transform bounding box to canonical reference frame if alignment succeeded
            if homography is not None and ref_size is not None:
                det.canonical_bbox = transform_bbox_to_canonical(
                    det.raw_bbox,
                    homography,
                    ref_size,
                )
            else:
                det.canonical_bbox = None

        return batch

    def to_observations(
        self,
        batch: DetectionBatch,
        session_id: str,
        timestamp: str,
        image_reference: str,
        alignment_quality: Optional[float] = None,
    ) -> List[Observation]:
        """
        Converts detection batch to Domain Observation records.
        Identity match score is explicitly left as None (unmanufactured at detection stage).
        """
        observations: List[Observation] = []
        for idx, det in enumerate(batch.detections):
            obs_id = f"obs_{session_id}_{idx:03d}"
            obs = Observation(
                observation_id=obs_id,
                session_id=session_id,
                timestamp=timestamp,
                image_reference=image_reference,
                raw_bbox=det.raw_bbox,
                canonical_bbox=det.canonical_bbox,
                visibility_status=VisibilityStatus.CLEARLY_VISIBLE,
                detector_confidence=det.confidence,
                identity_match_score=None,  # Left unmanufactured until re-ID
                alignment_quality=alignment_quality,
                appearance_descriptor_ref=det.label,
                raw_evidence={
                    "label": det.label,
                    "detector": batch.detector_name,
                    "is_synthetic": batch.is_synthetic,
                    "is_clipped": det.metadata.get("is_clipped", False),
                },
            )
            observations.append(obs)
        return observations
