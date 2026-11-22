"""
Object Detection and Visual Signature Extraction module for ObjectMemory.

Provides:
- Pluggable ObjectDetector interface.
- OpenCVDNNDetector for CPU-first pretrained inference (ONNX / Caffe / TF).
- MockObjectDetector for deterministic testing without external downloads.
- VisualSignatureExtractor computing invariant color and texture descriptors.
- DetectionPipeline mapping detections to canonical reference frames via homography.
"""

from objectmemory.detection.exceptions import (
    DetectorConfigurationError,
    DetectorError,
    DetectorInferenceError,
    DetectorUnavailableError,
)
from objectmemory.detection.models import (
    Detection,
    DetectionBatch,
    DetectorConfiguration,
    SignatureMatch,
    VisualSignature,
)
from objectmemory.detection.detector import (
    COCO_CLASSES_80,
    MockObjectDetector,
    ObjectDetector,
    OpenCVDNNDetector,
)
from objectmemory.detection.registry import DetectorRegistry
from objectmemory.detection.signature import VisualSignatureExtractor
from objectmemory.detection.pipeline import DetectionPipeline
from objectmemory.detection.supervision_adapter import (
    annotate_with_supervision,
    evaluate_detection_map,
    from_supervision_detections,
    to_supervision_detections,
)

__all__ = [
    "COCO_CLASSES_80",
    "Detection",
    "DetectionBatch",
    "DetectorConfiguration",
    "DetectorConfigurationError",
    "DetectorError",
    "DetectorInferenceError",
    "DetectorRegistry",
    "DetectorUnavailableError",
    "VisualSignature",
    "SignatureMatch",
    "ObjectDetector",
    "OpenCVDNNDetector",
    "MockObjectDetector",
    "VisualSignatureExtractor",
    "DetectionPipeline",
    "to_supervision_detections",
    "from_supervision_detections",
    "annotate_with_supervision",
    "evaluate_detection_map",
]
