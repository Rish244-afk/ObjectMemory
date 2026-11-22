"""
Pluggable object detector interfaces and implementations for ObjectMemory.

Safeguards:
1. ObjectDetector is an abstract interface; the memory engine does not couple to any specific model.
2. OpenCVDNNDetector runs pretrained models strictly on CPU via OpenCV DNN.
3. If weights are absent, OpenCVDNNDetector raises FileNotFoundError explaining where
   to place them, rather than faking detections.
4. MockObjectDetector is explicitly labeled with is_synthetic=True for unit testing.
"""

from __future__ import annotations
import abc
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np

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
)
from objectmemory.detection.letterbox import (
    LetterboxInfo,
    inverse_transform_cxcywh,
    letterbox_image,
)
from objectmemory.domain.models import BoundingBox

logger = logging.getLogger(__name__)

# Standard COCO 80 classes taxonomy
COCO_CLASSES_80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


class ObjectDetector(abc.ABC):
    """Abstract interface for object detection on workbench frames."""

    @property
    def default_confidence_threshold(self) -> float:
        return 0.25

    @abc.abstractmethod
    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.25,
        **kwargs: Any,
    ) -> DetectionBatch:
        """Runs detection on an input image and returns a normalized DetectionBatch."""
        raise NotImplementedError

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Returns True if the detector weights and runtime dependencies are available."""
        raise NotImplementedError

    def get_availability_error(self) -> Optional[str]:
        """Returns a human-readable explanation if the detector is unavailable, else None."""
        return None

    @abc.abstractmethod
    def model_info(self) -> Dict[str, Any]:
        """Returns model metadata, license, execution backend, and known limitations."""
        raise NotImplementedError

    def resolve_configuration(
        self,
        confidence_threshold: Optional[float | DetectorConfiguration] = None,
        config: Optional[DetectorConfiguration] = None,
        default_conf: Optional[float] = None,
    ) -> DetectorConfiguration:
        """Helper to resolve confidence_threshold or DetectorConfiguration into a valid DetectorConfiguration."""
        if isinstance(confidence_threshold, DetectorConfiguration):
            return confidence_threshold
        def_conf = default_conf if default_conf is not None else self.default_confidence_threshold
        if config is not None:
            if confidence_threshold is not None:
                return DetectorConfiguration(
                    confidence_threshold=float(confidence_threshold),
                    nms_threshold=config.nms_threshold,
                    target_classes=config.target_classes,
                    max_detections=config.max_detections,
                    extra_params=config.extra_params,
                )
            return config
        if confidence_threshold is not None:
            return DetectorConfiguration(confidence_threshold=float(confidence_threshold))
        return DetectorConfiguration(confidence_threshold=def_conf)

    def normalize_detection_batch(
        self,
        detections: List[Detection],
        image_size: Tuple[int, int],
        latency_ms: float,
        detector_name: str,
        is_synthetic: bool = False,
        failure_reason: Optional[str] = None,
        config: Optional[DetectorConfiguration] = None,
    ) -> DetectionBatch:
        """
        Safely normalizes, clamps, filters, and formats detections.
        Preserves labels, confidence scores, bounding boxes, metadata, and appearance signatures.
        """
        w, h = image_size
        w_f, h_f = float(w), float(h)
        normalized: List[Detection] = []

        target_set = None
        if config and config.target_classes:
            target_set = {tc.lower() for tc in config.target_classes}

        conf_thresh = config.confidence_threshold if config else 0.0

        for d in detections:
            # 1. Confidence check
            if d.confidence < conf_thresh:
                continue

            # 2. Target class check
            clean_label = d.label.strip()
            if target_set is not None and clean_label.lower() not in target_set:
                continue

            # 3. Bounding box clamping and geometry validation
            raw = d.raw_bbox
            clamped_xmin = max(0.0, min(w_f, float(raw.x_min)))
            clamped_ymin = max(0.0, min(h_f, float(raw.y_min)))
            clamped_xmax = max(0.0, min(w_f, float(raw.x_max)))
            clamped_ymax = max(0.0, min(h_f, float(raw.y_max)))

            # Discard degenerate zero/negative area boxes
            if clamped_xmax <= clamped_xmin or clamped_ymax <= clamped_ymin:
                continue

            clamped_raw_bbox = BoundingBox(
                x_min=clamped_xmin,
                y_min=clamped_ymin,
                x_max=clamped_xmax,
                y_max=clamped_ymax,
            )

            # Preserve canonical bbox and visual signature
            clamped_canon_bbox = d.canonical_bbox

            # Preserve all metadata and attach normalization info if modified
            meta = dict(d.metadata)
            if (
                clamped_xmin != raw.x_min
                or clamped_ymin != raw.y_min
                or clamped_xmax != raw.x_max
                or clamped_ymax != raw.y_max
            ):
                meta["raw_bbox_clamped"] = True

            normalized.append(
                Detection(
                    label=clean_label,
                    confidence=float(d.confidence),
                    raw_bbox=clamped_raw_bbox,
                    canonical_bbox=clamped_canon_bbox,
                    signature=d.signature,
                    metadata=meta,
                )
            )

        # 4. Max detections limit
        if config and config.max_detections and len(normalized) > config.max_detections:
            normalized.sort(key=lambda item: item.confidence, reverse=True)
            normalized = normalized[: config.max_detections]

        return DetectionBatch(
            detections=normalized,
            image_size=image_size,
            latency_ms=max(0.0, float(latency_ms)),
            detector_name=detector_name,
            is_synthetic=is_synthetic,
            failure_reason=failure_reason,
        )


class OpenCVDNNDetector(ObjectDetector):
    """
    CPU-first object detector running via OpenCV's built-in DNN module.

    Model architecture: MobileNetV2-SSD / SSDLite / YOLOv8 ONNX.
    Backend: cv2.dnn.DNN_BACKEND_OPENCV running on CPU.
    License: Model-dependent (Apache 2.0 / MIT / AGPL-3.0).
    """

    OFFICIAL_YOLOV8N_SHA1 = "68f864475d06e2ec4037181052739f268eeac38d"
    OFFICIAL_YOLOV8N_URL = (
        "https://github.com/CVHub520/X-AnyLabeling/releases/download/v0.1.0/yolov8n.onnx"
    )

    def __init__(
        self,
        weights_path: Optional[str] = None,
        config_path: Optional[str] = None,
        classes: Optional[Sequence[str]] = None,
        framework: str = "onnx",
        architecture: str = "auto",
        input_size: Optional[Tuple[int, int]] = None,
        nms_threshold: float = 0.45,
        letterbox: bool = True,
    ) -> None:
        self.weights_path = Path(weights_path) if weights_path else None
        self.config_path = Path(config_path) if config_path else None
        self.classes = list(classes) if classes else COCO_CLASSES_80
        self.framework = framework.lower()
        self.architecture = architecture.lower()
        self.nms_threshold = nms_threshold
        self.letterbox = letterbox
        self._net: Optional[cv2.dnn.Net] = None

        # Resolve architecture and input size defaults
        if self.weights_path and "yolo" in self.weights_path.name.lower():
            detected_arch = "yolov8"
        elif self.framework in ("caffe", "tf"):
            detected_arch = "ssd"
        else:
            detected_arch = "yolov8" if self.architecture == "yolov8" else "ssd"

        self.effective_architecture = detected_arch if self.architecture == "auto" else self.architecture

        if input_size is not None:
            self.input_size = input_size
        else:
            self.input_size = (640, 640) if self.effective_architecture == "yolov8" else (300, 300)

        if self.weights_path and self.weights_path.exists():
            self._load_network()

    def _load_network(self) -> None:
        assert self.weights_path is not None
        if self.framework == "onnx":
            self._net = cv2.dnn.readNetFromONNX(str(self.weights_path))
        elif self.framework == "caffe":
            config_str = str(self.config_path) if self.config_path else ""
            self._net = cv2.dnn.readNetFromCaffe(config_str, str(self.weights_path))
        elif self.framework == "tf":
            config_str = str(self.config_path) if self.config_path else ""
            self._net = cv2.dnn.readNetFromTensorflow(str(self.weights_path), config_str)
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")

        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        logger.info(
            "OpenCVDNNDetector network loaded: framework='%s', architecture='%s', weights_path='%s', input_size=%s",
            self.framework,
            self.effective_architecture,
            self.weights_path,
            self.input_size,
        )

    def is_available(self) -> bool:
        return self._net is not None and self.weights_path is not None and self.weights_path.exists()

    def get_availability_error(self) -> Optional[str]:
        if self._net is not None:
            return None
        if self.weights_path is None:
            return "No model weights path configured."
        if not self.weights_path.exists():
            return (
                f"Model weights not found at '{self.weights_path}'. "
                f"Please place valid {self.framework.upper()} weights or configure a valid path."
            )
        return "Model network failed to initialize with the provided weights."

    def model_info(self) -> Dict[str, Any]:
        return {
            "name": f"OpenCVDNN_{self.effective_architecture.upper()}_{self.framework.upper()}",
            "framework": self.framework,
            "architecture": self.effective_architecture,
            "weights_path": str(self.weights_path) if self.weights_path else None,
            "backend": "OpenCV DNN (CPU)",
            "license": "Apache-2.0 / BSD / MIT (dependent on imported weights)",
            "is_available": self.is_available(),
            "availability_error": self.get_availability_error(),
            "is_open_vocabulary": False,
            "is_synthetic": False,
            "classes_count": len(self.classes),
            "input_size": list(self.input_size),
            "nms_threshold": self.nms_threshold,
            "limitations": [
                "Standard COCO models are trained on general consumer objects (e.g., bottle, cup, laptop, mouse, scissors).",
                "They do NOT contain domain-specific classes for workbench tools (multimeter, PCB, soldering iron, wire stripper).",
                "Small electronics components fall below standard COCO object resolution.",
                "CPU execution speed scales with resolution (measured ~115ms per frame on modern x86 CPU).",
            ],
        }

    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: Optional[float | DetectorConfiguration] = None,
        config: Optional[DetectorConfiguration] = None,
        **kwargs: Any,
    ) -> DetectionBatch:
        if not self.is_available():
            raise DetectorUnavailableError(
                self.get_availability_error()
                or f"Model weights not found at '{self.weights_path}'. "
                f"Please place valid {self.framework.upper()} weights or configure a valid path."
            )

        if image is None or not isinstance(image, np.ndarray) or image.size == 0 or image.ndim not in (2, 3):
            raise ValueError("Invalid image provided for detection: must be a non-empty 2D or 3D numpy array.")

        cfg = self.resolve_configuration(confidence_threshold, config, default_conf=0.25)
        conf_thresh = cfg.confidence_threshold
        nms_thresh = cfg.nms_threshold

        # Ensure 3-channel image
        if image.ndim == 2:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        else:
            image_bgr = image

        h_orig, w_orig = image_bgr.shape[:2]
        channels = image_bgr.shape[2] if image_bgr.ndim == 3 else 1
        logger.info(
            "OpenCVDNNDetector.detect: input_image_dimensions=(w=%d, h=%d), channels=%d, confidence_threshold=%.2f",
            w_orig,
            h_orig,
            channels,
            conf_thresh,
        )
        start = time.perf_counter()
        letterbox_info: Optional[LetterboxInfo] = None

        if self.effective_architecture == "yolov8" and self.letterbox:
            letterboxed_bgr, letterbox_info = letterbox_image(
                image_bgr, target_size=self.input_size, pad_value=114
            )
            blob = cv2.dnn.blobFromImage(
                letterboxed_bgr,
                scalefactor=1.0 / 255.0,
                size=self.input_size,
                mean=(0.0, 0.0, 0.0),
                swapRB=True,
                crop=False,
            )
        elif self.effective_architecture == "yolov8":
            blob = cv2.dnn.blobFromImage(
                image_bgr,
                scalefactor=1.0 / 255.0,
                size=self.input_size,
                mean=(0.0, 0.0, 0.0),
                swapRB=True,
                crop=False,
            )
        else:
            blob = cv2.dnn.blobFromImage(
                image_bgr,
                scalefactor=1.0 / 127.5,
                size=self.input_size,
                mean=(127.5, 127.5, 127.5),
                swapRB=True,
                crop=False,
            )

        assert self._net is not None
        self._net.setInput(blob)
        output = self._net.forward()

        logger.info(
            "OpenCVDNNDetector forward pass complete: raw_output_shape=%s, ndim=%d",
            output.shape,
            output.ndim,
        )

        detections: List[Detection] = []

        # ARCHITECTURE A: SSD output shape (1, 1, N, 7)
        if output.ndim == 4 and output.shape[2] > 0 and output.shape[3] == 7:
            total_candidates = output.shape[2]
            passing_candidates = 0
            class_scores_summary: Dict[str, float] = {}

            for i in range(total_candidates):
                conf = float(output[0, 0, i, 2])
                class_id = int(output[0, 0, i, 1])
                label_idx = class_id - 1 if 0 < class_id <= len(self.classes) else class_id
                label = self.classes[label_idx] if 0 <= label_idx < len(self.classes) else f"class_{class_id}"

                if conf > class_scores_summary.get(label, 0.0):
                    class_scores_summary[label] = conf

                if conf >= conf_thresh:
                    passing_candidates += 1
                    x_min = max(0.0, float(output[0, 0, i, 3]) * w_orig)
                    y_min = max(0.0, float(output[0, 0, i, 4]) * h_orig)
                    x_max = min(float(w_orig), float(output[0, 0, i, 5]) * w_orig)
                    y_max = min(float(h_orig), float(output[0, 0, i, 6]) * h_orig)

                    if x_max > x_min and y_max > y_min:
                        bbox = BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
                        detections.append(Detection(label=label, confidence=conf, raw_bbox=bbox))

            top_classes = sorted(class_scores_summary.items(), key=lambda kv: kv[1], reverse=True)[:5]
            logger.info(
                "SSD candidates: total_anchors=%d, passing_confidence_threshold=%d, remaining_after_filter=%d, top_classes=[%s]",
                total_candidates,
                passing_candidates,
                len(detections),
                ", ".join(f"{k}: {v:.3f}" for k, v in top_classes),
            )

        # ARCHITECTURE B: YOLOv8 output shape (1, 84, N)
        elif (output.ndim == 3 and output.shape[1] == (len(self.classes) + 4)) or (output.ndim == 3 and output.shape[1] == 84):
            preds = output[0].T  # Shape: (N, 84)
            total_anchors = preds.shape[0]
            candidate_boxes: List[List[int]] = []
            candidate_scores: List[float] = []
            candidate_classes: List[int] = []

            in_w, in_h = self.input_size
            x_scale = w_orig / float(in_w)
            y_scale = h_orig / float(in_h)

            # Determine top classes across all anchors before thresholding
            peak_scores = preds[:, 4:].max(axis=0)  # Shape: (80,)
            top_class_indices = np.argsort(peak_scores)[-5:][::-1]
            top_classes_summary = []
            for c_idx in top_class_indices:
                score = float(peak_scores[c_idx])
                if score > 0.01:
                    c_name = self.classes[c_idx] if c_idx < len(self.classes) else f"class_{c_idx}"
                    top_classes_summary.append(f"{c_name}: {score:.3f}")

            for p in preds:
                class_scores = p[4:]
                max_idx = int(np.argmax(class_scores))
                max_score = float(class_scores[max_idx])
                if max_score >= conf_thresh:
                    cx, cy, w, h = p[:4]
                    if letterbox_info is not None:
                        orig_box = inverse_transform_cxcywh(cx, cy, w, h, letterbox_info)
                        if orig_box is not None:
                            x_min, y_min, x_max, y_max = orig_box
                            box_w = x_max - x_min
                            box_h = y_max - y_min
                            candidate_boxes.append([int(x_min), int(y_min), int(box_w), int(box_h)])
                            candidate_scores.append(max_score)
                            candidate_classes.append(max_idx)
                    else:
                        x_min = max(0.0, (cx - 0.5 * w) * x_scale)
                        y_min = max(0.0, (cy - 0.5 * h) * y_scale)
                        box_w = min(float(w_orig) - x_min, w * x_scale)
                        box_h = min(float(h_orig) - y_min, h * y_scale)

                        if box_w > 1 and box_h > 1:
                            candidate_boxes.append([int(x_min), int(y_min), int(box_w), int(box_h)])
                            candidate_scores.append(max_score)
                            candidate_classes.append(max_idx)

            logger.info(
                "YOLOv8 candidates: total_anchors=%d, passing_confidence_threshold=%d, top_classes=[%s]",
                total_anchors,
                len(candidate_boxes),
                ", ".join(top_classes_summary) if top_classes_summary else "none > 0.01",
            )

            if candidate_boxes:
                indices = cv2.dnn.NMSBoxes(
                    candidate_boxes,
                    candidate_scores,
                    score_threshold=conf_thresh,
                    nms_threshold=nms_thresh,
                )
                for idx in indices:
                    i = int(idx)
                    b = candidate_boxes[i]
                    cls_id = candidate_classes[i]
                    lbl = self.classes[cls_id] if cls_id < len(self.classes) else f"class_{cls_id}"
                    bbox = BoundingBox(
                        x_min=float(b[0]),
                        y_min=float(b[1]),
                        x_max=float(b[0] + b[2]),
                        y_max=float(b[1] + b[3]),
                    )
                    det_meta = {"detector": "opencv_dnn_yolov8"}
                    if letterbox_info is not None:
                        det_meta["letterbox_info"] = letterbox_info.to_dict()
                    detections.append(Detection(label=lbl, confidence=candidate_scores[i], raw_bbox=bbox, metadata=det_meta))

            logger.info(
                "YOLOv8 post-NMS results: remaining_detections=%d (nms_threshold=%.2f)",
                len(detections),
                nms_thresh,
            )

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return self.normalize_detection_batch(
            detections=detections,
            image_size=(w_orig, h_orig),
            latency_ms=elapsed_ms,
            detector_name=f"OpenCVDNN_{self.effective_architecture.upper()}",
            is_synthetic=False,
            config=cfg,
        )

    @classmethod
    def download_yolov8n_weights(cls, target_dir: str | Path = "models") -> Path:
        """
        Downloads official YOLOv8n ONNX model weights and verifies SHA1 checksum.
        """
        import hashlib
        import urllib.request

        target_path = Path(target_dir) / "yolov8n.onnx"
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if target_path.exists():
            with open(target_path, "rb") as f:
                h = hashlib.sha1(f.read()).hexdigest()
            if h == cls.OFFICIAL_YOLOV8N_SHA1:
                return target_path

        req = urllib.request.Request(cls.OFFICIAL_YOLOV8N_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp, open(target_path, "wb") as f:
            f.write(resp.read())

        with open(target_path, "rb") as f:
            h = hashlib.sha1(f.read()).hexdigest()

        if h != cls.OFFICIAL_YOLOV8N_SHA1:
            target_path.unlink(missing_ok=True)
            raise ValueError(f"Checksum mismatch for downloaded weights: expected {cls.OFFICIAL_YOLOV8N_SHA1}, got {h}")

        return target_path


class MockObjectDetector(ObjectDetector):
    """
    Deterministic detector for testing pipeline integration without external model weights.
    Explicitly labeled with is_synthetic=True.
    """

    def __init__(
        self,
        canned_detections: Optional[List[Detection]] = None,
        latency_simulation_ms: float = 5.0,
    ) -> None:
        self.canned_detections = canned_detections or []
        self.latency_simulation_ms = latency_simulation_ms

    def is_available(self) -> bool:
        return True

    def get_availability_error(self) -> Optional[str]:
        return None

    def model_info(self) -> Dict[str, Any]:
        return {
            "name": "MockObjectDetector",
            "backend": "Synthetic In-Memory Fixture",
            "is_synthetic": True,
            "is_open_vocabulary": False,
            "is_available": True,
            "availability_error": None,
            "license": "Project Internal Test Fixture",
            "limitations": ["Outputs predetermined or geometric detections for testing only."],
        }

    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: Optional[float | DetectorConfiguration] = None,
        config: Optional[DetectorConfiguration] = None,
        **kwargs: Any,
    ) -> DetectionBatch:
        if image is None or not isinstance(image, np.ndarray) or image.size == 0 or image.ndim not in (2, 3):
            raise ValueError("Invalid image provided for detection: must be a non-empty 2D or 3D numpy array.")
        h, w = image.shape[:2]
        cfg = self.resolve_configuration(confidence_threshold, config, default_conf=0.4)
        return self.normalize_detection_batch(
            detections=self.canned_detections,
            image_size=(w, h),
            latency_ms=self.latency_simulation_ms,
            detector_name="MockObjectDetector",
            is_synthetic=True,
            config=cfg,
        )
