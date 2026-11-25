"""
YOLO26m ONNX Runtime detector adapter for ObjectMemory.

Safeguards:
1. Uses official Ultralytics YOLO26m weights exported to ONNX (opset 12, 640x640).
2. Runs via ONNX Runtime CPUExecutionProvider for deterministic, exact decoding parity.
3. Lazy-loads the InferenceSession on first inference (or explicit ensure_loaded()) so
   listing detectors does not unnecessarily load large models into memory.
4. Logs model version, weights path, and SHA-256 integrity hash; never silently falls back.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from objectmemory.detection.detector import COCO_CLASSES_80, ObjectDetector
from objectmemory.detection.exceptions import (
    DetectorInferenceError,
    DetectorUnavailableError,
)
from objectmemory.detection.letterbox import inverse_transform_cxcywh, letterbox_image
from objectmemory.detection.models import (
    Detection,
    DetectionBatch,
    DetectorConfiguration,
)
from objectmemory.domain.models import BoundingBox

logger = logging.getLogger(__name__)


class YOLO26mDetector(ObjectDetector):
    """
    Pretrained YOLO26m (COCO 80-class) detector running on CPU via ONNX Runtime.
    """

    DEFAULT_WEIGHTS_PATH = Path("models/yolo26m.onnx")
    OFFICIAL_SOURCE_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt"
    MODEL_VERSION = "ultralytics-8.4.218 / yolo26m"
    EXPECTED_ONNX_SHA256 = "992d8f3a0ff47b0ce19a6b889f6ac2dc14c2088a955e7505775acead012fbbd8"
    LICENSE = "AGPL-3.0 (Ultralytics YOLO26)"

    def __init__(
        self,
        weights_path: Optional[str | Path] = None,
        classes: Optional[Sequence[str]] = None,
        input_size: Tuple[int, int] = (640, 640),
        nms_threshold: float = 0.45,
        lazy_load: bool = True,
    ) -> None:
        self.weights_path = Path(weights_path) if weights_path else self.DEFAULT_WEIGHTS_PATH
        self.classes = list(classes) if classes else list(COCO_CLASSES_80)
        self.input_size = tuple(input_size)
        self.nms_threshold = float(nms_threshold)
        self._session: Any = None
        self._input_name: Optional[str] = None
        self._output_name: Optional[str] = None
        self._load_error: Optional[str] = None
        self._weights_sha256: Optional[str] = None
        self._weights_size_bytes: Optional[int] = None

        if self.weights_path.exists():
            self._weights_size_bytes = self.weights_path.stat().st_size

        if not lazy_load and self.weights_path.exists():
            self.ensure_loaded()

    def _compute_sha256(self) -> Optional[str]:
        if self._weights_sha256 is not None:
            return self._weights_sha256
        if not self.weights_path.exists():
            return None
        h = hashlib.sha256()
        with self.weights_path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        self._weights_sha256 = h.hexdigest()
        return self._weights_sha256

    def ensure_loaded(self) -> None:
        """Loads the ONNX Runtime InferenceSession if not already loaded."""
        if self._session is not None:
            return
        if not self.weights_path.exists():
            self._load_error = (
                f"YOLO26m ONNX weights not found at '{self.weights_path}'. "
                f"Export from official '{self.OFFICIAL_SOURCE_URL}' via Ultralytics."
            )
            raise DetectorUnavailableError(self._load_error)

        try:
            import onnxruntime as ort
        except Exception as exc:
            self._load_error = f"onnxruntime is required for YOLO26mDetector: {exc}"
            raise DetectorUnavailableError(self._load_error) from exc

        try:
            sha256 = self._compute_sha256()
            sess_opts = ort.SessionOptions()
            session = ort.InferenceSession(
                str(self.weights_path),
                sess_options=sess_opts,
                providers=["CPUExecutionProvider"],
            )
            inputs = session.get_inputs()
            outputs = session.get_outputs()
            if not inputs or not outputs:
                raise RuntimeError("ONNX model has missing input or output tensors.")
            self._input_name = inputs[0].name
            self._output_name = outputs[0].name
            self._session = session
            self._load_error = None
            logger.info(
                "YOLO26mDetector loaded: version='%s', weights='%s', sha256='%s', input='%s' %s, output='%s' %s",
                self.MODEL_VERSION,
                self.weights_path,
                sha256,
                self._input_name,
                inputs[0].shape,
                self._output_name,
                outputs[0].shape,
            )
        except Exception as exc:
            self._session = None
            self._load_error = f"Failed to load YOLO26m ONNX model from '{self.weights_path}': {exc}"
            logger.error(self._load_error)
            raise DetectorUnavailableError(self._load_error) from exc

    def unload(self) -> None:
        """Releases the ONNX Runtime session to free memory when switching models."""
        self._session = None
        self._input_name = None
        self._output_name = None

    def is_available(self) -> bool:
        if self._load_error is not None:
            return False
        if not self.weights_path.exists():
            return False
        try:
            import onnxruntime  # noqa: F401
        except Exception:
            return False
        return True

    def get_availability_error(self) -> Optional[str]:
        if self._load_error is not None:
            return self._load_error
        if not self.weights_path.exists():
            return (
                f"YOLO26m ONNX weights not found at '{self.weights_path}'. "
                f"Export from official '{self.OFFICIAL_SOURCE_URL}'."
            )
        try:
            import onnxruntime  # noqa: F401
        except Exception as exc:
            return f"onnxruntime is unavailable: {exc}"
        return None

    def model_info(self) -> Dict[str, Any]:
        sha256 = self._compute_sha256() if self.weights_path.exists() else None
        return {
            "name": "YOLO26m_ONNXRUNTIME",
            "framework": "onnxruntime",
            "architecture": "yolo26m",
            "model_version": self.MODEL_VERSION,
            "official_source": self.OFFICIAL_SOURCE_URL,
            "weights_path": str(self.weights_path),
            "weights_sha256": sha256,
            "weights_size_bytes": self._weights_size_bytes,
            "backend": "ONNX Runtime (CPUExecutionProvider)",
            "license": self.LICENSE,
            "is_available": self.is_available(),
            "is_loaded": self._session is not None,
            "availability_error": self.get_availability_error(),
            "is_open_vocabulary": False,
            "is_synthetic": False,
            "classes_count": len(self.classes),
            "input_size": list(self.input_size),
            "nms_threshold": self.nms_threshold,
            "output_format": "(1, 84, 8400)",
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

        self.ensure_loaded()
        assert self._session is not None and self._input_name is not None and self._output_name is not None

        cfg = self.resolve_configuration(confidence_threshold, config, default_conf=0.25)
        conf_thresh = cfg.confidence_threshold
        nms_thresh = cfg.nms_threshold if config is not None else self.nms_threshold

        if image.ndim == 2:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        else:
            image_bgr = image

        h_orig, w_orig = image_bgr.shape[:2]
        t0 = time.perf_counter()

        try:
            lb_img, lb_info = letterbox_image(image_bgr, self.input_size)
            blob = cv2.dnn.blobFromImage(
                lb_img,
                scalefactor=1.0 / 255.0,
                size=self.input_size,
                swapRB=True,
                crop=False,
            )
            raw_out = self._session.run([self._output_name], {self._input_name: blob})[0]
        except Exception as exc:
            raise DetectorInferenceError(f"YOLO26m ONNX inference failed: {exc}") from exc

        detections = self._decode_predictions(
            raw_out=raw_out,
            lb_info=lb_info,
            conf_thresh=conf_thresh,
            nms_thresh=nms_thresh,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return self.normalize_detection_batch(
            detections=detections,
            image_size=(w_orig, h_orig),
            latency_ms=latency_ms,
            detector_name="yolo26m",
            is_synthetic=False,
            config=cfg,
        )

    def _decode_predictions(
        self,
        raw_out: np.ndarray,
        lb_info: Any,
        conf_thresh: float,
        nms_thresh: float,
    ) -> List[Detection]:
        if raw_out.ndim != 3 or raw_out.shape[0] != 1:
            raise DetectorInferenceError(f"Unexpected YOLO26m output shape: {raw_out.shape}")

        # Expected (1, 84, 8400) -> transpose to (8400, 84); support (1, 8400, 84) defensively.
        if raw_out.shape[1] == 4 + len(self.classes):
            preds = raw_out[0].T
        elif raw_out.shape[2] == 4 + len(self.classes):
            preds = raw_out[0]
        else:
            raise DetectorInferenceError(
                f"YOLO26m output channel count mismatch: shape={raw_out.shape}, expected 4+{len(self.classes)}."
            )

        boxes_cxcywh = preds[:, :4]
        scores = preds[:, 4:]
        cls_ids = np.argmax(scores, axis=1)
        confs = scores[np.arange(len(scores)), cls_ids]

        mask = confs >= conf_thresh
        if not np.any(mask):
            return []

        f_boxes = boxes_cxcywh[mask]
        f_confs = confs[mask]
        f_cls = cls_ids[mask]

        xywh_list = [
            [float(cx - w / 2.0), float(cy - h / 2.0), float(w), float(h)]
            for cx, cy, w, h in f_boxes
        ]
        indices = cv2.dnn.NMSBoxes(xywh_list, f_confs.tolist(), conf_thresh, nms_thresh)
        if len(indices) == 0:
            return []

        detections: List[Detection] = []
        for idx in indices.flatten():
            cx, cy, w, h = f_boxes[idx]
            x1, y1, x2, y2 = inverse_transform_cxcywh(float(cx), float(cy), float(w), float(h), lb_info)
            cls_idx = int(f_cls[idx])
            label = self.classes[cls_idx] if 0 <= cls_idx < len(self.classes) else f"class_{cls_idx}"
            detections.append(
                Detection(
                    label=label,
                    confidence=float(f_confs[idx]),
                    raw_bbox=BoundingBox(x_min=x1, y_min=y1, x_max=x2, y_max=y2),
                    metadata={"class_id": cls_idx, "model": "yolo26m"},
                )
            )
        return detections
