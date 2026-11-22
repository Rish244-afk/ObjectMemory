"""
YOLO-World Open-Vocabulary Detector implementation for ObjectMemory.

Safeguards:
1. Operates strictly on CPU via ONNX Runtime without requiring full PyTorch or GPU.
2. Accepts dynamic open-vocabulary text prompts encoded via a quantized CLIP ViT-B/32 text model.
3. Does not fabricate detections when weights are missing; raises DetectorUnavailableError explicitly.
4. Preserves backward compatibility with the ObjectDetector base interface and normalization contract.
5. Caches text prompt embeddings across inference calls to minimize latency.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np

from objectmemory.detection.detector import ObjectDetector
from objectmemory.detection.exceptions import (
    DetectorConfigurationError,
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

# Default domain vocabulary for common desk/workbench objects
DEFAULT_WORLD_CLASSES: List[str] = [
    "notebook",
    "book",
    "bottle",
    "cup",
    "scissors",
    "drawing compass",
    "pliers",
    "screwdriver",
    "hammer",
    "tape",
    "pen",
    "pencil",
    "phone",
    "laptop",
    "mouse",
    "keyboard",
    "box",
    "container",
    "tool",
    "metal object",
    "wooden object",
]


class YOLOWorldDetector(ObjectDetector):
    """
    Open-vocabulary zero-shot object detector using YOLOv8-World v2 Small
    and quantized CLIP ViT-B/32 text encoder on ONNX Runtime CPU.
    """

    OFFICIAL_YOLO_WORLD_SHA256 = "381ced485b23ed8f06de3e82bb2745e1420c181c64f0a176784c34a959d550a1"

    def __init__(
        self,
        weights_path: Optional[str | Path] = None,
        clip_model_path: Optional[str | Path] = None,
        tokenizer_path: Optional[str | Path] = None,
        default_prompts: Optional[Sequence[str]] = None,
        input_size: Tuple[int, int] = (640, 640),
        nms_threshold: float = 0.45,
        letterbox: bool = True,
    ) -> None:
        self.weights_path = Path(weights_path) if weights_path else Path("models/yolov8s-worldv2.onnx")
        self.clip_model_path = (
            Path(clip_model_path) if clip_model_path else Path("models/clip_text_quantized.onnx")
        )
        self.tokenizer_path = (
            Path(tokenizer_path) if tokenizer_path else Path("models/clip_tokenizer.json")
        )
        self.default_prompts = list(default_prompts) if default_prompts else list(DEFAULT_WORLD_CLASSES)
        self.input_size = input_size
        self.nms_threshold = nms_threshold
        self.letterbox = letterbox

        self._yolo_session: Any = None
        self._clip_session: Any = None
        self._tokenizer: Any = None
        self._embedding_cache: Dict[Tuple[str, ...], np.ndarray] = {}

        if self.is_available():
            self._load_sessions()

    @property
    def default_confidence_threshold(self) -> float:
        # Open-vocabulary zero-shot classification scores are typically lower than closed-set COCO.
        # A threshold of 0.15 is appropriate for zero-shot text-prompt matching.
        return 0.15

    def is_available(self) -> bool:
        """Checks if all required model artifacts and ONNX runtime are present."""
        try:
            import onnxruntime  # noqa: F401
            from tokenizers import Tokenizer  # noqa: F401
        except ImportError:
            return False

        return (
            self.weights_path.is_file()
            and self.clip_model_path.is_file()
            and self.tokenizer_path.is_file()
        )

    def get_availability_error(self) -> Optional[str]:
        """Returns human-readable explanation if unavailable, or None if ready."""
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            return "Missing Python package 'onnxruntime'. Please install via: pip install onnxruntime"

        try:
            from tokenizers import Tokenizer  # noqa: F401
        except ImportError:
            return "Missing Python package 'tokenizers'. Please install via: pip install tokenizers"

        if not self.weights_path.is_file():
            return f"YOLO-World ONNX model not found at {self.weights_path}."
        if not self.clip_model_path.is_file():
            return f"CLIP text encoder ONNX model not found at {self.clip_model_path}."
        if not self.tokenizer_path.is_file():
            return f"CLIP tokenizer JSON not found at {self.tokenizer_path}."

        return None

    def _load_sessions(self) -> None:
        """Loads ONNX Runtime inference sessions and BPE tokenizer."""
        import onnxruntime as ort
        from tokenizers import Tokenizer

        # Configure session options for efficient multi-threaded CPU inference
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = max(1, min(4, (ort.get_available_providers() and 4) or 2))
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._yolo_session = ort.InferenceSession(
            str(self.weights_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self._clip_session = ort.InferenceSession(
            str(self.clip_model_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )

        tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        tokenizer.enable_padding(length=77, pad_id=0)
        tokenizer.enable_truncation(max_length=77)
        self._tokenizer = tokenizer

        logger.info(
            "YOLOWorldDetector initialized successfully with weights=%s, clip_model=%s",
            self.weights_path,
            self.clip_model_path,
        )

    def _get_text_embeddings(self, prompts: Sequence[str]) -> np.ndarray:
        """
        Encodes a list of text prompts into L2-normalized 512-dim CLIP embeddings.
        Uses an in-memory cache to avoid repeated forward passes for the same prompt set.
        Output shape: (1, num_prompts, 512) float32.
        """
        key = tuple(p.strip().lower() for p in prompts)
        if key in self._embedding_cache:
            return self._embedding_cache[key]

        encoded = self._tokenizer.encode_batch(list(prompts))
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)

        text_embeds = self._clip_session.run(
            ["text_embeds"],
            {"input_ids": input_ids},
        )[0]  # Shape: (num_prompts, 512)

        # L2 normalization along feature axis
        norms = np.linalg.norm(text_embeds, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        normalized_embeds = (text_embeds / norms).astype(np.float32)

        txt_feats = normalized_embeds[np.newaxis, ...]  # Shape: (1, num_prompts, 512)
        self._embedding_cache[key] = txt_feats
        return txt_feats

    def model_info(self) -> Dict[str, Any]:
        """Returns metadata about the YOLO-World detector configuration and state."""
        return {
            "name": "yolo-world",
            "model_type": "yolov8s-worldv2",
            "framework": "onnxruntime-cpu",
            "weights_path": str(self.weights_path),
            "clip_model_path": str(self.clip_model_path),
            "tokenizer_path": str(self.tokenizer_path),
            "default_classes": self.default_prompts,
            "input_size": list(self.input_size),
            "is_open_vocabulary": True,
            "is_synthetic": False,
            "is_available": self.is_available(),
            "availability_error": self.get_availability_error(),
            "license": "AGPL-3.0",
        }

    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: Optional[float] = None,
        config: Optional[DetectorConfiguration] = None,
        **kwargs: Any,
    ) -> DetectionBatch:
        """
        Runs open-vocabulary object detection using YOLO-World v2 Small.
        """
        if not self.is_available():
            err = self.get_availability_error() or "YOLO-World detector is unavailable."
            raise DetectorUnavailableError(f"Cannot perform detection: {err}")

        if self._yolo_session is None or self._clip_session is None or self._tokenizer is None:
            self._load_sessions()

        if image is None or not isinstance(image, np.ndarray) or image.size == 0 or image.ndim not in (2, 3):
            raise ValueError("Invalid image provided: must be a non-empty 2D or 3D numpy array.")

        cfg = self.resolve_configuration(confidence_threshold, config, default_conf=self.default_confidence_threshold)
        conf_thresh = cfg.confidence_threshold
        nms_thresh = cfg.nms_threshold

        # Resolve text category prompts
        prompts: List[str] = []
        if cfg.target_classes:
            prompts = [str(c).strip() for c in cfg.target_classes if str(c).strip()]
        elif "prompts" in cfg.extra_params and isinstance(cfg.extra_params["prompts"], (list, tuple)):
            prompts = [str(c).strip() for c in cfg.extra_params["prompts"] if str(c).strip()]
        elif "classes" in cfg.extra_params and isinstance(cfg.extra_params["classes"], (list, tuple)):
            prompts = [str(c).strip() for c in cfg.extra_params["classes"] if str(c).strip()]
        else:
            prompts = list(self.default_prompts)

        if not prompts:
            logger.warning("YOLOWorldDetector called with empty category prompts list.")
            return self.normalize_detection_batch(
                detections=[],
                image_size=(image.shape[1], image.shape[0]),
                latency_ms=0.0,
                detector_name="yolo-world",
                config=cfg,
                is_synthetic=False,
            )

        # Preprocess input image to RGB and resize to (640, 640)
        h_orig, w_orig = image.shape[:2]
        if image.ndim == 2:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            # Assume BGR input from OpenCV cv2.imread
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unsupported image shape {image.shape}")

        start_time = time.perf_counter()

        in_w, in_h = self.input_size
        letterbox_info: Optional[LetterboxInfo] = None
        if self.letterbox:
            img_letterboxed, letterbox_info = letterbox_image(
                img_rgb, target_size=self.input_size, pad_value=114
            )
            img_norm = (img_letterboxed.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]
        else:
            img_resized = cv2.resize(img_rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
            img_norm = (img_resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]

        # Obtain dynamic text embeddings for the current categories
        txt_feats = self._get_text_embeddings(prompts)

        # Run YOLO-World inference
        try:
            outputs = self._yolo_session.run(
                None,
                {
                    "images": img_norm,
                    "txt_feats": txt_feats,
                },
            )
        except Exception as ex:
            raise DetectorInferenceError(f"YOLO-World ONNX inference execution failed: {ex}") from ex

        # Decode output tensor: shape is (1, 4 + num_prompts, num_anchors)
        raw_output = outputs[0]
        if raw_output.ndim != 3 or raw_output.shape[1] < 5:
            raise DetectorInferenceError(f"Unexpected YOLO-World output tensor shape: {raw_output.shape}")

        preds = raw_output[0].T  # Shape: (num_anchors, 4 + num_prompts)
        total_anchors = preds.shape[0]

        candidate_boxes: List[List[int]] = []
        candidate_scores: List[float] = []
        candidate_classes: List[int] = []

        x_scale = float(w_orig) / float(in_w)
        y_scale = float(h_orig) / float(in_h)

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

        detections: List[Detection] = []
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
                label = prompts[cls_id] if 0 <= cls_id < len(prompts) else f"class_{cls_id}"
                score = candidate_scores[i]

                x_min, y_min, bw, bh = b
                x_max = min(float(w_orig), float(x_min + bw))
                y_max = min(float(h_orig), float(y_min + bh))

                bbox = BoundingBox(x_min=float(x_min), y_min=float(y_min), x_max=x_max, y_max=y_max)
                det_meta = {
                    "detector": "yolo-world",
                    "open_vocabulary": True,
                    "prompt_idx": cls_id,
                }
                if letterbox_info is not None:
                    det_meta["letterbox_info"] = letterbox_info.to_dict()

                detections.append(
                    Detection(
                        label=label,
                        confidence=score,
                        raw_bbox=bbox,
                        metadata=det_meta,
                    )
                )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        logger.info(
            "YOLOWorldDetector.detect complete: prompts=%d, total_anchors=%d, candidates=%d, detections=%d, latency=%.1fms",
            len(prompts),
            total_anchors,
            len(candidate_boxes),
            len(detections),
            latency_ms,
        )

        return self.normalize_detection_batch(
            detections=detections,
            image_size=(w_orig, h_orig),
            latency_ms=latency_ms,
            detector_name="yolo-world",
            config=cfg,
            is_synthetic=False,
        )
