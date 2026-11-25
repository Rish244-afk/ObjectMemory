"""
Central detector registry and factory for ObjectMemory.

Safeguards:
1. Thread-safe registry for pluggable object detectors.
2. Preserves YOLOv8n baseline as default detector.
3. Provides unambiguous error reporting when models or weights are unavailable.
4. Cleanly decouples pipeline runners from concrete detector model implementations.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from objectmemory.detection.detector import (
    MockObjectDetector,
    ObjectDetector,
    OpenCVDNNDetector,
)
from objectmemory.detection.exceptions import DetectorUnavailableError

logger = logging.getLogger(__name__)

DetectorFactory = Callable[..., ObjectDetector]


class DetectorRegistry:
    """
    Central registry and factory for ObjectDetector implementations.
    """

    _lock: threading.Lock = threading.Lock()
    _factories: Dict[str, Tuple[DetectorFactory, Dict[str, Any]]] = {}
    _instance_cache: Dict[str, ObjectDetector] = {}
    _default_name: str = "yolov8m"

    @classmethod
    def register(
        cls,
        name: str,
        factory: DetectorFactory,
        metadata: Optional[Dict[str, Any]] = None,
        is_default: bool = False,
    ) -> None:
        """
        Registers a detector factory function under a unique name.
        """
        key = name.strip().lower()
        if not key:
            raise ValueError("Detector registration name cannot be empty.")

        meta = metadata or {}
        with cls._lock:
            cls._factories[key] = (factory, meta)
            if is_default:
                cls._default_name = key
        logger.info("Registered detector '%s' (is_default=%s)", key, is_default)

    @classmethod
    def unregister(cls, name: str) -> None:
        """Unregisters a detector by name."""
        key = name.strip().lower()
        with cls._lock:
            cls._instance_cache.pop(key, None)
            if key in cls._factories:
                del cls._factories[key]
                if cls._default_name == key:
                    cls._default_name = "yolov8m" if "yolov8m" in cls._factories else ("yolov8n" if "yolov8n" in cls._factories else "")

    @classmethod
    def get(cls, name: Optional[str] = None, **kwargs: Any) -> ObjectDetector:
        """
        Instantiates and returns the requested detector.
        If name is None or 'default', instantiates the default registered detector.
        Caches default instances so weights are not re-read from disk on every invocation.
        """
        target_name = (name or cls._default_name or "yolov8m").strip().lower()
        if target_name == "default":
            target_name = cls._default_name

        with cls._lock:
            if target_name not in cls._factories:
                avail = sorted(cls._factories.keys())
                raise KeyError(
                    f"Detector '{target_name}' is not registered. Available detectors: {avail}"
                )
            factory, _ = cls._factories[target_name]

            # Return cached instance if standard arguments (no overrides)
            if not kwargs and target_name in cls._instance_cache:
                return cls._instance_cache[target_name]

        instance = factory(**kwargs)
        if not kwargs:
            with cls._lock:
                cls._instance_cache[target_name] = instance
        return instance

    @classmethod
    def set_default(cls, name: str) -> None:
        """Sets the default detector name."""
        key = name.strip().lower()
        with cls._lock:
            if key not in cls._factories:
                raise KeyError(f"Cannot set default to unregistered detector '{key}'.")
            cls._default_name = key

    @classmethod
    def get_default_name(cls) -> str:
        """Returns the current default detector name."""
        with cls._lock:
            return cls._default_name

    @classmethod
    def get_default(cls, **kwargs: Any) -> ObjectDetector:
        """Returns the default registered detector instance."""
        return cls.get(name=None, **kwargs)

    @classmethod
    def list_names(cls) -> List[str]:
        """Returns the list of all registered detector names."""
        with cls._lock:
            return sorted(cls._factories.keys())

    @classmethod
    def list_available(cls) -> List[Dict[str, Any]]:
        """
        Inspects all registered detectors and returns status, availability,
        open-vocabulary capability, and metadata.
        Uses cached detector instances to avoid disk I/O on every status check.
        """
        results: List[Dict[str, Any]] = []
        with cls._lock:
            items = list(cls._factories.items())
            default_key = cls._default_name

        for name, (_, meta) in items:
            is_avail = False
            error_msg: Optional[str] = None
            model_info: Dict[str, Any] = {}

            try:
                instance = cls.get(name)
                is_avail = instance.is_available()
                error_msg = instance.get_availability_error()
                model_info = instance.model_info()
            except Exception as ex:
                is_avail = False
                error_msg = str(ex)
                model_info = {"error": str(ex)}

            results.append(
                {
                    "name": name,
                    "is_default": (name == default_key),
                    "is_available": is_avail,
                    "availability_error": error_msg,
                    "is_open_vocabulary": meta.get("is_open_vocabulary", False),
                    "is_synthetic": meta.get("is_synthetic", False),
                    "description": meta.get("description", ""),
                    "model_info": model_info,
                }
            )

        return results

    @classmethod
    def reset_to_defaults(cls) -> None:
        """Resets the registry to standard built-in detectors."""
        with cls._lock:
            cls._factories.clear()
            cls._instance_cache.clear()
            cls._default_name = "yolov8m"

        # 1. Register YOLOv8m production default
        def yolov8m_factory(
            weights_path: Optional[str | Path] = None,
            framework: str = "onnx",
            architecture: str = "yolov8",
            **kwargs: Any,
        ) -> ObjectDetector:
            path = Path(weights_path) if weights_path else Path("models/yolov8m.onnx")
            return OpenCVDNNDetector(
                weights_path=str(path),
                framework=framework,
                architecture=architecture,
                **kwargs,
            )

        cls.register(
            "yolov8m",
            yolov8m_factory,
            metadata={
                "description": "Pretrained YOLOv8 Medium running via OpenCV DNN on CPU (80 COCO classes, high accuracy).",
                "is_open_vocabulary": False,
                "is_synthetic": False,
            },
            is_default=True,
        )

        # 2. Register YOLOv8n baseline
        def yolov8n_factory(
            weights_path: Optional[str | Path] = None,
            framework: str = "onnx",
            architecture: str = "yolov8",
            **kwargs: Any,
        ) -> ObjectDetector:
            path = Path(weights_path) if weights_path else Path("models/yolov8n.onnx")
            return OpenCVDNNDetector(
                weights_path=str(path),
                framework=framework,
                architecture=architecture,
                **kwargs,
            )

        cls.register(
            "yolov8n",
            yolov8n_factory,
            metadata={
                "description": "Pretrained YOLOv8 Nano running via OpenCV DNN on CPU (80 COCO classes, lightweight).",
                "is_open_vocabulary": False,
                "is_synthetic": False,
            },
            is_default=False,
        )

        # 3. Alias yolov8 -> yolov8m
        cls.register(
            "yolov8",
            yolov8m_factory,
            metadata={
                "description": "Alias for yolov8m production default detector.",
                "is_open_vocabulary": False,
                "is_synthetic": False,
            },
            is_default=False,
        )

        # 3b. Register YOLO26m optional alternative
        def yolo26m_factory(
            weights_path: Optional[str | Path] = None,
            **kwargs: Any,
        ) -> ObjectDetector:
            from objectmemory.detection.yolo26 import YOLO26mDetector

            return YOLO26mDetector(
                weights_path=weights_path,
                **kwargs,
            )

        cls.register(
            "yolo26m",
            yolo26m_factory,
            metadata={
                "description": "Pretrained YOLO26 Medium running via ONNX Runtime on CPU (80 COCO classes, optional alternative).",
                "is_open_vocabulary": False,
                "is_synthetic": False,
            },
            is_default=False,
        )

        # 3. Register Mock detector
        def mock_factory(**kwargs: Any) -> ObjectDetector:
            return MockObjectDetector(**kwargs)

        cls.register(
            "mock",
            mock_factory,
            metadata={
                "description": "Deterministic mock detector for unit testing and offline fixtures.",
                "is_open_vocabulary": False,
                "is_synthetic": True,
            },
            is_default=False,
        )

        # 4. Register YOLO-World open-vocabulary detector
        def yolo_world_factory(
            weights_path: Optional[str | Path] = None,
            clip_model_path: Optional[str | Path] = None,
            tokenizer_path: Optional[str | Path] = None,
            default_prompts: Optional[Sequence[str]] = None,
            **kwargs: Any,
        ) -> ObjectDetector:
            from objectmemory.detection.yolo_world import YOLOWorldDetector

            return YOLOWorldDetector(
                weights_path=weights_path,
                clip_model_path=clip_model_path,
                tokenizer_path=tokenizer_path,
                default_prompts=default_prompts,
                **kwargs,
            )

        cls.register(
            "yolo-world",
            yolo_world_factory,
            metadata={
                "description": "YOLO-World v2 Small open-vocabulary detector running via ONNX Runtime CPU with dynamic CLIP text prompts.",
                "is_open_vocabulary": True,
                "is_synthetic": False,
            },
            is_default=False,
        )

        cls.register(
            "yolov8-world",
            yolo_world_factory,
            metadata={
                "description": "Alias for yolo-world open-vocabulary detector.",
                "is_open_vocabulary": True,
                "is_synthetic": False,
            },
            is_default=False,
        )


# Initialize built-ins on module load
DetectorRegistry.reset_to_defaults()
