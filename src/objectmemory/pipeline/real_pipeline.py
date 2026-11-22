"""
End-to-end processing pipeline for real multi-visit workbench imagery.

Coordinates:
1. Real object detection (OpenCV DNN CPU inference).
2. Planar image co-registration and alignment quality gating.
3. Bounding-box projection into canonical reference coordinates.
4. Visual appearance signature extraction from real pixel crops.
5. Cross-visit identity association and multi-visit temporal belief updating.
6. Atomic SQLite persistence of sessions, observations, entities, and change events.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from objectmemory.datasets.real_dataset import RealVisitFrame, RealWorkbenchDataset
from objectmemory.detection.detector import ObjectDetector
from objectmemory.detection.models import (
    Detection,
    DetectionBatch,
    DetectorConfiguration,
)
from objectmemory.detection.pipeline import is_bbox_boundary_clipped, transform_bbox_to_canonical
from objectmemory.detection.signature import VisualSignatureExtractor
from objectmemory.domain.models import BoundingBox
from objectmemory.domain.states import BaselineChangeType, EpistemicState
from objectmemory.persistence.repository import SQLiteMemoryStore
from objectmemory.registration.keyframe import AdaptiveKeyframeManager, KeyframeConfig
from objectmemory.registration.models import AlignmentStatus, RegistrationResult
from objectmemory.registration.registrar import PlanarRegistrar
from objectmemory.temporal.engine import TemporalMemoryEngine
from objectmemory.temporal.models import TemporalEngineConfig, VisitInput, VisitProcessingReport


@dataclass
class RealVisitProcessingResult:
    """Outcome of running the end-to-end pipeline on one physical visit frame."""
    visit_id: str
    session_name: str
    timestamp: str
    registration: RegistrationResult
    detection_batch: DetectionBatch
    temporal_report: VisitProcessingReport
    latency_breakdown_ms: Dict[str, float]
    total_latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "visit_id": self.visit_id,
            "session_name": self.session_name,
            "timestamp": self.timestamp,
            "registration_status": self.registration.status.value,
            "registration_inlier_ratio": self.registration.inlier_ratio,
            "detections_count": len(self.detection_batch.detections),
            "new_entities": self.temporal_report.new_entities_created,
            "transitions_count": self.temporal_report.state_transitions_count,
            "total_latency_ms": self.total_latency_ms,
            "latency_breakdown_ms": self.latency_breakdown_ms,
        }


@dataclass
class RealPipelineDatasetReport:
    """Comprehensive summary across all visits in a real-image dataset."""
    dataset_name: str
    total_visits_processed: int
    visit_results: List[RealVisitProcessingResult]
    total_active_entities: int
    entity_states_summary: Dict[str, int]
    mean_pipeline_latency_ms: float
    cpu_profile: Dict[str, Any]


class RealImagePipelineRunner:
    """
    Executes the end-to-end ObjectMemory pipeline on real-world workbench imagery.
    """

    def __init__(
        self,
        store: SQLiteMemoryStore,
        detector: ObjectDetector,
        registrar: Optional[PlanarRegistrar] = None,
        keyframe_manager: Optional[AdaptiveKeyframeManager] = None,
        keyframe_config: Optional[KeyframeConfig] = None,
        signature_extractor: Optional[VisualSignatureExtractor] = None,
        temporal_config: Optional[TemporalEngineConfig] = None,
    ) -> None:
        self.store = store
        self.detector = detector
        self.registrar = registrar or PlanarRegistrar()
        self.keyframe_config = keyframe_config or KeyframeConfig()
        self.keyframe_manager = keyframe_manager or AdaptiveKeyframeManager(
            registrar=self.registrar,
            config=self.keyframe_config,
        )
        self.signature_extractor = signature_extractor or VisualSignatureExtractor()
        self.temporal_config = temporal_config or TemporalEngineConfig()
        self.engine = TemporalMemoryEngine(
            store=self.store,
            config=self.temporal_config,
            signature_extractor=self.signature_extractor,
        )

        self._ref_image: Optional[np.ndarray] = None
        self._ref_visit_id: Optional[str] = None
        self.processed_visits_count: int = 0

    @property
    def ref_image(self) -> Optional[np.ndarray]:
        if self._ref_image is not None:
            return self._ref_image
        if self.keyframe_manager and self.keyframe_manager.canonical_keyframe_id:
            kf = self.keyframe_manager.keyframes.get(self.keyframe_manager.canonical_keyframe_id)
            if kf is not None:
                return kf.image
        return None

    @ref_image.setter
    def ref_image(self, img: Optional[np.ndarray]) -> None:
        self._ref_image = img
        if img is not None and not self.keyframe_manager.keyframes:
            self.keyframe_manager.set_canonical_reference(
                image=img,
                visit_id=self._ref_visit_id or "visit_01",
                timestamp="1970-01-01T00:00:00Z",
            )

    @property
    def ref_visit_id(self) -> Optional[str]:
        return self._ref_visit_id or self.keyframe_manager.canonical_keyframe_id

    @ref_visit_id.setter
    def ref_visit_id(self, vid: Optional[str]) -> None:
        self._ref_visit_id = vid

    def process_visit(
        self,
        visit_frame: RealVisitFrame,
        confidence_threshold: float = 0.25,
        detector_config: Optional[DetectorConfiguration] = None,
    ) -> RealVisitProcessingResult:
        """
        Executes end-to-end processing for a single visit frame.
        """
        t0 = time.perf_counter()
        img = visit_frame.load_image()
        h_img, w_img = img.shape[:2]

        latencies: Dict[str, float] = {}

        # 1. Co-Registration against keyframe topology
        t_reg_start = time.perf_counter()
        kf_res = self.keyframe_manager.register_query(
            query_image=img,
            visit_id=visit_frame.visit_id,
            timestamp=visit_frame.timestamp,
            metadata=visit_frame.metadata,
        )
        reg_result = kf_res.registration_to_canonical

        if self._ref_image is None and self.keyframe_manager.canonical_keyframe_id:
            canon_kf = self.keyframe_manager.keyframes[self.keyframe_manager.canonical_keyframe_id]
            self._ref_image = canon_kf.image
            self._ref_visit_id = canon_kf.keyframe_id

        latencies["registration_ms"] = (time.perf_counter() - t_reg_start) * 1000.0

        # 2. Real Object Detection
        t_det_start = time.perf_counter()
        if detector_config is not None:
            detection_batch = self.detector.detect(img, config=detector_config)
        else:
            detection_batch = self.detector.detect(img, confidence_threshold=confidence_threshold)
        latencies["detection_ms"] = (time.perf_counter() - t_det_start) * 1000.0

        # 3. Canonical Projection & Signature Extraction
        t_feat_start = time.perf_counter()
        ref_w, ref_h = (self.ref_image.shape[1], self.ref_image.shape[0]) if self.ref_image is not None else (w_img, h_img)

        processed_detections: List[Detection] = []
        for det in detection_batch.detections:
            # Spatial mapping into canonical reference coordinate frame (only if registration succeeded and is reliable)
            canon_box: Optional[BoundingBox] = None
            is_reg_reliable = (
                reg_result.success
                and reg_result.homography is not None
                and (
                    reg_result.inlier_ratio is None
                    or reg_result.inlier_ratio >= self.temporal_config.min_registration_inlier_ratio
                )
            )
            if is_reg_reliable:
                canon_box = transform_bbox_to_canonical(
                    det.raw_bbox,
                    reg_result.homography,
                    ref_size=(ref_w, ref_h),
                )

            # Extract 148-d appearance signature from real image crop
            sig = self.signature_extractor.extract(img, det.raw_bbox)
            is_clipped = is_bbox_boundary_clipped(det.raw_bbox, (w_img, h_img), margin=self.temporal_config.border_margin_px)

            processed_detections.append(
                Detection(
                    label=det.label,
                    confidence=det.confidence,
                    raw_bbox=det.raw_bbox,
                    canonical_bbox=canon_box,
                    signature=sig,
                    metadata={
                        "raw_detector": detection_batch.detector_name,
                        "is_clipped": is_clipped,
                    },
                )
            )
        latencies["feature_extraction_ms"] = (time.perf_counter() - t_feat_start) * 1000.0

        # 4. Multi-Visit Temporal Memory Association & Belief Updates
        t_temp_start = time.perf_counter()
        visit_input = VisitInput(
            session_id=visit_frame.visit_id,
            session_name=visit_frame.session_name,
            timestamp=visit_frame.timestamp,
            image=img,
            registration=reg_result,
            detections=processed_detections,
            metadata={"is_real_image": True},
        )
        report = self.engine.process_visit(visit_input)
        latencies["temporal_engine_ms"] = (time.perf_counter() - t_temp_start) * 1000.0

        self.processed_visits_count += 1
        total_time_ms = (time.perf_counter() - t0) * 1000.0

        return RealVisitProcessingResult(
            visit_id=visit_frame.visit_id,
            session_name=visit_frame.session_name,
            timestamp=visit_frame.timestamp,
            registration=reg_result,
            detection_batch=detection_batch,
            temporal_report=report,
            latency_breakdown_ms=latencies,
            total_latency_ms=total_time_ms,
        )

    def process_dataset(
        self,
        dataset: RealWorkbenchDataset,
        confidence_threshold: float = 0.35,
    ) -> RealPipelineDatasetReport:
        """Processes an entire multi-visit real-image dataset."""
        results: List[RealVisitProcessingResult] = []
        for visit_frame in dataset.visits:
            res = self.process_visit(visit_frame, confidence_threshold=confidence_threshold)
            results.append(res)
            # Free RAM
            visit_frame.release_cache()

        entities = self.store.list_entities()
        state_counts: Dict[str, int] = {}
        for ent in entities:
            st = ent.current_state.value
            state_counts[st] = state_counts.get(st, 0) + 1

        mean_lat = float(np.mean([r.total_latency_ms for r in results])) if results else 0.0

        return RealPipelineDatasetReport(
            dataset_name=dataset.dataset_name,
            total_visits_processed=len(results),
            visit_results=results,
            total_active_entities=len(entities),
            entity_states_summary=state_counts,
            mean_pipeline_latency_ms=mean_lat,
            cpu_profile={
                "mean_total_latency_ms": mean_lat,
                "detector_info": self.detector.model_info(),
            },
        )
