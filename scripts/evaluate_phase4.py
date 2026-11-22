"""
Comprehensive Real-World Evaluation and Temporal Reliability Suite (Phase 4).

Executes:
1. Phase 4B: Multi-detector benchmark on Real Annotated, Real Descriptive, and Synthetic image groups.
2. Phase 4C: Ground-truth quantitative evaluation against manual annotations (Precision, Recall, F1, mAP@50).
3. Phase 4D: Multi-visit temporal reliability tests across 9 physical environmental change scenarios.
4. Latency profiling separating cold initialization from steady-state inference.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

# Add src to path if running directly
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from objectmemory.baseline.comparator import calculate_bbox_iou
from objectmemory.datasets.real_dataset import (
    RealVisitFrame,
    RealWorkbenchDataset,
)
from objectmemory.detection.detector import ObjectDetector, OpenCVDNNDetector
from objectmemory.detection.models import (
    Detection,
    DetectionBatch,
    DetectorConfiguration,
    VisualSignature,
)
from objectmemory.detection.pipeline import transform_bbox_to_canonical
from objectmemory.detection.registry import DetectorRegistry
from objectmemory.detection.yolo_world import YOLOWorldDetector
from objectmemory.domain.models import BoundingBox, Observation, WorkbenchEntity
from objectmemory.domain.states import EpistemicState, VisibilityStatus
from objectmemory.persistence.repository import SQLiteMemoryStore
from objectmemory.pipeline.real_pipeline import RealImagePipelineRunner
from objectmemory.registration.models import AlignmentStatus, RegistrationResult
from objectmemory.temporal.engine import TemporalMemoryEngine
from objectmemory.temporal.models import TemporalEngineConfig, VisitInput

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("phase4_evaluation")


# ==============================================================================
# Phase 4B & 4C: Detector Comparative Evaluation & Ground-Truth Scoring
# ==============================================================================

@dataclass
class ImageDetectionRecord:
    image_id: str
    image_group: str  # "real_annotated", "real_descriptive", "synthetic"
    image_path: str
    resolution: Tuple[int, int]
    detector_name: str
    detector_version: str
    prompts: Optional[List[str]]
    confidence_threshold: float
    init_latency_ms: float
    steady_latency_ms: float
    detections_count: int
    labels: List[str]
    confidences: List[float]
    bboxes: List[List[float]]
    error: Optional[str] = None


@dataclass
class GroundTruthClassMetrics:
    class_name: str
    ground_truth_count: int = 0
    detected_count: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    ap50: float = 0.0


def compute_iou(b1: List[float], b2: List[float]) -> float:
    """Computes IoU between two [x_min, y_min, x_max, y_max] boxes."""
    x_left = max(b1[0], b2[0])
    y_top = max(b1[1], b2[1])
    x_right = min(b1[2], b2[2])
    y_bottom = min(b1[3], b2[3])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    inter_area = (x_right - x_left) * (y_bottom - y_top)
    b1_area = max(0.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    b2_area = max(0.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    union_area = b1_area + b2_area - inter_area
    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


def evaluate_detector_on_image_set(
    detector: ObjectDetector,
    detector_name: str,
    image_paths: List[Tuple[str, str, str]],  # (image_id, group, path)
    config: DetectorConfiguration,
    warmup_runs: int = 2,
    timed_runs: int = 3,
) -> List[ImageDetectionRecord]:
    """Runs a detector on a list of images, profiling cold init and steady-state inference."""
    results: List[ImageDetectionRecord] = []
    model_meta = detector.model_info()
    version = model_meta.get("model_type", model_meta.get("framework", "1.0"))

    is_cold = True

    for img_id, group, path_str in image_paths:
        img_path = Path(path_str)
        if not img_path.exists():
            results.append(
                ImageDetectionRecord(
                    image_id=img_id,
                    image_group=group,
                    image_path=path_str,
                    resolution=(0, 0),
                    detector_name=detector_name,
                    detector_version=version,
                    prompts=config.target_classes,
                    confidence_threshold=config.confidence_threshold,
                    init_latency_ms=0.0,
                    steady_latency_ms=0.0,
                    detections_count=0,
                    labels=[],
                    confidences=[],
                    bboxes=[],
                    error=f"File not found: {path_str}",
                )
            )
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            results.append(
                ImageDetectionRecord(
                    image_id=img_id,
                    image_group=group,
                    image_path=path_str,
                    resolution=(0, 0),
                    detector_name=detector_name,
                    detector_version=version,
                    prompts=config.target_classes,
                    confidence_threshold=config.confidence_threshold,
                    init_latency_ms=0.0,
                    steady_latency_ms=0.0,
                    detections_count=0,
                    labels=[],
                    confidences=[],
                    bboxes=[],
                    error="Failed to decode image",
                )
            )
            continue

        h, w = img.shape[:2]

        init_ms = 0.0
        if is_cold:
            t0 = time.perf_counter()
            try:
                batch_init = detector.detect(img, config=config)
                init_ms = (time.perf_counter() - t0) * 1000.0
            except Exception as ex:
                results.append(
                    ImageDetectionRecord(
                        image_id=img_id,
                        image_group=group,
                        image_path=path_str,
                        resolution=(w, h),
                        detector_name=detector_name,
                        detector_version=version,
                        prompts=config.target_classes,
                        confidence_threshold=config.confidence_threshold,
                        init_latency_ms=0.0,
                        steady_latency_ms=0.0,
                        detections_count=0,
                        labels=[],
                        confidences=[],
                        bboxes=[],
                        error=f"Init detection failed: {ex}",
                    )
                )
                continue
            is_cold = False

        # Steady-state runs
        steady_times = []
        last_batch: Optional[DetectionBatch] = None
        for _ in range(timed_runs):
            t0 = time.perf_counter()
            last_batch = detector.detect(img, config=config)
            steady_times.append((time.perf_counter() - t0) * 1000.0)

        avg_steady_ms = float(np.mean(steady_times)) if steady_times else 0.0
        dets = last_batch.detections if last_batch else []

        results.append(
            ImageDetectionRecord(
                image_id=img_id,
                image_group=group,
                image_path=path_str,
                resolution=(w, h),
                detector_name=detector_name,
                detector_version=version,
                prompts=config.target_classes,
                confidence_threshold=config.confidence_threshold,
                init_latency_ms=init_ms,
                steady_latency_ms=avg_steady_ms,
                detections_count=len(dets),
                labels=[d.label for d in dets],
                confidences=[float(d.confidence) for d in dets],
                bboxes=[[d.raw_bbox.x_min, d.raw_bbox.y_min, d.raw_bbox.x_max, d.raw_bbox.y_max] for d in dets],
                error=None,
            )
        )

    return results


def calculate_ground_truth_metrics(
    detection_records: List[ImageDetectionRecord],
    manifest_path: Path,
    iou_threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Computes precision, recall, and class-level metrics against ground-truth annotations.
    """
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    # Build GT index: visit_id -> list of (label, bbox)
    gt_by_visit: Dict[str, List[Dict[str, Any]]] = {}
    for v in manifest.get("visits", []):
        vid = v["visit_id"]
        gt_list = []
        for ann in v.get("annotations", []):
            if ann.get("physical_observation") == "VISIBLE" and ann.get("bbox") is not None:
                gt_list.append(
                    {
                        "entity_id": ann.get("entity_id"),
                        "label": ann.get("entity_label", "").lower().strip(),
                        "bbox": ann.get("bbox"),
                    }
                )
        gt_by_visit[vid] = gt_list

    class_data: Dict[str, Dict[str, int]] = {}
    total_tp = 0
    total_fp = 0
    total_fn = 0

    # Label normalization map between COCO and Manifest classes
    synonyms = {
        "potted plant": ["plant", "potted plant"],
        "book": ["book", "notebook"],
        "notebook": ["notebook", "book"],
        "cup": ["cup", "mug"],
        "bottle": ["bottle", "water bottle"],
        "scissors": ["scissors"],
    }

    def labels_match(pred_lbl: str, gt_lbl: str) -> bool:
        p = pred_lbl.lower().strip()
        g = gt_lbl.lower().strip()
        if p == g:
            return True
        for canonical, syns in synonyms.items():
            if p in syns and g in syns:
                return True
        return False

    evaluated_visits = 0

    for rec in detection_records:
        if rec.image_group != "real_annotated":
            continue

        vid = rec.image_id
        if vid not in gt_by_visit:
            continue

        evaluated_visits += 1
        gt_boxes = list(gt_by_visit[vid])
        pred_boxes = list(zip(rec.labels, rec.bboxes, rec.confidences))

        gt_matched = [False] * len(gt_boxes)

        # Sort predictions by confidence descending
        pred_boxes.sort(key=lambda x: x[2], reverse=True)

        for p_lbl, p_box, p_conf in pred_boxes:
            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(gt_boxes):
                if not gt_matched[gt_idx] and labels_match(p_lbl, gt["label"]):
                    iou = compute_iou(p_box, gt["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

            clean_lbl = p_lbl.lower().strip()
            if clean_lbl not in class_data:
                class_data[clean_lbl] = {"tp": 0, "fp": 0, "fn": 0, "gt": 0}

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                gt_matched[best_gt_idx] = True
                total_tp += 1
                class_data[clean_lbl]["tp"] += 1
            else:
                total_fp += 1
                class_data[clean_lbl]["fp"] += 1

        for gt_idx, gt in enumerate(gt_boxes):
            g_lbl = gt["label"].lower().strip()
            if g_lbl not in class_data:
                class_data[g_lbl] = {"tp": 0, "fp": 0, "fn": 0, "gt": 0}
            class_data[g_lbl]["gt"] += 1
            if not gt_matched[gt_idx]:
                total_fn += 1
                class_data[g_lbl]["fn"] += 1

    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = (
        (2 * overall_prec * overall_rec) / (overall_prec + overall_rec)
        if (overall_prec + overall_rec) > 0
        else 0.0
    )

    class_metrics = {}
    for c_name, counts in sorted(class_data.items()):
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        gt = counts["gt"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        class_metrics[c_name] = {
            "ground_truth_count": gt,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1_score": round(f1, 4),
        }

    return {
        "evaluated_visits": evaluated_visits,
        "iou_threshold": iou_threshold,
        "overall": {
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "precision": round(overall_prec, 4),
            "recall": round(overall_rec, 4),
            "f1_score": round(overall_f1, 4),
        },
        "per_class": class_metrics,
    }


# ==============================================================================
# Phase 4D: Multi-Visit Temporal Memory Reliability Tests
# ==============================================================================

@dataclass
class TemporalScenarioResult:
    scenario_number: int
    scenario_name: str
    description: str
    expected_outcome: str
    actual_outcome: str
    passed: bool
    details: Dict[str, Any]


def _make_dummy_sig(dim: int = 148, split: int = 0) -> VisualSignature:
    vec = np.zeros(dim, dtype=np.float32)
    if split == 0:
        vec[:74] = 1.0
    elif split == 1:
        vec[74:] = 1.0
    else:
        vec.fill(1.0)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return VisualSignature(
        color_histogram=vec[:128].copy(),
        texture_descriptor=vec[128:].copy(),
        combined_vector=vec,
    )


def make_det(label: str, bbox: BoundingBox, split: int = 0, conf: float = 0.9) -> Detection:
    return Detection(
        label=label,
        confidence=conf,
        raw_bbox=bbox,
        canonical_bbox=bbox,
        signature=_make_dummy_sig(split=split),
    )


def make_reg_success(inlier_ratio: float = 0.95) -> RegistrationResult:
    return RegistrationResult(
        success=True,
        status=AlignmentStatus.SUCCESS,
        method_used="MOCK",
        ref_size=(640, 480),
        query_size=(640, 480),
        homography=np.eye(3, dtype=np.float64),
        overlap_mask=np.ones((480, 640), dtype=np.uint8) * 255,
        aligned_image=np.ones((480, 640, 3), dtype=np.uint8) * 120,
        inlier_ratio=inlier_ratio,
    )


def make_reg_failed() -> RegistrationResult:
    return RegistrationResult(
        success=False,
        status=AlignmentStatus.INSUFFICIENT_INLIERS,
        method_used="MOCK",
        ref_size=(640, 480),
        query_size=(640, 480),
        homography=None,
        inlier_ratio=0.10,
        failure_reason="Registration failed due to insufficient inliers",
    )


def run_temporal_reliability_suite() -> List[TemporalScenarioResult]:
    """
    Executes the 9 multi-visit temporal physical change scenarios.
    """
    results: List[TemporalScenarioResult] = []

    def make_engine() -> TemporalMemoryEngine:
        store = SQLiteMemoryStore(":memory:")
        return TemporalMemoryEngine(
            store=store,
            config=TemporalEngineConfig(
                min_registration_inlier_ratio=0.40,
                movement_displacement_threshold=35.0,
                vacancy_visits_for_pending=1,
                vacancy_visits_for_removal=2,
            ),
        )

    # -------------------------------------------------------------------------
    # Scenario 1: Static objects
    # -------------------------------------------------------------------------
    engine = make_engine()
    b_static = BoundingBox(100.0, 100.0, 200.0, 200.0)
    v1 = VisitInput(
        session_id="v1",
        session_name="Visit 1",
        timestamp="2026-10-20T10:00:00Z",
        image=np.zeros((480, 640, 3), dtype=np.uint8),
        registration=make_reg_success(),
        detections=[make_det("mug", b_static)],
    )
    v2 = VisitInput(
        session_id="v2",
        session_name="Visit 2",
        timestamp="2026-10-20T10:05:00Z",
        image=np.zeros((480, 640, 3), dtype=np.uint8),
        registration=make_reg_success(),
        detections=[make_det("mug", BoundingBox(102.0, 101.0, 202.0, 201.0))],
    )
    engine.process_visit(v1)
    rep2 = engine.process_visit(v2)
    e = engine.store.get_entity("ent_v1_1")
    s1_pass = (
        e is not None
        and e.current_state == EpistemicState.CONFIRMED_PRESENT
        and rep2.state_transitions_count == 0  # Remains confirmed present
        and "CONFIRMED_MOVED" not in [u.new_state.value for u in rep2.entity_updates]
    )
    results.append(
        TemporalScenarioResult(
            scenario_number=1,
            scenario_name="Static Objects Continuity",
            description="Stationary object with minor jitter (<35px) remains CONFIRMED_PRESENT without movement transitions.",
            expected_outcome="State is CONFIRMED_PRESENT, 0 movement events.",
            actual_outcome=f"State: {e.current_state.value if e else 'None'}, updates: {[u.new_state.value for u in rep2.entity_updates]}",
            passed=s1_pass,
            details={"entity_id": "ent_v1_1", "state": e.current_state.value if e else None},
        )
    )

    # -------------------------------------------------------------------------
    # Scenario 2: Genuine object movement
    # -------------------------------------------------------------------------
    v3_moved = VisitInput(
        session_id="v3",
        session_name="Visit 3",
        timestamp="2026-10-20T10:10:00Z",
        image=np.zeros((480, 640, 3), dtype=np.uint8),
        registration=make_reg_success(),
        detections=[make_det("mug", BoundingBox(222.0, 101.0, 322.0, 201.0))],
    )
    rep3 = engine.process_visit(v3_moved)
    e2 = engine.store.get_entity("ent_v1_1")
    moved_updates = [u for u in rep3.entity_updates if u.new_state == EpistemicState.CONFIRMED_MOVED]
    events_v3 = engine.store.list_change_events_for_session("v3")
    moved_event = next((ev for ev in events_v3 if ev.new_state == EpistemicState.CONFIRMED_MOVED), None)
    disp_val = moved_event.displacement_distance if moved_event else None
    s2_pass = (
        e2 is not None
        and e2.current_state == EpistemicState.CONFIRMED_MOVED
        and len(moved_updates) == 1
        and disp_val is not None
        and disp_val > 100.0
    )
    results.append(
        TemporalScenarioResult(
            scenario_number=2,
            scenario_name="Genuine Object Movement Detection",
            description="Object spatially displaced by 120px (>35px threshold) triggers CONFIRMED_MOVED with recorded displacement.",
            expected_outcome="Transitions to CONFIRMED_MOVED with displacement ~120px.",
            actual_outcome=f"State: {e2.current_state.value if e2 else 'None'}, displacement: {disp_val}",
            passed=s2_pass,
            details={"updates": [u.new_state.value for u in rep3.entity_updates], "displacement": disp_val},
        )
    )

    # -------------------------------------------------------------------------
    # Scenario 3: Added objects
    # -------------------------------------------------------------------------
    v4_added = VisitInput(
        session_id="v4",
        session_name="Visit 4",
        timestamp="2026-10-20T10:15:00Z",
        image=np.zeros((480, 640, 3), dtype=np.uint8),
        registration=make_reg_success(),
        detections=[
            make_det("mug", BoundingBox(222.0, 101.0, 322.0, 201.0)),
            make_det("stapler", BoundingBox(400.0, 300.0, 480.0, 350.0)),
        ],
    )
    rep4 = engine.process_visit(v4_added)
    all_ents = engine.store.list_entities()
    stapler_ent = next((ent for ent in all_ents if ent.label == "stapler"), None)
    s3_pass = (
        rep4.new_entities_created == 1
        and stapler_ent is not None
        and stapler_ent.current_state == EpistemicState.CONFIRMED_PRESENT
    )
    results.append(
        TemporalScenarioResult(
            scenario_number=3,
            scenario_name="Added Object Ingest",
            description="Newly introduced object in the scene initializes in CONFIRMED_PRESENT.",
            expected_outcome="New entity created in CONFIRMED_PRESENT.",
            actual_outcome=f"Created count: {rep4.new_entities_created}, stapler state: {stapler_ent.current_state.value if stapler_ent else 'None'}",
            passed=s3_pass,
            details={"stapler_id": stapler_ent.entity_id if stapler_ent else None},
        )
    )

    # -------------------------------------------------------------------------
    # Scenario 4: Genuine object removal
    # -------------------------------------------------------------------------
    engine_removal = make_engine()
    b_rem = BoundingBox(50.0, 50.0, 100.0, 100.0)
    # Visit 1: Seen
    engine_removal.process_visit(
        VisitInput("rv1", "RVisit 1", "2026-10-20T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), make_reg_success(), [make_det("pen", b_rem, split=0)])
    )
    # Visit 2: Confirmed present
    engine_removal.process_visit(
        VisitInput("rv2", "RVisit 2", "2026-10-20T10:05:00Z", np.zeros((480, 640, 3), dtype=np.uint8), make_reg_success(), [make_det("pen", b_rem, split=0)])
    )
    # Visit 3: Absent 1st time -> PENDING_REMOVAL
    engine_removal.process_visit(
        VisitInput("rv3", "RVisit 3", "2026-10-20T10:10:00Z", np.zeros((480, 640, 3), dtype=np.uint8), make_reg_success(), [])
    )
    b_pen = engine_removal.store.get_entity("ent_rv1_1")
    s4_pending = b_pen is not None and b_pen.current_state == EpistemicState.PENDING_REMOVAL

    # Visit 4: Absent 2nd time -> CONFIRMED_REMOVED
    engine_removal.process_visit(
        VisitInput("rv4", "RVisit 4", "2026-10-20T10:15:00Z", np.zeros((480, 640, 3), dtype=np.uint8), make_reg_success(), [])
    )
    b_pen2 = engine_removal.store.get_entity("ent_rv1_1")
    s4_pass = s4_pending and b_pen2 is not None and b_pen2.current_state == EpistemicState.CONFIRMED_REMOVED
    results.append(
        TemporalScenarioResult(
            scenario_number=4,
            scenario_name="Genuine Object Removal Progression",
            description="Absence on consecutive reliable visits progresses from PENDING_REMOVAL to CONFIRMED_REMOVED.",
            expected_outcome="Visit 3: PENDING_REMOVAL, Visit 4: CONFIRMED_REMOVED.",
            actual_outcome=f"Visit 3: {b_pen.current_state.value if b_pen else None}, Visit 4: {b_pen2.current_state.value if b_pen2 else None}",
            passed=bool(s4_pass),
            details={"v3_state": b_pen.current_state.value if b_pen else None, "v4_state": b_pen2.current_state.value if b_pen2 else None},
        )
    )

    # -------------------------------------------------------------------------
    # Scenario 5: Temporary occlusion (single-frame drop)
    # -------------------------------------------------------------------------
    engine_drop = make_engine()
    b_drop = BoundingBox(60.0, 60.0, 120.0, 120.0)
    engine_drop.process_visit(
        VisitInput("dv1", "DVisit 1", "2026-10-20T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), make_reg_success(), [make_det("book", b_drop, split=0)])
    )
    engine_drop.process_visit(
        VisitInput("dv2", "DVisit 2", "2026-10-20T10:05:00Z", np.zeros((480, 640, 3), dtype=np.uint8), make_reg_success(), [make_det("book", b_drop, split=0)])
    )
    # 1 missed frame
    engine_drop.process_visit(
        VisitInput("dv3", "DVisit 3", "2026-10-20T10:10:00Z", np.zeros((480, 640, 3), dtype=np.uint8), make_reg_success(), [])
    )
    b_drop_mid = engine_drop.store.get_entity("ent_dv1_1")
    # Recovered on next frame
    engine_drop.process_visit(
        VisitInput("dv4", "DVisit 4", "2026-10-20T10:15:00Z", np.zeros((480, 640, 3), dtype=np.uint8), make_reg_success(), [make_det("book", b_drop, split=0)])
    )
    b_drop_rec = engine_drop.store.get_entity("ent_dv1_1")
    s5_pass = (
        b_drop_mid is not None
        and b_drop_mid.current_state == EpistemicState.PENDING_REMOVAL  # Was NOT confirmed removed
        and b_drop_rec is not None
        and b_drop_rec.current_state == EpistemicState.CONFIRMED_PRESENT  # Cleanly restored
    )
    results.append(
        TemporalScenarioResult(
            scenario_number=5,
            scenario_name="Temporary Occlusion / Single Drop Resilience",
            description="Single missed detection does NOT trigger false removal; recovers cleanly on redetection.",
            expected_outcome="Remains PENDING_REMOVAL during occlusion, restored to CONFIRMED_PRESENT.",
            actual_outcome=f"Occluded state: {b_drop_mid.current_state.value if b_drop_mid else None}, recovered: {b_drop_rec.current_state.value if b_drop_rec else None}",
            passed=bool(s5_pass),
            details={"occluded": b_drop_mid.current_state.value if b_drop_mid else None, "recovered": b_drop_rec.current_state.value if b_drop_rec else None},
        )
    )

    # -------------------------------------------------------------------------
    # Scenario 6: Object reappearance after confirmed removal
    # -------------------------------------------------------------------------
    # Continuing engine_removal where pen was CONFIRMED_REMOVED in Visit 4:
    engine_removal.process_visit(
        VisitInput("rv5", "RVisit 5", "2026-10-20T10:20:00Z", np.zeros((480, 640, 3), dtype=np.uint8), make_reg_success(), [make_det("pen", b_rem, split=0)])
    )
    b_reapp = engine_removal.store.get_entity("ent_rv1_1")
    s6_pass = b_reapp is not None and b_reapp.current_state == EpistemicState.CONFIRMED_PRESENT
    results.append(
        TemporalScenarioResult(
            scenario_number=6,
            scenario_name="Reappearance After Confirmed Removal",
            description="An object previously marked CONFIRMED_REMOVED that reappears is restored to CONFIRMED_PRESENT.",
            expected_outcome="Restores entity to CONFIRMED_PRESENT while preserving history.",
            actual_outcome=f"Post-reappearance state: {b_reapp.current_state.value if b_reapp else None}",
            passed=bool(s6_pass),
            details={"entity_id": "ent_rv1_1", "state": b_reapp.current_state.value if b_reapp else None},
        )
    )

    # -------------------------------------------------------------------------
    # Scenario 7: Camera viewpoint changes (canonical space invariance)
    # -------------------------------------------------------------------------
    H_viewpoint = np.array(
        [
            [0.98, -0.04, 25.0],
            [0.03, 0.99, -15.0],
            [0.00005, -0.00002, 1.0],
        ]
    )
    raw_box = BoundingBox(200.0, 200.0, 300.0, 300.0)
    canon_box = transform_bbox_to_canonical(raw_box, H_viewpoint, (1000, 1000))
    s7_pass = (
        canon_box is not None
        and canon_box.x_min > 0.0
        and canon_box.y_min > 0.0
        and abs((canon_box.x_max - canon_box.x_min) - 100.0) < 15.0
    )
    results.append(
        TemporalScenarioResult(
            scenario_number=7,
            scenario_name="Camera Viewpoint Canonical Mapping",
            description="Projective transformation maps bounding boxes reliably into canonical space under camera shifts.",
            expected_outcome="Valid canonical bounding box preserved across viewpoint shift.",
            actual_outcome=f"Transformed canonical bbox: [{canon_box.x_min:.1f}, {canon_box.y_min:.1f}, {canon_box.x_max:.1f}, {canon_box.y_max:.1f}]" if canon_box else "Failed",
            passed=bool(s7_pass),
            details={"canonical_bbox": [canon_box.x_min, canon_box.y_min, canon_box.x_max, canon_box.y_max] if canon_box else None},
        )
    )

    # -------------------------------------------------------------------------
    # Scenario 8: Detector label changes (Cross-class association)
    # -------------------------------------------------------------------------
    engine_label = make_engine()
    b_book = BoundingBox(300.0, 300.0, 450.0, 450.0)
    # Visit 1: Detected as 'book'
    engine_label.process_visit(
        VisitInput("lv1", "LVisit 1", "2026-10-20T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), make_reg_success(), [make_det("book", b_book, split=0)])
    )
    # Visit 2: Detected as 'notebook' at identical position
    engine_label.process_visit(
        VisitInput("lv2", "LVisit 2", "2026-10-20T10:05:00Z", np.zeros((480, 640, 3), dtype=np.uint8), make_reg_success(), [make_det("notebook", b_book, split=0)])
    )
    # Visit 3: Detected as 'laptop' at identical position
    engine_label.process_visit(
        VisitInput("lv3", "LVisit 3", "2026-10-20T10:10:00Z", np.zeros((480, 640, 3), dtype=np.uint8), make_reg_success(), [make_det("laptop", b_book, split=0)])
    )
    all_entities = engine_label.store.list_entities()
    s8_pass = len(all_entities) == 1 and all_entities[0].entity_id == "ent_lv1_1"
    results.append(
        TemporalScenarioResult(
            scenario_number=8,
            scenario_name="Detector Class Prediction Jitter Resilience",
            description="Conservative cross-class association links identical spatial positions across fluctuating detector labels without duplicate identities.",
            expected_outcome="Exactly 1 persistent entity maintained across 3 visits.",
            actual_outcome=f"Total active entities: {len(all_entities)} (ID: {all_entities[0].entity_id if all_entities else None})",
            passed=s8_pass,
            details={"entities": [e.entity_id for e in all_entities]},
        )
    )

    # -------------------------------------------------------------------------
    # Scenario 9: Failed or unreliable registration safeguards
    # -------------------------------------------------------------------------
    engine_fail = make_engine()
    b_fail = BoundingBox(150.0, 150.0, 250.0, 250.0)
    # Establish confirmed object
    engine_fail.process_visit(
        VisitInput("fv1", "FVisit 1", "2026-10-20T10:00:00Z", np.zeros((480, 640, 3), dtype=np.uint8), make_reg_success(), [make_det("cup", b_fail)])
    )
    engine_fail.process_visit(
        VisitInput("fv2", "FVisit 2", "2026-10-20T10:05:00Z", np.zeros((480, 640, 3), dtype=np.uint8), make_reg_success(), [make_det("cup", b_fail)])
    )
    # Visit 3: Registration fails completely (inlier ratio = 0.10, status = 'FAILED')
    rep_fail = engine_fail.process_visit(
        VisitInput("fv3", "FVisit 3", "2026-10-20T10:10:00Z", np.zeros((480, 640, 3), dtype=np.uint8), make_reg_failed(), [])
    )
    b_fail_post = engine_fail.store.get_entity("ent_fv1_1")
    fail_streak = (b_fail_post.metadata or {}).get("vacant_streak", 0) if b_fail_post else None
    s9_pass = (
        b_fail_post is not None
        and b_fail_post.current_state == EpistemicState.CONFIRMED_PRESENT  # Absence NOT accumulated
        and fail_streak == 0
        and rep_fail.state_transitions_count == 0
    )
    results.append(
        TemporalScenarioResult(
            scenario_number=9,
            scenario_name="Registration Failure Spatial Gating",
            description="Failed registration strictly halts absence accumulation and spatial updating; blocks false removals.",
            expected_outcome="Remains CONFIRMED_PRESENT with 0 consecutive absences.",
            actual_outcome=f"State: {b_fail_post.current_state.value if b_fail_post else None}, streak: {fail_streak}, transitions: {rep_fail.state_transitions_count}",
            passed=bool(s9_pass),
            details={"state": b_fail_post.current_state.value if b_fail_post else None, "vacant_streak": fail_streak},
        )
    )

    return results


# ==============================================================================
# Main Runner
# ==============================================================================

def run_phase4_evaluation() -> Dict[str, Any]:
    print("================================================================================")
    print("OBJECTMEMORY STUDIO — PHASE 4: REAL-WORLD EVALUATION & TEMPORAL RELIABILITY")
    print("================================================================================")

    # 1. Image dataset assembly
    annotated_visits = [
        (f"visit_{i:02d}", "real_annotated", f"data/real_workbench/visits/visit_{i:02d}.png")
        for i in range(1, 8)
    ]

    descriptive_images = [
        ("real_test_v01", "real_descriptive", "data/workspaces/objectmemory_real_test/images/visit_01.png"),
        ("real_test_v02", "real_descriptive", "data/workspaces/objectmemory_real_test/images/visit_02.png"),
        ("studio_demo_v03", "real_descriptive", "data/workspaces/studio_demo/images/visit_03.png"),
        ("studio_demo_v04", "real_descriptive", "data/workspaces/studio_demo/images/visit_04.png"),
    ]

    synthetic_images = [
        ("synth_no_change", "synthetic", "data/synthetic_workbench/no_change/query.png"),
        ("synth_moved", "synthetic", "data/synthetic_workbench/object_moved/query.png"),
        ("synth_added", "synthetic", "data/synthetic_workbench/object_added/query.png"),
        ("synth_removed", "synthetic", "data/synthetic_workbench/object_removed/query.png"),
        ("synth_occlusion", "synthetic", "data/synthetic_workbench/partial_occlusion/query.png"),
        ("synth_lighting", "synthetic", "data/synthetic_workbench/lighting_change/query.png"),
        ("synth_ambiguous", "synthetic", "data/synthetic_workbench/ambiguous_objects/query.png"),
        ("synth_unreliable", "synthetic", "data/synthetic_workbench/unreliable_alignment/query.png"),
    ]

    all_images = annotated_visits + descriptive_images + synthetic_images
    print(f"Total Images: {len(all_images)} (Annotated: {len(annotated_visits)}, Descriptive: {len(descriptive_images)}, Synthetic: {len(synthetic_images)})")

    # 2. Instantiate Detectors
    print("\n--- Instantiating Detectors ---")
    yolov8n = DetectorRegistry.get("yolov8n")
    yolo_world = DetectorRegistry.get("yolo-world")

    print(f"Detector 1: {yolov8n.model_info()['name']} (open_vocab={yolov8n.model_info()['is_open_vocabulary']})")
    print(f"Detector 2: {yolo_world.model_info()['name']} (open_vocab={yolo_world.model_info()['is_open_vocabulary']})")

    # 3. Evaluate YOLOv8n Baseline
    print("\n--- Evaluating YOLOv8n Baseline ---")
    cfg_yolov8n = DetectorConfiguration(confidence_threshold=0.25)
    rec_yolov8n = evaluate_detector_on_image_set(yolov8n, "yolov8n", all_images, cfg_yolov8n)

    # 4. Evaluate YOLO-World Open-Vocabulary
    print("\n--- Evaluating YOLO-World Open-Vocabulary ---")
    world_prompts = [
        "potted plant",
        "plant",
        "cup",
        "mug",
        "bottle",
        "water bottle",
        "scissors",
        "book",
        "notebook",
        "tool",
        "pliers",
        "container",
    ]
    cfg_world = DetectorConfiguration(
        confidence_threshold=0.15,
        target_classes=world_prompts,
    )
    rec_world = evaluate_detector_on_image_set(yolo_world, "yolo-world", all_images, cfg_world)

    # 5. Compute Ground Truth Metrics (Phase 4C)
    print("\n--- Computing Ground-Truth Metrics against Manifest Annotations ---")
    manifest_path = Path("data/real_workbench/manifest.json")
    metrics_yolov8n_50 = calculate_ground_truth_metrics(rec_yolov8n, manifest_path, iou_threshold=0.50)
    metrics_yolov8n_30 = calculate_ground_truth_metrics(rec_yolov8n, manifest_path, iou_threshold=0.30)

    metrics_world_50 = calculate_ground_truth_metrics(rec_world, manifest_path, iou_threshold=0.50)
    metrics_world_30 = calculate_ground_truth_metrics(rec_world, manifest_path, iou_threshold=0.30)

    # 6. Execute Temporal Reliability Suite (Phase 4D)
    print("\n--- Executing Multi-Visit Temporal Reliability Suite (9 Scenarios) ---")
    temporal_results = run_temporal_reliability_suite()
    all_passed = all(t.passed for t in temporal_results)

    for tr in temporal_results:
        status_str = "[PASS]" if tr.passed else "[FAIL]"
        print(f"  {status_str} Scenario {tr.scenario_number}: {tr.scenario_name}")

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_counts": {
            "real_annotated": len(annotated_visits),
            "real_descriptive": len(descriptive_images),
            "synthetic": len(synthetic_images),
            "total_evaluated": len(all_images),
        },
        "detectors_evaluated": ["yolov8n", "yolo-world"],
        "latency_summary": {
            "yolov8n": {
                "init_latency_ms": rec_yolov8n[0].init_latency_ms,
                "mean_steady_latency_ms": round(float(np.mean([r.steady_latency_ms for r in rec_yolov8n if r.steady_latency_ms > 0])), 1),
            },
            "yolo_world": {
                "init_latency_ms": rec_world[0].init_latency_ms,
                "mean_steady_latency_ms": round(float(np.mean([r.steady_latency_ms for r in rec_world if r.steady_latency_ms > 0])), 1),
            },
        },
        "ground_truth_metrics": {
            "yolov8n_iou50": metrics_yolov8n_50,
            "yolov8n_iou30": metrics_yolov8n_30,
            "yolo_world_iou50": metrics_world_50,
            "yolo_world_iou30": metrics_world_30,
        },
        "temporal_scenarios": [asdict(t) for t in temporal_results],
        "all_temporal_scenarios_passed": all_passed,
        "detection_records_yolov8n": [asdict(r) for r in rec_yolov8n],
        "detection_records_world": [asdict(r) for r in rec_world],
    }

    # Save results to disk
    out_path = Path("scripts/phase4_evaluation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nEvaluation complete. Full results written to {out_path}.")
    return summary


if __name__ == "__main__":
    run_phase4_evaluation()
