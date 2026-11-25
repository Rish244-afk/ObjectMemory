#!/usr/bin/env python3
"""
Physical Workbench Benchmark Evaluation Suite (Phase 14C).

Evaluates ObjectMemory on real physical workbench imagery across three decoupled tracks:
- Track A: Natural-Image Object Detection (YOLOv8n and YOLO-World with letterbox preprocessing)
- Track B: Oracle Temporal Reasoning (Ground-truth bounding boxes directly fed to state machine)
- Track C: End-to-End Physical Evaluation (Actual detector predictions -> Registration -> Association -> Persistence)

Scientific Boundaries:
1. Tracks are strictly separated: Track B oracle results are NEVER combined with Track C predictions.
2. If genuine photographs are not yet captured, status is reported as AWAITING_PHYSICAL_DATA.
3. No metrics, detections, or physical observations are fabricated.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import cv2
import numpy as np

# Ensure src is on python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from objectmemory.baseline.comparator import calculate_bbox_iou, calculate_centroid_distance, solve_optimal_assignment
from objectmemory.datasets.physical_workbench import (
    AbsenceReason,
    DatasetCaptureStatus,
    PhysicalVisibility,
    PhysicalWorkbenchAnnotation,
    PhysicalWorkbenchDataset,
    PhysicalWorkbenchSequence,
    PhysicalWorkbenchVisit,
    detect_aruco_corner_markers,
)
from objectmemory.detection.models import BoundingBox, Detection, DetectionBatch, DetectorConfiguration
from objectmemory.detection.registry import DetectorRegistry
from objectmemory.detection.signature import VisualSignatureExtractor
from objectmemory.domain.states import EpistemicState
from objectmemory.persistence.repository import SQLiteMemoryStore
from objectmemory.pipeline.real_pipeline import RealImagePipelineRunner
from objectmemory.registration.models import AlignmentStatus
from objectmemory.registration.registrar import PlanarRegistrar
from objectmemory.temporal.engine import TemporalMemoryEngine
from objectmemory.temporal.models import TemporalEngineConfig, VisitInput, VisitProcessingReport


def match_detections_to_annotations(
    detections: List[Detection],
    annotations: List[PhysicalWorkbenchAnnotation],
    iou_threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Performs greedy 1-to-1 bipartite matching between detector predictions
    and ground-truth annotations for a single image frame.
    """
    active_anns = [a for a in annotations if a.bbox is not None and a.visibility != PhysicalVisibility.ABSENT]
    matches: List[Tuple[int, int, float]] = []

    used_dets: Set[int] = set()
    used_anns: Set[int] = set()

    # Candidate pairs
    candidates = []
    for d_idx, det in enumerate(detections):
        for a_idx, ann in enumerate(active_anns):
            # Class match required
            if det.label == ann.class_label:
                iou = calculate_bbox_iou(det.raw_bbox, ann.bbox)
                if iou >= iou_threshold:
                    candidates.append((iou, d_idx, a_idx))

    # Sort descending by IoU
    candidates.sort(key=lambda x: x[0], reverse=True)
    for iou, d_idx, a_idx in candidates:
        if d_idx not in used_dets and a_idx not in used_anns:
            used_dets.add(d_idx)
            used_anns.add(a_idx)
            matches.append((d_idx, a_idx, iou))

    tp = len(matches)
    fp = len(detections) - tp
    fn = len(active_anns) - tp

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "total_ground_truth": len(active_anns),
        "total_detections": len(detections),
        "matches": matches,
    }


def evaluate_physical_track_a(
    dataset: PhysicalWorkbenchDataset,
    registry: Optional[DetectorRegistry] = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    TRACK A: Natural-Image Object Detection on physical photographs.
    Evaluates both YOLOv8n and YOLO-World with letterbox preprocessing.
    """
    if registry is None:
        registry = DetectorRegistry()

    models_to_test = [
        ("yolov8n", registry.get("yolov8n")),
        ("yolo-world", registry.get("yolo-world")),
    ]

    target_classes = sorted(list(dataset.get_all_annotated_classes() or dataset.CANONICAL_CLASSES))
    results: Dict[str, Any] = {}

    for model_name, detector in models_to_test:
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_gt = 0
        total_preds = 0
        latencies_ms: List[float] = []

        class_counts: Dict[str, Dict[str, int]] = {
            cls_name: {"tp": 0, "fp": 0, "fn": 0, "gt": 0}
            for cls_name in target_classes
        }

        # Warm-up pass if any image exists
        sample_img = None
        for seq in dataset.sequences.values():
            for v in seq.visits:
                if v.is_image_present():
                    sample_img = v.load_image()
                    break
            if sample_img is not None:
                break

        if sample_img is not None:
            if model_name == "yolo-world":
                cfg = DetectorConfiguration(confidence_threshold=conf_threshold, target_classes=target_classes)
                detector.detect(sample_img, config=cfg)
            else:
                detector.detect(sample_img, confidence_threshold=conf_threshold)

        # Timed evaluation across all available visits
        for seq in dataset.sequences.values():
            for v in seq.visits:
                if not v.is_image_present():
                    continue

                img = v.load_image()
                if img is None:
                    continue

                t0 = time.perf_counter()
                if model_name == "yolo-world":
                    cfg = DetectorConfiguration(confidence_threshold=conf_threshold, target_classes=target_classes)
                    batch = detector.detect(img, config=cfg)
                else:
                    batch = detector.detect(img, confidence_threshold=conf_threshold)
                lat_ms = (time.perf_counter() - t0) * 1000.0
                latencies_ms.append(lat_ms)

                # Filter detections to canonical/annotated target classes
                canonical_dets = [d for d in batch.detections if d.label in target_classes]
                total_preds += len(canonical_dets)

                frame_res = match_detections_to_annotations(
                    detections=canonical_dets,
                    annotations=v.annotations,
                    iou_threshold=iou_threshold,
                )

                total_tp += frame_res["true_positives"]
                total_fp += frame_res["false_positives"]
                total_fn += frame_res["false_negatives"]
                total_gt += frame_res["total_ground_truth"]

                # Per-class counts
                active_anns = [a for a in v.annotations if a.bbox is not None and a.visibility != PhysicalVisibility.ABSENT]
                for a in active_anns:
                    if a.class_label in class_counts:
                        class_counts[a.class_label]["gt"] += 1

                for d_idx, a_idx, _ in frame_res["matches"]:
                    matched_cls = canonical_dets[d_idx].label
                    if matched_cls in class_counts:
                        class_counts[matched_cls]["tp"] += 1

                # Matched indices
                matched_d_indices = {m[0] for m in frame_res["matches"]}
                matched_a_indices = {m[1] for m in frame_res["matches"]}

                for d_idx, det in enumerate(canonical_dets):
                    if d_idx not in matched_d_indices and det.label in class_counts:
                        class_counts[det.label]["fp"] += 1

                for a_idx, ann in enumerate(active_anns):
                    if a_idx not in matched_a_indices and ann.class_label in class_counts:
                        class_counts[ann.class_label]["fn"] += 1

        precision = (total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
        recall = (total_tp / total_gt) if total_gt > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        mean_lat = float(np.mean(latencies_ms)) if latencies_ms else 0.0

        # Per-class precision, recall, f1
        per_class_summary = {}
        for cls_name, counts in class_counts.items():
            c_tp = counts["tp"]
            c_fp = counts["fp"]
            c_fn = counts["fn"]
            c_gt = counts["gt"]
            c_p = (c_tp / (c_tp + c_fp)) if (c_tp + c_fp) > 0 else 0.0
            c_r = (c_tp / c_gt) if c_gt > 0 else 0.0
            c_f1 = (2 * c_p * c_r / (c_p + c_r)) if (c_p + c_r) > 0 else 0.0
            per_class_summary[cls_name] = {
                "ground_truth_count": c_gt,
                "true_positives": c_tp,
                "false_positives": c_fp,
                "false_negatives": c_fn,
                "precision": round(c_p, 4) if (c_tp + c_fp) > 0 else "N/A",
                "recall": round(c_r, 4) if c_gt > 0 else "N/A",
                "f1_score": round(c_f1, 4) if (c_p + c_r) > 0 else "N/A",
            }

        results[model_name] = {
            "model_identifier": model_name,
            "preprocessing": "aspect-ratio-preserving letterbox with neutral-grey (114) padding",
            "confidence_threshold": conf_threshold,
            "iou_threshold": iou_threshold,
            "total_ground_truth_objects": total_gt,
            "total_predictions": total_preds,
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "precision": round(precision, 4) if (total_tp + total_fp) > 0 else "N/A",
            "recall": round(recall, 4) if total_gt > 0 else "N/A",
            "f1_score": round(f1, 4) if (precision + recall) > 0 else "N/A",
            "mean_inference_latency_ms": round(mean_lat, 2),
            "total_frames_timed": len(latencies_ms),
            "per_class_metrics": per_class_summary,
        }

    return {
        "track_name": "TRACK_A_NATURAL_IMAGE_DETECTION",
        "evaluation_mode": "REAL_DETECTOR_INFERENCE",
        "description": "Evaluates object detection on real workbench photographs using letterbox preprocessing.",
        "models": results,
    }


def compute_ground_truth_entity_mapping(
    seq: PhysicalWorkbenchSequence,
    store: SQLiteMemoryStore,
) -> Tuple[Dict[str, str], List[str]]:
    """
    Computes bipartite Kuhn-Munkres matching between ground-truth entities and system entities.
    Matches initial entities from Visit 1, then associates subsequent appearances (e.g. notebook in V04).
    Returns (gt_to_sys mapping, list of uninstantiated ground-truth entity IDs).
    """
    all_gt_ids = sorted(list({a.object_id for v in seq.visits for a in v.annotations}))
    v1_anns = [a for a in seq.visits[0].annotations if a.bbox is not None and a.visibility != PhysicalVisibility.ABSENT]

    sys_entities = store.list_entities()
    sys_v1 = [e for e in sys_entities if "visit_01" in e.entity_id or "visit_1" in e.entity_id]
    if not sys_v1:
        sys_v1 = sys_entities

    gt_to_sys: Dict[str, str] = {}
    if v1_anns and sys_v1:
        cost_matrix = np.full((len(v1_anns), len(sys_v1)), 1000.0)
        for i, gt in enumerate(v1_anns):
            for j, sys_ent in enumerate(sys_v1):
                if gt.class_label == sys_ent.label and sys_ent.canonical_bbox is not None:
                    iou = calculate_bbox_iou(gt.bbox, sys_ent.canonical_bbox)
                    dist = calculate_centroid_distance(gt.bbox, sys_ent.canonical_bbox) or 999.0
                    if iou >= 0.05:
                        cost_matrix[i, j] = 1.0 - iou
                    elif dist <= 150.0:
                        cost_matrix[i, j] = 1.0 + (dist / 150.0)

        for i, j in solve_optimal_assignment(cost_matrix):
            if cost_matrix[i, j] < 999.0:
                gt_to_sys[v1_anns[i].object_id] = sys_v1[j].entity_id

    # Allow newly appearing GT entities in subsequent visits to bind to unmapped system entities
    mapped_sys_ids = set(gt_to_sys.values())
    for v in seq.visits[1:]:
        for a in v.annotations:
            if a.object_id not in gt_to_sys and a.bbox is not None and a.visibility != PhysicalVisibility.ABSENT:
                best_match = None
                best_cost = 1000.0
                for ent in sys_entities:
                    if ent.entity_id in mapped_sys_ids or ent.label != a.class_label:
                        continue
                    if ent.canonical_bbox is not None:
                        iou = calculate_bbox_iou(a.bbox, ent.canonical_bbox)
                        dist = calculate_centroid_distance(a.bbox, ent.canonical_bbox) or 999.0
                        cost = (1.0 - iou) if iou >= 0.05 else (1.0 + dist / 150.0 if dist <= 150.0 else 1000.0)
                        if cost < best_cost:
                            best_cost = cost
                            best_match = ent.entity_id
                if best_match is not None and best_cost < 999.0:
                    gt_to_sys[a.object_id] = best_match
                    mapped_sys_ids.add(best_match)

    uninstantiated_gt = [gid for gid in all_gt_ids if gid not in gt_to_sys]
    return gt_to_sys, uninstantiated_gt


def evaluate_sequence_entity_level(
    seq: PhysicalWorkbenchSequence,
    store: SQLiteMemoryStore,
    visit_reports: Dict[int, VisitProcessingReport],
    gt_to_sys: Dict[str, str],
    uninstantiated_gt: List[str],
) -> Dict[str, Any]:
    """
    Primary evaluation measuring per-entity epistemic state correctness across all visits.
    Preserves all uncertainty states (UNKNOWN, AMBIGUOUS_IDENTITY, PENDING_REMOVAL, INSUFFICIENT_EVIDENCE).
    """
    valid_states = [
        EpistemicState.CONFIRMED_PRESENT.value,
        EpistemicState.CONFIRMED_MOVED.value,
        EpistemicState.PENDING_REMOVAL.value,
        EpistemicState.CONFIRMED_REMOVED.value,
        EpistemicState.INSUFFICIENT_EVIDENCE.value,
        EpistemicState.AMBIGUOUS_IDENTITY.value,
    ]
    confusion_matrix: Dict[str, Dict[str, int]] = {
        s1: {s2: 0 for s2 in valid_states} for s1 in valid_states
    }
    for s1 in valid_states:
        confusion_matrix[s1]["UNKNOWN"] = 0

    all_gt_ids = sorted(list({a.object_id for v in seq.visits for a in v.annotations}))
    per_entity_records: Dict[str, Dict[str, Any]] = {}
    for gid in all_gt_ids:
        sample_ann = next(a for v in seq.visits for a in v.annotations if a.object_id == gid)
        per_entity_records[gid] = {
            "entity_id": gid,
            "class_label": sample_ann.class_label,
            "is_instantiated": gid not in uninstantiated_gt,
            "mapped_system_id": gt_to_sys.get(gid),
            "total_visits": 0,
            "correct_visits": 0,
            "accuracy": 0.0,
            "evaluations": [],
        }

    per_class_counts: Dict[str, Dict[str, int]] = {}
    total_evals = 0
    correct_evals = 0

    for v in seq.visits:
        v_rep = visit_reports.get(v.visit_index)
        rep_updates = {u.entity_id: u for u in (v_rep.entity_updates if v_rep else [])}

        for ann in v.annotations:
            gid = ann.object_id
            cls_lbl = ann.class_label
            if cls_lbl not in per_class_counts:
                per_class_counts[cls_lbl] = {"total": 0, "correct": 0}

            expected = ann.expected_state or EpistemicState.INSUFFICIENT_EVIDENCE
            sys_id = gt_to_sys.get(gid)
            observed = None
            explanation = ""

            if sys_id is None:
                observed = "UNKNOWN"
                explanation = "Unmatched ground-truth entity (never instantiated by detector)"
            else:
                if sys_id in rep_updates:
                    observed = rep_updates[sys_id].new_state
                    explanation = rep_updates[sys_id].explanation
                else:
                    for k in range(v.visit_index - 1, 0, -1):
                        prev_rep = visit_reports.get(k)
                        if prev_rep:
                            prev_updates = {u.entity_id: u for u in prev_rep.entity_updates}
                            if sys_id in prev_updates:
                                observed = prev_updates[sys_id].new_state
                                explanation = f"Carried from visit {k}: {prev_updates[sys_id].explanation}"
                                break
                    if observed is None:
                        ent = store.get_entity(sys_id)
                        if ent is not None and v.visit_index == 1:
                            observed = EpistemicState.CONFIRMED_PRESENT
                            explanation = f"Initial baseline entity instantiated in memory store ({ent.entity_id})"
                        elif ent is not None and (
                            ent.metadata.get("created_in_session") == f"{seq.sequence_id}_{v.visit_id}"
                            or ent.metadata.get("created_in_session") == v.visit_id
                        ):
                            observed = EpistemicState.CONFIRMED_PRESENT
                            explanation = f"Entity newly instantiated in memory store for visit {v.visit_id} ({ent.entity_id})"
                        else:
                            observed = EpistemicState.INSUFFICIENT_EVIDENCE
                            explanation = "Historical state record missing for visit"

            # Check match (including return to baseline)
            is_ret = ("return" in (ann.notes or "").lower() or ann.expected_event == "return_to_baseline")
            is_match = (observed == expected)
            if is_ret and observed == EpistemicState.CONFIRMED_PRESENT:
                is_match = True
            if sys_id is None:
                is_match = False

            total_evals += 1
            per_class_counts[cls_lbl]["total"] += 1
            per_entity_records[gid]["total_visits"] += 1

            if is_match:
                correct_evals += 1
                per_class_counts[cls_lbl]["correct"] += 1
                per_entity_records[gid]["correct_visits"] += 1

            exp_str = expected.value if isinstance(expected, EpistemicState) else str(expected)
            obs_str = observed.value if isinstance(observed, EpistemicState) else str(observed)

            if exp_str in confusion_matrix:
                if obs_str in confusion_matrix[exp_str]:
                    confusion_matrix[exp_str][obs_str] += 1
                else:
                    confusion_matrix[exp_str]["UNKNOWN"] += 1

            per_entity_records[gid]["evaluations"].append({
                "visit_id": v.visit_id,
                "visit_index": v.visit_index,
                "expected_state": exp_str,
                "observed_state": obs_str,
                "is_match": is_match,
                "explanation": explanation,
            })

    for gid, d in per_entity_records.items():
        d["accuracy"] = round(d["correct_visits"] / d["total_visits"], 4) if d["total_visits"] > 0 else 0.0

    per_class_results = {
        cls_lbl: {
            "total_evaluations": d["total"],
            "correct_evaluations": d["correct"],
            "accuracy": round(d["correct"] / d["total"], 4) if d["total"] > 0 else 0.0,
        }
        for cls_lbl, d in per_class_counts.items()
    }

    return {
        "metric_name": "ENTITY_LEVEL_STATE_EVALUATION",
        "evaluation_tier": "PRIMARY_METRIC",
        "description": "Evaluates per-entity epistemic state correctness across all visits using Kuhn-Munkres matching.",
        "total_entity_visits_evaluated": total_evals,
        "correct_entity_visits": correct_evals,
        "incorrect_entity_visits": total_evals - correct_evals,
        "entity_state_accuracy": round(correct_evals / total_evals, 4) if total_evals > 0 else 0.0,
        "confusion_matrix": confusion_matrix,
        "per_entity_results": per_entity_records,
        "per_class_results": per_class_results,
        "uninstantiated_gt_entities": uninstantiated_gt,
    }


def evaluate_sequence_event_level(
    seq: PhysicalWorkbenchSequence,
    visit_reports: Dict[int, VisitProcessingReport],
    gt_to_sys: Dict[str, str],
) -> Dict[str, Any]:
    """
    Secondary diagnostic metric evaluating visit-level events bound to their specific target entities.
    Prevents unrelated entity state transitions from satisfying events.
    """
    evaluated_events: List[Dict[str, Any]] = []
    total_evaluated = 0
    total_passed = 0

    primary_visit_records: List[Dict[str, Any]] = []
    primary_total = 0
    primary_passed = 0

    for v in seq.visits:
        if v.visit_index == 1:
            continue

        rep = visit_reports.get(v.visit_index)
        rep_updates = {u.entity_id: u for u in (rep.entity_updates if rep else [])}

        events_to_eval = v.expected_events if v.expected_events else ([v.expected_event] if v.expected_event else [])
        for ev_type in events_to_eval:
            if ev_type in ("baseline_established", "unchanged", "continued_absence"):
                continue

            target_anns = [a for a in v.annotations if a.expected_event == ev_type]
            if not target_anns:
                target_anns = v.annotations

            for target_ann in target_anns:
                total_evaluated += 1
                target_gid = target_ann.object_id
                target_sys_id = gt_to_sys.get(target_gid)
                target_update = rep_updates.get(target_sys_id) if target_sys_id else None

                passed = False
                reason = ""

                if ev_type == "return_to_baseline":
                    if target_update is not None:
                        passed = bool(
                            target_update.supporting_evidence.get("is_return_to_baseline")
                            or (target_update.prior_state == EpistemicState.CONFIRMED_MOVED and target_update.new_state == EpistemicState.CONFIRMED_PRESENT)
                            or target_update.new_state == EpistemicState.CONFIRMED_PRESENT
                        )
                    reason = f"Target {target_gid} ({target_sys_id}): {target_update.new_state.value if target_update else 'no update'}"

                elif ev_type == "object_moved":
                    if target_update is not None:
                        passed = (target_update.new_state == EpistemicState.CONFIRMED_MOVED)
                    reason = f"Target {target_gid} ({target_sys_id}): {target_update.new_state.value if target_update else 'no update'}"

                elif ev_type in ("reappeared", "newly_appeared", "revealed"):
                    if target_update is not None:
                        passed = (target_update.new_state == EpistemicState.CONFIRMED_PRESENT)
                    reason = f"Target {target_gid} ({target_sys_id}): {target_update.new_state.value if target_update else 'no update'}"

                elif ev_type == "object_pending_removal":
                    if target_update is not None:
                        passed = (target_update.new_state == EpistemicState.PENDING_REMOVAL)
                    reason = f"Target {target_gid} ({target_sys_id}): {target_update.new_state.value if target_update else 'no update'}"

                elif ev_type == "object_removed":
                    if target_update is not None:
                        passed = (target_update.new_state == EpistemicState.CONFIRMED_REMOVED)
                    reason = f"Target {target_gid} ({target_sys_id}): {target_update.new_state.value if target_update else 'no update'}"

                elif ev_type in ("baseline_stable", "lighting_change", "viewpoint_jitter"):
                    spurious_moves = [u for u in (rep.entity_updates if rep else []) if u.new_state == EpistemicState.CONFIRMED_MOVED]
                    spurious_removals = [u for u in (rep.entity_updates if rep else []) if u.new_state == EpistemicState.CONFIRMED_REMOVED]
                    passed = (len(spurious_moves) == 0 and len(spurious_removals) == 0)
                    reason = f"Spurious moves: {len(spurious_moves)}, removals: {len(spurious_removals)}"

                if passed:
                    total_passed += 1

                evaluated_events.append({
                    "visit_id": v.visit_id,
                    "visit_index": v.visit_index,
                    "expected_event": ev_type,
                    "target_entity_id": target_gid,
                    "target_sys_id": target_sys_id,
                    "passed": passed,
                    "reason": reason,
                })

        # Backward compatibility: evaluate primary event of the visit
        if v.expected_event and v.expected_event not in ("baseline_established", "unchanged", "continued_absence"):
            primary_total += 1
            primary_anns = [a for a in v.annotations if a.expected_event == v.expected_event]
            p_target = primary_anns[0] if primary_anns else (v.annotations[0] if v.annotations else None)
            p_sys_id = gt_to_sys.get(p_target.object_id) if p_target else None
            p_up = rep_updates.get(p_sys_id) if p_sys_id else None
            p_pass = False

            ev_p = v.expected_event
            if ev_p == "return_to_baseline":
                p_pass = bool(p_up and (p_up.supporting_evidence.get("is_return_to_baseline") or p_up.new_state == EpistemicState.CONFIRMED_PRESENT))
            elif ev_p == "object_moved":
                p_pass = bool(p_up and p_up.new_state == EpistemicState.CONFIRMED_MOVED)
            elif ev_p in ("reappeared", "newly_appeared"):
                p_pass = bool(p_up and p_up.new_state == EpistemicState.CONFIRMED_PRESENT)
            elif ev_p == "object_pending_removal":
                p_pass = bool(p_up and p_up.new_state == EpistemicState.PENDING_REMOVAL)
            elif ev_p == "object_removed":
                p_pass = bool(p_up and p_up.new_state == EpistemicState.CONFIRMED_REMOVED)
            elif ev_p in ("baseline_stable", "lighting_change"):
                p_pass = (len([u for u in (rep.entity_updates if rep else []) if u.new_state in (EpistemicState.CONFIRMED_MOVED, EpistemicState.CONFIRMED_REMOVED)]) == 0)

            if p_pass:
                primary_passed += 1

            primary_visit_records.append({
                "visit_id": v.visit_id,
                "visit_index": v.visit_index,
                "expected_event": ev_p,
                "target_entity_id": p_target.object_id if p_target else None,
                "passed": p_pass,
            })

    acc = round(total_passed / total_evaluated, 4) if total_evaluated > 0 else 0.0
    prim_acc = round(primary_passed / primary_total, 4) if primary_total > 0 else 0.0

    return {
        "metric_name": "TARGET_ENTITY_EVENT_EVALUATION",
        "evaluation_tier": "SECONDARY_DIAGNOSTIC_METRIC",
        "description": "Evaluates visit events bound to specific target entities, preventing unrelated entity transitions from satisfying events.",
        "total_events_evaluated": total_evaluated,
        "successful_events": total_passed,
        "failed_events": total_evaluated - total_passed,
        "event_accuracy": acc,
        "primary_visit_events_evaluated": primary_total,
        "primary_visit_events_passed": primary_passed,
        "primary_visit_event_accuracy": prim_acc,
        "primary_visit_records": primary_visit_records,
        "event_records": evaluated_events,
    }


def evaluate_physical_track_b(
    dataset: PhysicalWorkbenchDataset,
) -> Dict[str, Any]:
    """
    TRACK B: Oracle Temporal Reasoning Evaluation on physical sequences.
    Passes ground-truth bounding boxes directly into TemporalMemoryEngine to isolate
    state machine logic from detector failures.
    Canonical projection via homography is performed by the engine when canonical_bbox=None.
    """
    registrar = PlanarRegistrar()
    extractor = VisualSignatureExtractor()

    sequence_evaluations: Dict[str, Any] = {}
    total_identity_switches = 0
    total_duplicate_tracks = 0

    overall_entity_total = 0
    overall_entity_correct = 0
    overall_event_total = 0
    overall_event_passed = 0
    overall_primary_total = 0
    overall_primary_passed = 0

    for seq_id, seq in dataset.sequences.items():
        store = SQLiteMemoryStore(":memory:")
        engine = TemporalMemoryEngine(store, signature_extractor=extractor)
        canonical_img: Optional[np.ndarray] = None
        visit_reports: Dict[int, VisitProcessingReport] = {}

        seq_identity_switches = 0
        prior_mappings: Dict[str, str] = {}

        for visit in seq.visits:
            if not visit.is_image_present():
                continue

            img = visit.load_image()
            if img is None:
                continue

            if canonical_img is None:
                canonical_img = img
                reg = registrar.register(img, img)
            else:
                reg = registrar.register(canonical_img, img)

            # Construct oracle detections directly from ground-truth annotations
            # canonical_bbox is set to None so the engine automatically projects det.raw_bbox via homography
            oracle_dets = [
                Detection(
                    label=ann.class_label,
                    confidence=1.0,
                    raw_bbox=ann.bbox,
                    canonical_bbox=None,
                    signature=extractor.extract(img, ann.bbox) if ann.bbox else None,
                )
                for ann in visit.annotations
                if ann.bbox is not None and ann.visibility != PhysicalVisibility.ABSENT
            ]

            v_in = VisitInput(
                session_id=f"{seq_id}_{visit.visit_id}",
                session_name=f"{seq_id} {visit.visit_id}",
                timestamp=visit.timestamp or f"2026-10-29T0{visit.visit_index}:00:00Z",
                image=img,
                registration=reg,
                detections=oracle_dets,
            )
            rep = engine.process_visit(v_in)
            visit_reports[visit.visit_index] = rep

            # Track association mapping for active entities
            active_ents = store.list_entities()
            current_mapping: Dict[str, str] = {}
            for ann in visit.annotations:
                if ann.bbox is None or ann.visibility == PhysicalVisibility.ABSENT:
                    continue
                best_id = None
                best_iou = 0.0
                for ent in active_ents:
                    if ent.label == ann.class_label and ent.canonical_bbox is not None:
                        iou = calculate_bbox_iou(ann.bbox, ent.canonical_bbox)
                        if iou > best_iou:
                            best_iou = iou
                            best_id = ent.entity_id
                if best_id is not None and best_iou >= 0.3:
                    current_mapping[ann.object_id] = best_id

            for obj_id, sys_id in current_mapping.items():
                if obj_id in prior_mappings and prior_mappings[obj_id] != sys_id:
                    seq_identity_switches += 1
            prior_mappings = current_mapping

        total_system_ents = len(store.list_entities())
        unique_gt_ids = len({a.object_id for v in seq.visits for a in v.annotations})
        seq_duplicates = max(0, total_system_ents - unique_gt_ids)

        total_identity_switches += seq_identity_switches
        total_duplicate_tracks += seq_duplicates

        # Bipartite Ground-Truth Mapping
        gt_to_sys, uninstantiated_gt = compute_ground_truth_entity_mapping(seq, store)

        # Primary Metric: Entity-Level State Evaluation
        entity_report = evaluate_sequence_entity_level(
            seq=seq,
            store=store,
            visit_reports=visit_reports,
            gt_to_sys=gt_to_sys,
            uninstantiated_gt=uninstantiated_gt,
        )

        # Secondary Diagnostic Metric: Target-Entity Event Evaluation
        event_report = evaluate_sequence_event_level(
            seq=seq,
            visit_reports=visit_reports,
            gt_to_sys=gt_to_sys,
        )

        overall_entity_total += entity_report["total_entity_visits_evaluated"]
        overall_entity_correct += entity_report["correct_entity_visits"]
        overall_event_total += event_report["total_events_evaluated"]
        overall_event_passed += event_report["successful_events"]
        overall_primary_total += event_report["primary_visit_events_evaluated"]
        overall_primary_passed += event_report["primary_visit_events_passed"]

        sequence_evaluations[seq_id] = {
            "identity_switches": seq_identity_switches,
            "duplicate_tracks": seq_duplicates,
            "entity_level_evaluation": entity_report,
            "secondary_event_evaluation": event_report,
        }

    overall_ent_acc = (overall_entity_correct / overall_entity_total) if overall_entity_total > 0 else 0.0
    overall_prim_acc = (overall_primary_passed / overall_primary_total) if overall_primary_total > 0 else 0.0

    event_acc = round(overall_prim_acc, 4) if overall_primary_total > 0 else "N/A"

    return {
        "track_name": "TRACK_B_ORACLE_TEMPORAL_EVALUATION",
        "evaluation_mode": "ORACLE_GROUND_TRUTH_INPUTS",
        "description": "Evaluates multi-visit temporal reasoning and entity association with idealized ground-truth bounding boxes.",
        "primary_metric": {
            "metric_name": "ENTITY_LEVEL_STATE_ACCURACY",
            "total_entity_visits": overall_entity_total,
            "correct_entity_visits": overall_entity_correct,
            "accuracy": round(overall_ent_acc, 4) if overall_entity_total > 0 else "N/A",
        },
        "secondary_diagnostic_metric": {
            "metric_name": "TARGET_ENTITY_EVENT_ACCURACY",
            "primary_visit_events_evaluated": overall_primary_total,
            "primary_visit_events_passed": overall_primary_passed,
            "primary_visit_event_accuracy": event_acc,
            "multi_events_evaluated": overall_event_total,
            "multi_events_passed": overall_event_passed,
        },
        "total_events_evaluated": overall_primary_total,
        "successful_events": overall_primary_passed,
        "failed_events": overall_primary_total - overall_primary_passed,
        "event_accuracy": event_acc,
        "total_identity_switches": total_identity_switches,
        "total_duplicate_tracks": total_duplicate_tracks,
        "sequences": sequence_evaluations,
    }


def evaluate_physical_track_c(
    dataset: PhysicalWorkbenchDataset,
    detector_name: str = "yolov8n",
    registry: Optional[DetectorRegistry] = None,
    conf_threshold: float = 0.25,
) -> Dict[str, Any]:
    """
    TRACK C: End-to-End Physical Pipeline Evaluation.
    Processes real photographs through the actual complete pipeline:
    Detector (actual predictions) -> Keyframe Registration -> Coordinate Projection ->
    Visual Signatures -> Entity Association -> Temporal Engine -> Persistence.
    """
    if registry is None:
        registry = DetectorRegistry()

    detector = registry.get(detector_name)
    sequence_reports: Dict[str, Any] = {}

    total_visits_processed = 0
    total_detections_produced = 0
    total_registration_failures = 0
    total_identity_switches = 0
    total_duplicate_tracks = 0

    overall_entity_total = 0
    overall_entity_correct = 0
    overall_event_total = 0
    overall_event_passed = 0
    overall_primary_total = 0
    overall_primary_passed = 0

    for seq_id, seq in dataset.sequences.items():
        store = SQLiteMemoryStore(":memory:")
        runner = RealImagePipelineRunner(
            store=store,
            detector=detector,
            temporal_config=TemporalEngineConfig(
                movement_displacement_threshold=35.0,
                min_registration_inlier_ratio=0.15,
            ),
        )

        seq_detections = 0
        seq_reg_failures = 0
        visit_reports: Dict[int, VisitProcessingReport] = {}
        prior_mappings: Dict[str, str] = {}
        seq_identity_switches = 0

        for visit in seq.visits:
            if not visit.is_image_present():
                continue

            img = visit.load_image()
            if img is None:
                continue

            total_visits_processed += 1
            h_img, w_img = img.shape[:2]

            from objectmemory.datasets.real_dataset import RealVisitFrame
            frame = RealVisitFrame(
                visit_id=visit.visit_id,
                session_name=f"{seq_id} {visit.visit_id}",
                timestamp=visit.timestamp or f"2026-10-29T0{visit.visit_index}:00:00Z",
                image_path=Path(visit.image_absolute_path),
            )

            res = runner.process_visit(frame, confidence_threshold=conf_threshold)
            seq_detections += len(res.detection_batch.detections)
            total_detections_produced += len(res.detection_batch.detections)

            if res.registration.status != AlignmentStatus.SUCCESS:
                seq_reg_failures += 1
                total_registration_failures += 1

            visit_reports[visit.visit_index] = res.temporal_report

            active_ents = store.list_entities()
            current_mapping: Dict[str, str] = {}
            for ann in visit.annotations:
                if ann.bbox is None or ann.visibility == PhysicalVisibility.ABSENT:
                    continue
                best_id = None
                best_iou = 0.0
                for ent in active_ents:
                    if ent.label == ann.class_label and ent.canonical_bbox is not None:
                        iou = calculate_bbox_iou(ann.bbox, ent.canonical_bbox)
                        if iou > best_iou:
                            best_iou = iou
                            best_id = ent.entity_id
                if best_id is not None and best_iou >= 0.3:
                    current_mapping[ann.object_id] = best_id

            for obj_id, sys_id in current_mapping.items():
                if obj_id in prior_mappings and prior_mappings[obj_id] != sys_id:
                    seq_identity_switches += 1
            prior_mappings = current_mapping

        total_system_ents = len(store.list_entities())
        unique_gt_ids = len({a.object_id for v in seq.visits for a in v.annotations})
        seq_duplicates = max(0, total_system_ents - unique_gt_ids)

        total_identity_switches += seq_identity_switches
        total_duplicate_tracks += seq_duplicates

        # Bipartite Ground-Truth Mapping
        gt_to_sys, uninstantiated_gt = compute_ground_truth_entity_mapping(seq, store)

        # Primary Metric: Entity-Level State Evaluation
        entity_report = evaluate_sequence_entity_level(
            seq=seq,
            store=store,
            visit_reports=visit_reports,
            gt_to_sys=gt_to_sys,
            uninstantiated_gt=uninstantiated_gt,
        )

        # Secondary Diagnostic Metric: Target-Entity Event Evaluation
        event_report = evaluate_sequence_event_level(
            seq=seq,
            visit_reports=visit_reports,
            gt_to_sys=gt_to_sys,
        )

        overall_entity_total += entity_report["total_entity_visits_evaluated"]
        overall_entity_correct += entity_report["correct_entity_visits"]
        overall_event_total += event_report["total_events_evaluated"]
        overall_event_passed += event_report["successful_events"]
        overall_primary_total += event_report["primary_visit_events_evaluated"]
        overall_primary_passed += event_report["primary_visit_events_passed"]

        sequence_reports[seq_id] = {
            "visits_processed": len(visit_reports),
            "detections_produced": seq_detections,
            "registration_failures": seq_reg_failures,
            "entities_persisted": total_system_ents,
            "duplicate_tracks": seq_duplicates,
            "identity_switches": seq_identity_switches,
            "entity_level_evaluation": entity_report,
            "secondary_event_evaluation": event_report,
        }

    overall_ent_acc = (overall_entity_correct / overall_entity_total) if overall_entity_total > 0 else 0.0
    overall_prim_acc = (overall_primary_passed / overall_primary_total) if overall_primary_total > 0 else 0.0

    event_acc = round(overall_prim_acc, 4) if overall_primary_total > 0 else "N/A"

    return {
        "track_name": "TRACK_C_END_TO_END_PHYSICAL_EVALUATION",
        "evaluation_mode": "REAL_DETECTOR_INPUT",
        "detector_used": f"{detector_name} (letterbox=True)",
        "confidence_threshold": conf_threshold,
        "description": "Full end-to-end processing of photographic workbench imagery using actual detector outputs.",
        "primary_metric": {
            "metric_name": "ENTITY_LEVEL_STATE_ACCURACY",
            "total_entity_visits": overall_entity_total,
            "correct_entity_visits": overall_entity_correct,
            "accuracy": round(overall_ent_acc, 4) if overall_entity_total > 0 else "N/A",
        },
        "secondary_diagnostic_metric": {
            "metric_name": "TARGET_ENTITY_EVENT_ACCURACY",
            "primary_visit_events_evaluated": overall_primary_total,
            "primary_visit_events_passed": overall_primary_passed,
            "primary_visit_event_accuracy": event_acc,
            "multi_events_evaluated": overall_event_total,
            "multi_events_passed": overall_event_passed,
        },
        "total_visits_processed": total_visits_processed,
        "total_detections_produced": total_detections_produced,
        "total_registration_failures": total_registration_failures,
        "total_events_evaluated": overall_primary_total,
        "successful_events": overall_primary_passed,
        "failed_events": overall_primary_total - overall_primary_passed,
        "end_to_end_event_accuracy": event_acc,
        "total_identity_switches": total_identity_switches,
        "total_duplicate_tracks": total_duplicate_tracks,
        "sequences": sequence_reports,
    }


def main():
    parser = argparse.ArgumentParser(description="ObjectMemory Phase 14C Physical Workbench Benchmark")
    parser.add_argument(
        "--dataset-dir",
        default="data/physical_workbench_v1",
        help="Path to physical workbench dataset directory",
    )
    parser.add_argument(
        "--output-file",
        default="scripts/phase14c3_metric_integrity_results.json",
        help="Path to save machine-readable evaluation results",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="Confidence threshold for detector inference",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.50,
        help="IoU matching threshold for detection evaluation",
    )
    args = parser.parse_args()

    dataset_path = os.path.abspath(args.dataset_dir)
    print(f"Loading Physical Workbench Dataset from: {dataset_path}")

    try:
        dataset = PhysicalWorkbenchDataset.load(dataset_path)
    except Exception as e:
        print(f"[FATAL ERROR] Failed to load dataset: {e}")
        sys.exit(1)

    capture_state = dataset.capture_status
    print(f"Dataset capture status: {capture_state.value}")

    total_sequences = len(dataset.sequences)
    total_visits = sum(len(seq.visits) for seq in dataset.sequences.values())
    total_annotations = sum(len(v.annotations) for seq in dataset.sequences.values() for v in seq.visits)
    images_found = sum(1 for seq in dataset.sequences.values() for v in seq.visits if v.is_image_present())

    # Build evaluation output structure
    results_payload: Dict[str, Any] = {
        "benchmark_name": "ObjectMemory Physical Workbench Benchmark (7 Sequences, 35 Visits)",
        "dataset_name": dataset.manifest_data.get("dataset_name", "Physical Workbench Benchmark v1"),
        "dataset_version": dataset.manifest_data.get("version", "1.0.0"),
        "evaluation_version": "14c.3",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": dataset.manifest_data.get("provenance", "Physical laboratory capture"),
        "capture_status": capture_state.value,
        "summary_counts": {
            "total_sequences": total_sequences,
            "total_visits": total_visits,
            "images_present_on_disk": images_found,
            "images_awaiting_capture": total_visits - images_found,
            "total_annotations": total_annotations,
        },
        "evaluation_configuration": {
            "confidence_threshold": args.conf_threshold,
            "iou_threshold": args.iou_threshold,
            "canonical_classes": sorted(list(dataset.CANONICAL_CLASSES)),
            "movement_threshold_px": 35.0,
            "return_to_baseline_threshold_px": 35.0,
            "min_registration_inlier_ratio": 0.15,
        },
        "tracks": {},
        "known_limitations_and_provenance_notes": [
            "Track A requires genuine photographic imagery to compute valid precision, recall, and mAP.",
            "Track B evaluates multi-visit state transitions with oracle bounding boxes to establish algorithmic upper bounds.",
            "Track C evaluates actual detector predictions through the complete pipeline without substituting ground-truth boxes.",
            "Planar homography registration assumes approximately coplanar workbench surface; significant 3D camera parallax will cause perspective distortion.",
            "The 35-pixel movement threshold is defined in nominal 1920x1080 pixel coordinates.",
        ],
    }

    if capture_state == DatasetCaptureStatus.AWAITING_PHYSICAL_DATA:
        print("\n" + "=" * 65)
        print("  NOTICE: AWAITING PHYSICAL PHOTOGRAPHS")
        print("=" * 65)
        print("  The Physical Workbench Benchmark ingestion and evaluation harness")
        print("  is fully implemented, validated, and ready.")
        print(f"  Total Sequences Configured: {total_sequences}")
        print(f"  Total Visits Configured:    {total_visits}")
        print(f"  Total Annotations Ready:    {total_annotations}")
        print("  Real photographs have not yet been placed in the images/ directory.")
        print("  Per Phase 14C scientific rules: NO FABRICATED METRICS ARE GENERATED.")
        print("  Benchmark status recorded honestly as AWAITING_PHYSICAL_DATA.")
        print("=" * 65)

        results_payload["tracks"] = {
            "track_a_natural_image_detection": {
                "status": "PENDING_PHYSICAL_DATA",
                "message": "Awaiting user capture of 35 photographic visit frames in data/physical_workbench_v1/images/.",
                "metrics": None,
            },
            "track_b_oracle_temporal_evaluation": {
                "status": "PENDING_PHYSICAL_DATA",
                "message": "Awaiting user capture of photographic visit frames for image feature extraction.",
                "metrics": None,
            },
            "track_c_end_to_end_physical_evaluation": {
                "status": "PENDING_PHYSICAL_DATA",
                "message": "Awaiting user capture of photographic visit frames for full pipeline execution.",
                "metrics": None,
            },
        }

    else:
        print("\n--- Executing Track A: Natural-Image Object Detection ---")
        registry = DetectorRegistry()
        track_a_res = evaluate_physical_track_a(
            dataset=dataset,
            registry=registry,
            conf_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold,
        )

        print("\n--- Executing Track B: Oracle Temporal Reasoning ---")
        track_b_res = evaluate_physical_track_b(dataset=dataset)

        print("\n--- Executing Track C: End-to-End Physical Evaluation ---")
        track_c_res = evaluate_physical_track_c(
            dataset=dataset,
            detector_name="yolov8n",
            registry=registry,
            conf_threshold=args.conf_threshold,
        )

        results_payload["tracks"] = {
            "track_a_natural_image_detection": track_a_res,
            "track_b_oracle_temporal_evaluation": track_b_res,
            "track_c_end_to_end_physical_evaluation": track_c_res,
        }

    out_file = os.path.abspath(args.output_file)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print(f"\nSaved Phase 14C evaluation results to: {out_file}")
    sys.exit(0)


if __name__ == "__main__":
    main()
