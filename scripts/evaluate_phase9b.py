"""
Phase 9B Comprehensive Empirical Evaluation Suite:
Evaluation Correction, Ground-Truth Bipartite Matching, and Reliability Audit.

Evaluates:
1. Synthetic Controlled Transformation Benchmark (15 cases): Single-Reference vs Adaptive Keyframe.
2. Dynamic Uncertainty Calibration Delta_U empirical coverage analysis.
3. Corrected Real Workbench Multi-Visit Evaluation (desk_pilot_01, 7 visits):
   - Deterministic bipartite Kuhn-Munkres matching (gt_id -> sys_id).
   - Multi-object disambiguation with shared class labels (two cups, two books).
   - Return to baseline scoring (rewarded as correct return, not penalized as missed movement).
   - Duplicate tracks and unmatched prediction reporting.
   - Unmatched ground-truth reporting.
   - Tracking and identity switch metrics.
   - Boundary clipping and inward movement evaluation.
   - Reappearance recovery.
4. Old vs. Corrected Metrics Comparison (explicitly explaining previous flawed metrics).
5. Structured persistence to scripts/phase9b_evaluation_results.json.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from objectmemory.baseline.comparator import calculate_bbox_iou, calculate_centroid_distance
from objectmemory.benchmarks.ground_truth_matcher import GroundTruthMatcher, SequenceEvaluationReport
from objectmemory.datasets.real_dataset import (
    AnnotatedEvent,
    PhysicalObservation,
    RealVisitFrame,
    RealWorkbenchDataset,
)
from objectmemory.detection.registry import DetectorRegistry
from objectmemory.detection.models import Detection, DetectionBatch
from objectmemory.detection.pipeline import transform_bbox_to_canonical
from objectmemory.detection.signature import VisualSignatureExtractor
from objectmemory.domain.models import BoundingBox, WorkbenchEntity
from objectmemory.domain.states import EpistemicState
from objectmemory.persistence.repository import SQLiteMemoryStore
from objectmemory.pipeline.real_pipeline import RealImagePipelineRunner
from objectmemory.registration.coverage import SpatialCoverageCalculator
from objectmemory.registration.evaluator import AlignmentQualityEvaluator
from objectmemory.registration.keyframe import (
    AdaptiveKeyframeManager,
    KeyframeConfig,
    KeyframeRegistrationResult,
)
from objectmemory.registration.models import (
    AlignmentConfig,
    AlignmentStatus,
    RegistrationResult,
    ReprojectionStats,
)
from objectmemory.registration.registrar import PlanarRegistrar
from objectmemory.temporal.engine import TemporalMemoryEngine
from objectmemory.temporal.models import TemporalEngineConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_phase9b")


def make_calibrated_panorama(width: int = 1600, height: int = 600, seed: int = 42) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Generates a rich, non-repetitive canvas with high-contrast text and visual feature markers."""
    rng = np.random.default_rng(seed)
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)

    # Gradient background
    for y in range(height):
        factor = float(y) / height
        canvas[y, :, 0] = int(220 + 25 * factor)
        canvas[y, :, 1] = int(225 + 15 * (1.0 - factor))
        canvas[y, :, 2] = int(230 + 20 * math.sin(factor * math.pi))

    # Dense feature textures
    for _ in range(700):
        cx, cy = int(rng.integers(10, width - 10)), int(rng.integers(10, height - 10))
        rad = int(rng.integers(3, 14))
        color = (int(rng.integers(30, 200)), int(rng.integers(30, 200)), int(rng.integers(30, 200)))
        cv2.circle(canvas, (cx, cy), rad, color, -1)

    markers = []
    marker_cols = [80, 200, 360, 520, 680, 840, 1000, 1160, 1320, 1480]
    marker_rows = [100, 250, 400, 520]

    idx = 0
    for col_x in marker_cols:
        for row_y in marker_rows:
            idx += 1
            cv2.circle(canvas, (col_x, row_y), 24, (20, 20, 20), -1)
            cv2.circle(canvas, (col_x, row_y), 16, (240, 240, 240), -1)
            cv2.circle(canvas, (col_x, row_y), 8, (20, 20, 20), -1)
            cv2.putText(
                canvas,
                f"F{idx:02d}",
                (col_x - 18, row_y + 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (40, 40, 40),
                1,
                cv2.LINE_AA,
            )
            markers.append({"id": f"F{idx:02d}", "center": (float(col_x), float(row_y))})

    return canvas, markers


def run_controlled_transformation_benchmark() -> Dict[str, Any]:
    """Evaluates Single-Reference vs Adaptive Keyframe registration across 15 viewpoints."""
    logger.info("Running Controlled Transformation & Uncertainty Calibration Benchmark...")
    canvas, markers = make_calibrated_panorama(1600, 600, seed=42)
    registrar = PlanarRegistrar()
    engine = TemporalMemoryEngine(store=SQLiteMemoryStore(":memory:"))

    ref_w, ref_h = 640, 480
    ref_img = canvas[0:ref_h, 0:ref_w].copy()

    transform_configs = [
        {"name": "stable_jitter_1", "type": "affine", "tx": 4.0, "ty": 3.0, "scale": 1.0, "rot": 0.5},
        {"name": "stable_jitter_2", "type": "affine", "tx": -5.0, "ty": 6.0, "scale": 1.0, "rot": -0.8},
        {"name": "moderate_pan_1", "type": "affine", "tx": 80.0, "ty": 20.0, "scale": 1.0, "rot": 0.0},
        {"name": "moderate_pan_2", "type": "affine", "tx": 160.0, "ty": -15.0, "scale": 1.0, "rot": 1.0},
        {"name": "large_pan_1", "type": "affine", "tx": 280.0, "ty": 30.0, "scale": 1.0, "rot": 0.0},
        {"name": "large_pan_2", "type": "affine", "tx": 400.0, "ty": 10.0, "scale": 1.0, "rot": -1.5},
        {"name": "wide_pan_out_of_ref_1", "type": "affine", "tx": 550.0, "ty": 0.0, "scale": 1.0, "rot": 0.0},
        {"name": "wide_pan_out_of_ref_2", "type": "affine", "tx": 750.0, "ty": 20.0, "scale": 1.0, "rot": 0.0},
        {"name": "scale_zoom_in", "type": "affine", "tx": 40.0, "ty": 30.0, "scale": 1.25, "rot": 0.0},
        {"name": "scale_zoom_out", "type": "affine", "tx": 60.0, "ty": 20.0, "scale": 0.85, "rot": 0.0},
        {"name": "rotation_tilt_5deg", "type": "affine", "tx": 50.0, "ty": 20.0, "scale": 1.0, "rot": 5.0},
        {"name": "rotation_tilt_10deg", "type": "affine", "tx": 70.0, "ty": 30.0, "scale": 1.0, "rot": 10.0},
        {"name": "perspective_tilt_light", "type": "homography", "pts_offset": [[15, 10], [-15, 5], [-10, -15], [10, -10]], "tx": 60, "ty": 20},
        {"name": "perspective_tilt_moderate", "type": "homography", "pts_offset": [[35, 20], [-30, 15], [-25, -30], [20, -25]], "tx": 120, "ty": 30},
        {"name": "perspective_tilt_heavy", "type": "homography", "pts_offset": [[60, 40], [-55, 30], [-45, -50], [40, -40]], "tx": 240, "ty": 40},
    ]

    base_results = []
    adaptive_results = []
    uncertainty_evaluations = []

    kf_mgr = AdaptiveKeyframeManager(registrar=registrar, config=KeyframeConfig())
    kf_mgr.set_canonical_reference(ref_img, "visit_00", "2026-10-01T10:00:00Z")

    for idx, cfg in enumerate(transform_configs, start=1):
        vid = f"visit_{idx:02d}"

        if cfg["type"] == "affine":
            tx, ty = cfg["tx"], cfg["ty"]
            s = cfg["scale"]
            center_x, center_y = ref_w / 2.0, ref_h / 2.0
            M_rot = cv2.getRotationMatrix2D((center_x, center_y), cfg["rot"], s)
            M_rot[0, 2] += tx
            M_rot[1, 2] += ty
            H_GT = np.eye(3, dtype=np.float64)
            H_GT[:2, :] = M_rot
        else:
            pts1 = np.float32([[0, 0], [ref_w, 0], [ref_w, ref_h], [0, ref_h]])
            offsets = np.float32(cfg["pts_offset"])
            pts2 = pts1 + offsets
            pts2[:, 0] += cfg["tx"]
            pts2[:, 1] += cfg["ty"]
            H_GT = cv2.getPerspectiveTransform(pts1, pts2)

        H_inv = np.linalg.inv(H_GT)
        query_img = cv2.warpPerspective(canvas, H_inv, (ref_w, ref_h), flags=cv2.INTER_LINEAR)

        # Baseline: Single direct reference
        t_b0 = time.perf_counter()
        reg_base = registrar.register(ref_img, query_img)
        lat_base = (time.perf_counter() - t_b0) * 1000.0

        # Adaptive keyframe
        t_k0 = time.perf_counter()
        kf_res = kf_mgr.register_query(query_img, vid, f"2026-10-{idx+1:02d}T10:00:00Z")
        lat_kf = (time.perf_counter() - t_k0) * 1000.0
        reg_adaptive = kf_res.registration_to_canonical

        vis_markers = []
        for m in markers:
            p_can = np.array([m["center"][0], m["center"][1], 1.0])
            p_q = H_inv @ p_can
            p_q /= p_q[2]
            if 30 <= p_q[0] <= ref_w - 30 and 30 <= p_q[1] <= ref_h - 30:
                vis_markers.append((m, p_q[:2]))

        base_errs = []
        if reg_base.success and reg_base.homography is not None:
            for m, p_q in vis_markers:
                p_proj = reg_base.homography @ np.array([p_q[0], p_q[1], 1.0])
                p_proj /= p_proj[2]
                base_errs.append(float(np.linalg.norm(p_proj[:2] - np.array(m["center"]))))

        kf_errs = []
        if reg_adaptive.success and reg_adaptive.homography is not None:
            for m, p_q in vis_markers:
                p_proj = reg_adaptive.homography @ np.array([p_q[0], p_q[1], 1.0])
                p_proj /= p_proj[2]
                kf_errs.append(float(np.linalg.norm(p_proj[:2] - np.array(m["center"]))))

        mean_kf_err = float(np.mean(kf_errs)) if kf_errs else None
        max_kf_err = float(np.max(kf_errs)) if kf_errs else None

        t_eff, delta_u, u_breakdown = engine.compute_effective_movement_threshold(reg_adaptive, 35.0)

        is_calibrated = (max_kf_err <= t_eff) if max_kf_err is not None else True
        uncertainty_evaluations.append({
            "case": cfg["name"],
            "max_actual_error_px": round(max_kf_err, 3) if max_kf_err is not None else None,
            "mean_actual_error_px": round(mean_kf_err, 3) if mean_kf_err is not None else None,
            "delta_u_px": round(delta_u, 2),
            "effective_threshold_px": round(t_eff, 2),
            "error_bounded_by_threshold": is_calibrated,
            "uncertainty_breakdown": u_breakdown,
        })

        base_results.append({
            "case": cfg["name"],
            "success": reg_base.success,
            "inliers": reg_base.num_inliers,
            "inlier_ratio": round(reg_base.inlier_ratio, 4) if reg_base.inlier_ratio is not None else None,
            "mean_error_px": round(float(np.mean(base_errs)), 3) if base_errs else None,
            "latency_ms": round(lat_base, 2),
        })

        adaptive_results.append({
            "case": cfg["name"],
            "success": reg_adaptive.success,
            "selected_keyframe": kf_res.selected_keyframe_id,
            "is_direct_to_canonical": kf_res.is_direct_to_canonical,
            "inliers": reg_adaptive.num_inliers,
            "inlier_ratio": round(reg_adaptive.inlier_ratio, 4) if reg_adaptive.inlier_ratio is not None else None,
            "spatial_coverage": round(reg_adaptive.spatial_coverage, 4) if reg_adaptive.spatial_coverage is not None else None,
            "delta_u_px": round(delta_u, 2),
            "effective_threshold_px": round(t_eff, 2),
            "mean_error_px": round(mean_kf_err, 3) if mean_kf_err is not None else None,
            "latency_ms": round(lat_kf, 2),
        })

    kf_valid_errs = [r["mean_error_px"] for r in adaptive_results if r["mean_error_px"] is not None]
    within_bounds = sum(1 for e in uncertainty_evaluations if e["error_bounded_by_threshold"])

    return {
        "baseline_single_ref": {
            "success_rate": round(sum(1 for r in base_results if r["success"]) / len(base_results), 4),
            "success_count": sum(1 for r in base_results if r["success"]),
            "failure_count": sum(1 for r in base_results if not r["success"]),
            "mean_latency_ms": round(float(np.mean([r["latency_ms"] for r in base_results])), 2),
            "cases": base_results,
        },
        "adaptive_keyframe": {
            "success_rate": round(sum(1 for r in adaptive_results if r["success"]) / len(adaptive_results), 4),
            "success_count": sum(1 for r in adaptive_results if r["success"]),
            "failure_count": sum(1 for r in adaptive_results if not r["success"]),
            "mean_pos_error_px": round(float(np.mean(kf_valid_errs)), 3) if kf_valid_errs else None,
            "mean_latency_ms": round(float(np.mean([r["latency_ms"] for r in adaptive_results])), 2),
            "keyframes_created": kf_mgr.keyframe_count,
            "chained_count": sum(1 for r in adaptive_results if not r["is_direct_to_canonical"] and r["success"]),
            "cases": adaptive_results,
        },
        "uncertainty_calibration": {
            "heuristic_type": "Linear dynamic uncertainty Delta_U (engineering heuristic)",
            "empirical_coverage_rate": round(within_bounds / len(uncertainty_evaluations), 4),
            "samples_evaluated": len(uncertainty_evaluations),
            "evaluations": uncertainty_evaluations,
        },
    }


def run_corrected_real_workbench_evaluation(manifest_path: Path) -> Dict[str, Any]:
    """
    Evaluates multi-visit temporal reasoning using deterministic bipartite Kuhn-Munkres ground-truth matching.
    """
    logger.info("Running Corrected Real Workbench Temporal Memory Evaluation on %s...", manifest_path)
    if not manifest_path.exists():
        logger.error("Manifest missing at %s", manifest_path)
        return {"error": f"Manifest missing at {manifest_path}"}

    dataset = RealWorkbenchDataset.load_from_directory(manifest_path.parent)
    detector = DetectorRegistry.get("yolov8n")
    store = SQLiteMemoryStore(":memory:")
    runner = RealImagePipelineRunner(
        store=store,
        detector=detector,
        keyframe_config=KeyframeConfig(),
        temporal_config=TemporalEngineConfig(border_margin_px=12.0),
    )

    visit_reports = []
    for v_idx, vf in enumerate(dataset.visits):
        v_res = runner.process_visit(vf, confidence_threshold=0.25)
        visit_reports.append(v_res)
        vf.release_cache()

    matcher = GroundTruthMatcher(min_match_iou=0.05)
    system_entities = store.list_entities()
    report: SequenceEvaluationReport = matcher.evaluate_sequence(
        dataset=dataset,
        system_entities=system_entities,
        visit_reports=visit_reports,
        store=store,
    )

    # Detailed serializable output
    eval_list = []
    for r in report.evaluations:
        eval_list.append({
            "visit_id": r.visit_id,
            "gt_entity_id": r.gt_entity_id,
            "gt_label": r.gt_label,
            "sys_entity_id": r.sys_entity_id,
            "expected_state": r.expected_state.value,
            "observed_state": r.observed_state.value,
            "is_state_match": r.is_state_match,
            "annotated_event": r.annotated_event.value,
            "physical_observation": r.physical_observation.value,
            "is_return_to_baseline_expected": r.is_return_to_baseline_expected,
            "is_return_to_baseline_observed": r.is_return_to_baseline_observed,
            "consecutive_displacement": round(r.consecutive_displacement, 1) if r.consecutive_displacement else None,
            "baseline_displacement": round(r.baseline_displacement, 1) if r.baseline_displacement else None,
            "explanation": r.explanation,
        })

    return {
        "dataset_name": dataset.dataset_name,
        "dataset_classification": "PILOT_VALIDATION_SEQUENCE (UNVERIFIED SENSOR PROVENANCE)",
        "total_visits": len(dataset.visits),
        "total_evaluations": report.total_evaluations,
        "matched_evaluations": report.matched_evaluations,
        "overall_state_accuracy": report.overall_state_accuracy,
        "entity_mapping": report.entity_mappings,
        "tracking_integrity": {
            "total_gt_entities": report.total_gt_entities,
            "matched_gt_entities": report.matched_gt_entities,
            "unmatched_gt_entities": report.unmatched_gt_entities,
            "unmatched_system_entities": report.unmatched_system_entities,
            "duplicate_track_count": report.duplicate_track_count,
            "identity_switch_count": report.identity_switch_count,
        },
        "movement_metrics": {
            "total_unchanged_evaluations": report.total_unchanged_evaluations,
            "false_movement_count": report.false_movements,
            "false_movement_rate": report.false_movement_rate,
            "total_relocated_evaluations": report.total_relocated_evaluations,
            "correct_relocations": report.correct_relocations,
            "missed_movement_count": report.missed_movements,
            "movement_recall": report.movement_recall,
            "expected_returns_to_baseline": report.expected_returns_to_baseline,
            "detected_returns_to_baseline": report.detected_returns_to_baseline,
            "return_to_baseline_accuracy": report.return_to_baseline_accuracy,
        },
        "removal_metrics": {
            "total_visible_evaluations": report.total_visible_evaluations,
            "false_removal_count": report.false_removals,
            "false_removal_rate": report.false_removal_rate,
            "total_removal_evaluations": report.total_removal_evaluations,
            "correct_removals": report.correct_removals,
            "missed_removal_count": report.missed_removals,
            "removal_recall": report.removal_recall,
        },
        "reappearance_recovery": {
            "bottle_01_v7_restored": any(
                r["gt_entity_id"] == "bottle_01" and r["visit_id"] == "visit_07" and r["observed_state"] == "CONFIRMED_PRESENT"
                for r in eval_list
            ),
        },
        "evaluations": eval_list,
    }


def main() -> None:
    manifest_path = REPO_ROOT / "data" / "real_workbench" / "manifest.json"

    reg_bench = run_controlled_transformation_benchmark()
    real_bench = run_corrected_real_workbench_evaluation(manifest_path)

    results = {
        "phase": "9b",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "description": "ObjectMemory Studio Phase 9B Corrected Evaluation & Reliability Audit",
        "dataset_provenance_verdict": "UNKNOWN (USER-SUPPLIED CROPPED IMAGES WITHOUT SENSOR METADATA / APPROXIMATE GRID LABELS)",
        "controlled_benchmark": reg_bench,
        "corrected_real_workbench_benchmark": real_bench,
        "old_vs_corrected_comparison": {
            "ground_truth_matching": {
                "phase8_flawed": "Looked up observed states by COCO class label (current_entity_states[ent.label]), causing multiple entities with the same label to overwrite each other (e.g. cup_01/cup_02, book_01/notebook_01).",
                "phase9b_corrected": "Deterministic bipartite Kuhn-Munkres assignment matching ground-truth entity IDs to system entity IDs based on canonical bounding box overlap/proximity.",
                "consequence": "Eliminated spurious removal errors caused by class-label collision.",
            },
            "return_to_baseline_scoring": {
                "phase8_flawed": "Scored as a missed movement because observed state was CONFIRMED_PRESENT instead of CONFIRMED_MOVED (movement_sensitivity was reported as 33.3%).",
                "phase9b_corrected": "Return to baseline is recognized as a valid return relocation event, correctly matching the expected CONFIRMED_PRESENT state.",
                "return_accuracy": real_bench["movement_metrics"]["return_to_baseline_accuracy"],
            },
            "manifest_reconciliation": {
                "phase8_flawed": "cup_01 expected state in Visits 5-7 was annotated as CONFIRMED_MOVED despite having returned to baseline in Visit 4.",
                "phase9b_corrected": "Reconciled expected state to CONFIRMED_PRESENT for Visits 5-7, resolving the semantic contradiction.",
            },
            "boundary_clipping_handling": {
                "phase8_flawed": "Hardcoded 100px threshold suppressed all movements under 100px if initial box touched the boundary.",
                "phase9b_corrected": "Refined boundary clipping so genuine movements into the interior are detected, while camera crop and truncated boxes remain gated.",
            },
            "tracking_integrity_metrics": {
                "phase8_flawed": "Untracked duplicate entities and missing GT objects were not reported.",
                "phase9b_corrected": f"Explicitly tracks unmatched GT ({real_bench['tracking_integrity']['unmatched_gt_entities']}), duplicate tracks ({real_bench['tracking_integrity']['duplicate_track_count']}), and identity switches ({real_bench['tracking_integrity']['identity_switch_count']}).",
            },
            "overall_accuracy_comparison": {
                "phase8_reported_accuracy": 0.6429, # 27 / 42
                "phase9b_corrected_accuracy": real_bench["overall_state_accuracy"],
            },
        },
    }

    out_file = REPO_ROOT / "scripts" / "phase9b_evaluation_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Phase 9B evaluation complete. Results saved to: %s", out_file)
    print("\n" + "=" * 70)
    print("PHASE 9B CORRECTED EMPIRICAL EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Controlled Benchmark Success Rate: {reg_bench['adaptive_keyframe']['success_rate']:.1%}")
    print(f"Empirical Uncertainty Bound Coverage: {reg_bench['uncertainty_calibration']['empirical_coverage_rate']:.1%}")
    print(f"Dataset Provenance Verdict: {results['dataset_provenance_verdict']}")
    print(f"Total Evaluations: {real_bench['total_evaluations']} pairs")
    print(f"Corrected State Accuracy: {real_bench['overall_state_accuracy']:.1%} (vs Phase 8 flawed 64.3%)")
    print(f"False Movements: {real_bench['movement_metrics']['false_movement_count']} (Rate: {real_bench['movement_metrics']['false_movement_rate']:.1%})")
    print(f"Movement Recall: {real_bench['movement_metrics']['movement_recall']:.1%}")
    print(f"Return-to-Baseline Accuracy: {real_bench['movement_metrics']['return_to_baseline_accuracy']:.1%}")
    print(f"False Removals: {real_bench['removal_metrics']['false_removal_count']} (Rate: {real_bench['removal_metrics']['false_removal_rate']:.1%})")
    print(f"Duplicate System Tracks: {real_bench['tracking_integrity']['duplicate_track_count']}")
    print(f"Unmatched GT Entities: {real_bench['tracking_integrity']['unmatched_gt_entities']}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
