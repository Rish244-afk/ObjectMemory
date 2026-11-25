#!/usr/bin/env python3
"""
Side-by-side Physical-Pilot Evaluation: YOLOv8m (baseline) vs YOLO26m
on data/physical_pilot_v2/manifest_v2.json (30 photographs, 6 scenarios).

Preserves frozen B3 thresholds (conf=0.25, nms=0.45, iou=0.50) and separates
detector-only metrics from complete ObjectMemory temporal-memory B3 metrics.
"""

from __future__ import annotations

import json
import os
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from objectmemory.baseline.comparator import calculate_bbox_iou
from objectmemory.benchmarks.config_factory import BenchmarkConfigFactory
from objectmemory.benchmarks.manifest import ManifestLoader
from objectmemory.benchmarks.runner import BenchmarkRunner
from objectmemory.detection.registry import DetectorRegistry

TARGET_CLASSES: List[str] = [
    "potted plant", "book", "bottle", "scissors", "cell phone", "cup", "banana",
]


def _get_rss_mb() -> float | None:
    try:
        import psutil
        return round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 2)
    except Exception:
        return None


def evaluate_detector_detailed(manifest: Any, detector: Any, iou_threshold: float = 0.50) -> Dict[str, Any]:
    per_class: Dict[str, Dict[str, Any]] = {
        c: {"gt_visible": 0, "predictions": 0, "tp": 0, "fp": 0, "fn": 0}
        for c in TARGET_CLASSES
    }
    latencies_ms: List[float] = []
    all_confs: List[float] = []
    target_confs: List[float] = []
    missed_details: List[Dict[str, Any]] = []
    incorrect_class_details: List[Dict[str, Any]] = []
    duplicate_count = 0
    non_target_fp = 0

    for seq in manifest.sequences:
        for obs in seq.observations:
            img = cv2.imread(str(obs.image_path))
            t0 = time.perf_counter()
            batch = detector.detect(img, confidence_threshold=0.25)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

            for d in batch.detections:
                all_confs.append(float(d.confidence))

            dets = [d for d in batch.detections if d.label in per_class]
            non_target_dets = [d for d in batch.detections if d.label not in per_class]
            non_target_fp += len(non_target_dets)

            for d in dets:
                per_class[d.label]["predictions"] += 1
                target_confs.append(float(d.confidence))

            visible_gt = [
                obj for obj in obs.objects
                if obj.bounding_box is not None and obj.visibility_status in ("CLEARLY_VISIBLE", "PARTIALLY_OCCLUDED")
            ]
            for obj in visible_gt:
                if obj.class_name in per_class:
                    per_class[obj.class_name]["gt_visible"] += 1

            # Same-class matching
            candidates = []
            for d_idx, det in enumerate(dets):
                for g_idx, obj in enumerate(visible_gt):
                    if det.label == obj.class_name:
                        iou = calculate_bbox_iou(det.raw_bbox, obj.bounding_box)
                        if iou >= iou_threshold:
                            candidates.append((iou, d_idx, g_idx))
            candidates.sort(key=lambda x: x[0], reverse=True)

            used_d, used_g = set(), set()
            for _, d_idx, g_idx in candidates:
                if d_idx not in used_d and g_idx not in used_g:
                    used_d.add(d_idx)
                    used_g.add(g_idx)
                    per_class[dets[d_idx].label]["tp"] += 1

            # Check duplicate detections (unmatched same-class dets with IoU >= threshold against an already-matched GT)
            for d_idx, det in enumerate(dets):
                if d_idx in used_d:
                    continue
                is_dup = False
                for g_idx, obj in enumerate(visible_gt):
                    if det.label == obj.class_name and calculate_bbox_iou(det.raw_bbox, obj.bounding_box) >= iou_threshold:
                        is_dup = True
                        break
                if is_dup:
                    duplicate_count += 1
                per_class[det.label]["fp"] += 1

            # Check missed GT & wrong-class predictions overlapping missed GT
            for g_idx, obj in enumerate(visible_gt):
                if g_idx in used_g or obj.class_name not in per_class:
                    continue
                per_class[obj.class_name]["fn"] += 1
                best_other = None
                best_iou = 0.0
                for det in batch.detections:
                    iou = calculate_bbox_iou(det.raw_bbox, obj.bounding_box)
                    if iou >= iou_threshold and iou > best_iou:
                        best_iou = iou
                        best_other = det
                if best_other is not None:
                    incorrect_class_details.append({
                        "observation": obs.observation_id,
                        "gt_class": obj.class_name,
                        "predicted_class": best_other.label,
                        "confidence": round(float(best_other.confidence), 4),
                        "iou": round(float(best_iou), 3),
                    })
                else:
                    missed_details.append({
                        "observation": obs.observation_id,
                        "gt_class": obj.class_name,
                        "visibility": obj.visibility_status,
                    })

    for c, st in per_class.items():
        gt, tp, preds = st["gt_visible"], st["tp"], st["predictions"]
        st["recall"] = round(tp / gt, 4) if gt > 0 else None
        st["precision"] = round(tp / preds, 4) if preds > 0 else None

    total_gt = sum(v["gt_visible"] for v in per_class.values())
    total_tp = sum(v["tp"] for v in per_class.values())
    total_preds = sum(v["predictions"] for v in per_class.values())
    conf_arr = np.array(target_confs, dtype=float) if target_confs else np.array([0.0])

    return {
        "iou_threshold": iou_threshold,
        "confidence_threshold": 0.25,
        "total_gt_visible": total_gt,
        "total_target_predictions": total_preds,
        "non_target_predictions": non_target_fp,
        "total_tp": total_tp,
        "total_fp_target": total_preds - total_tp,
        "total_fn": total_gt - total_tp,
        "overall_recall": round(total_tp / total_gt, 4) if total_gt > 0 else None,
        "overall_precision": round(total_tp / total_preds, 4) if total_preds > 0 else None,
        "duplicate_detections": duplicate_count,
        "incorrect_class_predictions_count": len(incorrect_class_details),
        "incorrect_class_predictions": incorrect_class_details,
        "missed_without_overlap_count": len(missed_details),
        "missed_without_overlap": missed_details,
        "confidence_distribution_target": {
            "count": len(target_confs),
            "min": round(float(np.min(conf_arr)), 4),
            "median": round(float(np.median(conf_arr)), 4),
            "mean": round(float(np.mean(conf_arr)), 4),
            "p90": round(float(np.percentile(conf_arr, 90)), 4),
            "max": round(float(np.max(conf_arr)), 4),
        },
        "mean_inference_ms": round(float(np.mean(latencies_ms)), 2),
        "p95_inference_ms": round(float(np.percentile(latencies_ms, 95)), 2),
        "per_class": per_class,
    }


def evaluate_model(model_name: str, manifest: Any) -> Dict[str, Any]:
    rss_before = _get_rss_mb()
    tracemalloc.start()
    detector = DetectorRegistry.get(model_name)
    if hasattr(detector, "ensure_loaded"):
        detector.ensure_loaded()
    # Warmup 1 frame
    first_img = cv2.imread(str(manifest.sequences[0].observations[0].image_path))
    _ = detector.detect(first_img, confidence_threshold=0.25)

    det_eval = evaluate_detector_detailed(manifest, detector, iou_threshold=0.50)

    runner = BenchmarkRunner(detector=detector, allow_partial=True)
    b3_res = runner.run_config(manifest, BenchmarkConfigFactory.create_config("B3"))
    _, peak_py_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = _get_rss_mb()

    b3_dict = b3_res.to_dict()
    per_seq = b3_dict.get("sequence_reports", {})

    return {
        "detector_name": model_name,
        "model_info": detector.model_info(),
        "memory": {
            "rss_before_mb": rss_before,
            "rss_after_mb": rss_after,
            "tracemalloc_peak_mb": round(peak_py_bytes / (1024 * 1024), 2),
        },
        "detector_only": det_eval,
        "objectmemory_b3": {
            "metrics": b3_res.metrics,
            "per_sequence": per_seq,
        },
    }


def main() -> None:
    manifest = ManifestLoader.load(PROJECT_ROOT / "data/physical_pilot_v2/manifest_v2.json", check_files=True)
    results = {
        "dataset": "data/physical_pilot_v2/manifest_v2.json",
        "frozen_config": "B3 (conf=0.25, nms=0.45, iou=0.50)",
        "models": {},
    }
    for name in ("yolov8m", "yolo26m"):
        print(f"=== Evaluating {name} ===")
        res = evaluate_model(name, manifest)
        results["models"][name] = res
        d = res["detector_only"]
        b3 = res["objectmemory_b3"]["metrics"]
        print(
            f"  Detector-only: TP={d['total_tp']}/{d['total_gt_visible']} "
            f"recall={d['overall_recall']} prec={d['overall_precision']} "
            f"wrong_cls={d['incorrect_class_predictions_count']} dups={d['duplicate_detections']} "
            f"mean_ms={d['mean_inference_ms']}"
        )
        print(f"  ObjectMemory B3: {b3}")

    out_path = PROJECT_ROOT / "research_validation/yolo26m_vs_yolov8m_pilot_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
