#!/usr/bin/env python3
"""
Isolated Experiment: Pretrained Official YOLOv8s (COCO) under frozen B3 config
on data/physical_pilot_v2/manifest_v2.json.
"""

from __future__ import annotations

import json
import sys
import time
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
from objectmemory.detection.detector import OpenCVDNNDetector

TARGET_CLASSES: List[str] = [
    "potted plant", "book", "bottle", "scissors", "cell phone", "cup", "banana",
]


def evaluate_detection(manifest, detector: OpenCVDNNDetector, iou_threshold: float = 0.50) -> Dict[str, Any]:
    per_class: Dict[str, Dict[str, Any]] = {
        c: {"gt_visible": 0, "predictions": 0, "tp": 0, "fp": 0, "fn": 0}
        for c in TARGET_CLASSES
    }
    latencies_ms: List[float] = []

    for seq in manifest.sequences:
        for obs in seq.observations:
            img = cv2.imread(str(obs.image_path))
            t0 = time.perf_counter()
            batch = detector.detect(img)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

            dets = [d for d in batch.detections if d.label in per_class]
            for d in dets:
                per_class[d.label]["predictions"] += 1

            visible_gt = [
                obj for obj in obs.objects
                if obj.bounding_box is not None and obj.visibility_status in ("CLEARLY_VISIBLE", "PARTIALLY_OCCLUDED")
            ]
            for obj in visible_gt:
                if obj.class_name in per_class:
                    per_class[obj.class_name]["gt_visible"] += 1

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

            for d_idx, det in enumerate(dets):
                if d_idx not in used_d:
                    per_class[det.label]["fp"] += 1
            for g_idx, obj in enumerate(visible_gt):
                if g_idx not in used_g and obj.class_name in per_class:
                    per_class[obj.class_name]["fn"] += 1

    for c, st in per_class.items():
        gt, tp, preds = st["gt_visible"], st["tp"], st["predictions"]
        st["recall"] = round(tp / gt, 4) if gt > 0 else None
        st["precision"] = round(tp / preds, 4) if preds > 0 else None

    total_gt = sum(v["gt_visible"] for v in per_class.values())
    total_tp = sum(v["tp"] for v in per_class.values())
    total_preds = sum(v["predictions"] for v in per_class.values())
    return {
        "iou_threshold": iou_threshold,
        "confidence_threshold": detector.default_confidence_threshold,
        "total_gt_visible": total_gt,
        "total_predictions": total_preds,
        "total_tp": total_tp,
        "overall_recall": round(total_tp / total_gt, 4) if total_gt > 0 else None,
        "overall_precision": round(total_tp / total_preds, 4) if total_preds > 0 else None,
        "mean_inference_ms": round(float(np.mean(latencies_ms)), 2),
        "p95_inference_ms": round(float(np.percentile(latencies_ms, 95)), 2),
        "per_class": per_class,
    }


def main() -> None:
    weights = Path(__file__).resolve().parent / "models" / "yolov8s.onnx"
    manifest = ManifestLoader.load(PROJECT_ROOT / "data/physical_pilot_v2/manifest_v2.json", check_files=True)
    detector = OpenCVDNNDetector(weights_path=weights)
    runner = BenchmarkRunner(detector=detector, allow_partial=True)
    res = runner.run_config(manifest, BenchmarkConfigFactory.create_config("B3"))
    det_eval = evaluate_detection(manifest, detector, iou_threshold=0.50)

    payload = res.to_dict()
    payload["model_weights"] = str(weights)
    payload["model_size_mb"] = round(weights.stat().st_size / (1024 * 1024), 2)
    payload["detection_evaluation"] = det_eval

    out_json = Path(__file__).resolve().parent / "yolov8s_b3_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    print("Saved:", out_json)
    print("Metrics:", res.metrics)
    print("Detection:", det_eval["total_tp"], "/", det_eval["total_gt_visible"], "recall=", det_eval["overall_recall"], "prec=", det_eval["overall_precision"], "mean_ms=", det_eval["mean_inference_ms"])
    for c, st in det_eval["per_class"].items():
        print(f"  {c}: tp={st['tp']}/{st['gt_visible']} recall={st['recall']} preds={st['predictions']} prec={st['precision']}")


if __name__ == "__main__":
    main()
