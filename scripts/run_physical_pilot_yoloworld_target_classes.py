#!/usr/bin/env python3
"""
Isolated Experiment: YOLOWorldDetector with explicit 7 physical-pilot target classes
on data/physical_pilot_v2/manifest_v2.json using frozen B3 configuration.

Does not modify src/objectmemory/, thresholds, model weights, manifest, or prior results.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from objectmemory.baseline.comparator import calculate_bbox_iou
from objectmemory.benchmarks.config_factory import BenchmarkConfigFactory
from objectmemory.benchmarks.manifest import ManifestLoader
from objectmemory.benchmarks.runner import BenchmarkRunner
from objectmemory.detection.yolo_world import YOLOWorldDetector

TARGET_CLASSES: List[str] = [
    "potted plant",
    "book",
    "bottle",
    "scissors",
    "cell phone",
    "cup",
    "banana",
]


def compute_detection_recall(manifest, detector: YOLOWorldDetector, iou_threshold: float = 0.50) -> Dict[str, Any]:
    """Computes per-class detection counts and recall (IoU >= 0.50) against manifest annotations."""
    per_class: Dict[str, Dict[str, int]] = {
        c: {"gt_visible": 0, "predictions": 0, "tp": 0, "fp": 0, "fn": 0}
        for c in TARGET_CLASSES
    }

    for seq in manifest.sequences:
        for obs in seq.observations:
            img = cv2.imread(str(obs.image_path))
            batch = detector.detect(img)

            for d in batch.detections:
                if d.label in per_class:
                    per_class[d.label]["predictions"] += 1

            visible_gt = [
                obj for obj in obs.objects
                if obj.bounding_box is not None and obj.visibility_status in ("CLEARLY_VISIBLE", "PARTIALLY_OCCLUDED")
            ]
            for obj in visible_gt:
                if obj.class_name in per_class:
                    per_class[obj.class_name]["gt_visible"] += 1

            # Greedy 1-to-1 class-matched IoU matching
            candidates = []
            for d_idx, det in enumerate(batch.detections):
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
                    cls_name = batch.detections[d_idx].label
                    if cls_name in per_class:
                        per_class[cls_name]["tp"] += 1

            for d_idx, det in enumerate(batch.detections):
                if d_idx not in used_d and det.label in per_class:
                    per_class[det.label]["fp"] += 1
            for g_idx, obj in enumerate(visible_gt):
                if g_idx not in used_g and obj.class_name in per_class:
                    per_class[obj.class_name]["fn"] += 1

    for c, stats in per_class.items():
        gt = stats["gt_visible"]
        tp = stats["tp"]
        preds = stats["predictions"]
        stats["recall"] = round(tp / gt, 4) if gt > 0 else None
        stats["precision"] = round(tp / preds, 4) if preds > 0 else None

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
        "per_class": per_class,
    }


def main() -> None:
    manifest_path = PROJECT_ROOT / "data/physical_pilot_v2/manifest_v2.json"
    manifest = ManifestLoader.load(manifest_path, check_files=True)

    detector = YOLOWorldDetector(default_prompts=TARGET_CLASSES)
    b3_config = BenchmarkConfigFactory.create_config("B3")
    runner = BenchmarkRunner(detector=detector, allow_partial=True)

    res = runner.run_config(manifest, b3_config)
    det_eval = compute_detection_recall(manifest, detector, iou_threshold=0.50)

    payload = res.to_dict()
    payload["explicit_target_classes"] = TARGET_CLASSES
    payload["detection_evaluation"] = det_eval

    out_json = PROJECT_ROOT / "research_validation/physical_pilot_v2_yoloworld_target_classes_results.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    print("Saved:", out_json)
    print("Metrics:", res.metrics)
    print("Detection overall:", det_eval["total_tp"], "/", det_eval["total_gt_visible"], "recall=", det_eval["overall_recall"], "preds=", det_eval["total_predictions"])
    for c, st in det_eval["per_class"].items():
        print(f"  {c}: preds={st['predictions']} tp={st['tp']}/{st['gt_visible']} recall={st['recall']} prec={st['precision']}")


if __name__ == "__main__":
    main()
