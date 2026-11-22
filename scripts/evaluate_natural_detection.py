#!/usr/bin/env python3
"""
Evaluates YOLOv8n (OpenCV DNN CPU) and YOLO-World (ONNX Runtime CPU) on the
reproducible 20-image COCO 2017 validation subset.

Uses Roboflow Supervision MeanAveragePrecision and IoU bipartite evaluation.
Reports:
- Precision, Recall, F1 score
- False Positives (FP) and False Negatives (FN)
- mAP@50 and mAP@50:95 via Supervision
- Latency per frame (mean, median, p95)
- Hardware metadata (CPU, RAM, OS, execution providers)
"""

import argparse
import json
import os
import platform
import sys
import time
from typing import Any, Dict, List, Set, Tuple
import cv2
import numpy as np
import psutil

# Ensure src is on python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from objectmemory.detection.detector import OpenCVDNNDetector
from objectmemory.detection.models import BoundingBox, Detection, DetectionBatch, DetectorConfiguration
from objectmemory.detection.registry import DetectorRegistry
from objectmemory.detection.supervision_adapter import (
    evaluate_detection_map,
    to_supervision_detections,
)
from objectmemory.detection.yolo_world import YOLOWorldDetector

TARGET_CLASSES = [
    "cup",
    "bottle",
    "book",
    "laptop",
    "scissors",
    "potted plant",
    "chair",
    "cell phone",
]


def compute_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH

    boxAArea = max(0.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(0.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    denom = boxAArea + boxBArea - interArea
    if denom <= 0:
        return 0.0
    return interArea / denom


def evaluate_image_matches_and_counts(
    pred_dets: List[Detection],
    gt_anns: List[Dict[str, Any]],
    target_classes: List[str],
    iou_threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Computes per-class and aggregate detection match metrics for an image frame.

    Guarantees:
    - Distinguishes ground-truth instances, predicted instances, true positives,
      false positives, and false negatives.
    - Each prediction matches at most one ground-truth box of matching class (IoU >= threshold).
    - Unmatched predictions are counted as false positives (never as true positives).
    - No ground-truth object is matched more than once.
    - Duplicate predictions for the same ground truth box count as false positives.
    """
    per_class: Dict[str, Dict[str, int]] = {
        c: {
            "gt_instances": 0,
            "predicted_instances": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }
        for c in target_classes
    }

    # Count ground-truth instances per target class
    for gt in gt_anns:
        cname = gt["category_name"]
        if cname in per_class:
            per_class[cname]["gt_instances"] += 1

    # Count predicted instances per target class
    for p in pred_dets:
        if p.label in per_class:
            per_class[p.label]["predicted_instances"] += 1

    matched_gt: Set[int] = set()
    sorted_preds = sorted(pred_dets, key=lambda d: d.confidence, reverse=True)

    for p in sorted_preds:
        if p.label not in per_class:
            continue

        p_box = (p.raw_bbox.x_min, p.raw_bbox.y_min, p.raw_bbox.x_max, p.raw_bbox.y_max)
        best_iou = 0.0
        best_gt_idx = -1

        for idx, gt in enumerate(gt_anns):
            if idx in matched_gt:
                continue
            if gt["category_name"] != p.label:
                continue
            gt_box = gt["xyxy"]
            iou = compute_iou(p_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx

        if best_iou >= iou_threshold and best_gt_idx != -1:
            per_class[p.label]["tp"] += 1
            matched_gt.add(best_gt_idx)
        else:
            per_class[p.label]["fp"] += 1

    # Unmatched ground-truth instances are false negatives
    for c in target_classes:
        per_class[c]["fn"] = per_class[c]["gt_instances"] - per_class[c]["tp"]

    img_tp = sum(per_class[c]["tp"] for c in target_classes)
    img_fp = sum(per_class[c]["fp"] for c in target_classes)
    img_fn = sum(per_class[c]["fn"] for c in target_classes)

    return {
        "per_class": per_class,
        "total_tp": img_tp,
        "total_fp": img_fp,
        "total_fn": img_fn,
    }


def match_detections_for_image(
    pred_dets: List[Detection],
    gt_anns: List[Dict[str, Any]],
    iou_threshold: float = 0.50,
) -> Tuple[int, int, int]:
    """
    Backwards-compatible wrapper returning (tp, fp, fn).
    """
    res = evaluate_image_matches_and_counts(pred_dets, gt_anns, TARGET_CLASSES, iou_threshold)
    return res["total_tp"], res["total_fp"], res["total_fn"]


def get_hardware_info() -> Dict[str, Any]:
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "python_version": platform.python_version(),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO models on COCO 2017 subset")
    parser.add_argument(
        "--coco-dir",
        default="data/coco_val2017_subset",
        help="Path to COCO validation subset directory",
    )
    parser.add_argument(
        "--output-file",
        default="scripts/phase14b_coco_subset_evaluation_results.json",
        help="Path to save evaluation JSON results",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="Confidence threshold for inference",
    )
    parser.add_argument(
        "--compare-preprocessing",
        action="store_true",
        default=True,
        help="Compare both letterboxed and direct-stretch preprocessing",
    )
    args = parser.parse_args()

    coco_dir = os.path.abspath(args.coco_dir)
    ann_file = os.path.join(coco_dir, "annotations", "instances_val2017_subset.json")
    img_dir = os.path.join(coco_dir, "images")

    if not os.path.exists(ann_file):
        print(f"ERROR: Annotations file not found: {ann_file}", file=sys.stderr)
        sys.exit(1)

    with open(ann_file, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in coco_data["categories"]}

    # Group annotations by image_id
    anns_by_image: Dict[int, List[Dict[str, Any]]] = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        x, y, w, h = ann["bbox"]
        cname = cat_id_to_name[ann["category_id"]]
        if cname in TARGET_CLASSES:
            anns_by_image.setdefault(img_id, []).append({
                "id": ann["id"],
                "category_name": cname,
                "xyxy": (float(x), float(y), float(x + w), float(y + h)),
            })

    # Prepare ground truth DetectionBatch per image for Supervision mAP
    target_batches: List[DetectionBatch] = []
    image_paths: List[Tuple[int, str]] = []

    for img_info in coco_data["images"]:
        img_id = img_info["id"]
        img_path = os.path.join(img_dir, img_info["file_name"])
        image_paths.append((img_id, img_path))
        gt_dets = [
            Detection(
                label=ann["category_name"],
                confidence=1.0,
                raw_bbox=BoundingBox(ann["xyxy"][0], ann["xyxy"][1], ann["xyxy"][2], ann["xyxy"][3]),
            )
            for ann in anns_by_image.get(img_id, [])
        ]
        target_batches.append(
            DetectionBatch(
                detections=gt_dets,
                image_size=(img_info["width"], img_info["height"]),
                latency_ms=0.0,
                detector_name="ground_truth",
            )
        )

    print(f"Loaded {len(image_paths)} images with {sum(len(b.detections) for b in target_batches)} ground-truth annotations across {len(TARGET_CLASSES)} target classes.")

    if args.compare_preprocessing:
        detectors_to_eval = [
            ("yolov8n_letterboxed", "OpenCV DNN (CPU) - Letterbox Preprocessing", "yolov8n", True, False),
            ("yolov8n_direct_stretch", "OpenCV DNN (CPU) - Direct Stretch Resize (Legacy)", "yolov8n", False, False),
            ("yolo_world_letterboxed", "ONNX Runtime (CPU) - Letterbox Preprocessing", "yolo-world", True, True),
            ("yolo_world_direct_stretch", "ONNX Runtime (CPU) - Direct Stretch Resize (Legacy)", "yolo-world", False, True),
        ]
    else:
        detectors_to_eval = [
            ("yolov8n", "OpenCV DNN (CPU) - Letterbox Preprocessing", "yolov8n", True, False),
            ("yolo-world", "ONNX Runtime (CPU) - Letterbox Preprocessing", "yolo-world", True, True),
        ]

    results_by_model: Dict[str, Any] = {}

    for model_name, backend_desc, model_type, use_letterbox, is_world in detectors_to_eval:
        print(f"\n=======================================================")
        print(f"Evaluating {model_name} ({backend_desc})...")
        if model_type == "yolov8n":
            detector = OpenCVDNNDetector(
                weights_path="models/yolov8n.onnx",
                framework="onnx",
                architecture="yolov8",
                letterbox=use_letterbox,
            )
        else:
            detector = YOLOWorldDetector(
                weights_path="models/yolov8s-worldv2.onnx",
                letterbox=use_letterbox,
            )

        pred_batches: List[DetectionBatch] = []
        latencies: List[float] = []
        total_tp = 0
        total_fp = 0
        total_fn = 0

        total_per_class: Dict[str, Dict[str, Any]] = {
            c: {
                "gt_instances": 0,
                "predicted_instances": 0,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
            }
            for c in TARGET_CLASSES
        }

        # Warm-up pass to eliminate initial tokenization and runtime JIT overhead from timed metrics
        warmup_img = cv2.imread(image_paths[0][1])
        if is_world:
            detector.detect(
                warmup_img,
                config=DetectorConfiguration(
                    confidence_threshold=args.conf_threshold,
                    target_classes=TARGET_CLASSES,
                ),
            )
        else:
            detector.detect(warmup_img, confidence_threshold=args.conf_threshold)

        for (img_id, img_path), gt_batch in zip(image_paths, target_batches):
            img = cv2.imread(img_path)
            t0 = time.time()
            if is_world:
                config = DetectorConfiguration(
                    confidence_threshold=args.conf_threshold,
                    target_classes=TARGET_CLASSES,
                )
                batch = detector.detect(img, config=config)
            else:
                batch = detector.detect(img, confidence_threshold=args.conf_threshold)
            latency = (time.time() - t0) * 1000.0
            latencies.append(latency)

            # Filter predictions to target classes
            filtered_preds = [d for d in batch.detections if d.label in TARGET_CLASSES]
            pred_batches.append(
                DetectionBatch(
                    detections=filtered_preds,
                    image_size=batch.image_size,
                    latency_ms=latency,
                    detector_name=model_name,
                )
            )

            # Match for precision / recall and per-class breakdown at IoU=0.50
            gt_anns = anns_by_image.get(img_id, [])
            match_res = evaluate_image_matches_and_counts(
                filtered_preds, gt_anns, TARGET_CLASSES, iou_threshold=0.50
            )

            for c in TARGET_CLASSES:
                c_dict = match_res["per_class"][c]
                total_per_class[c]["gt_instances"] += c_dict["gt_instances"]
                total_per_class[c]["predicted_instances"] += c_dict["predicted_instances"]
                total_per_class[c]["tp"] += c_dict["tp"]
                total_per_class[c]["fp"] += c_dict["fp"]
                total_per_class[c]["fn"] += c_dict["fn"]

            total_tp += match_res["total_tp"]
            total_fp += match_res["total_fp"]
            total_fn += match_res["total_fn"]

        # Compute per-class precision, recall, and f1
        for c in TARGET_CLASSES:
            tp_c = total_per_class[c]["tp"]
            fp_c = total_per_class[c]["fp"]
            fn_c = total_per_class[c]["fn"]
            prec_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
            rec_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
            f1_c = (2 * prec_c * rec_c) / (prec_c + rec_c) if (prec_c + rec_c) > 0 else 0.0
            total_per_class[c]["precision"] = round(prec_c, 4)
            total_per_class[c]["recall"] = round(rec_c, 4)
            total_per_class[c]["f1_score"] = round(f1_c, 4)

        # Supervision mAP computation
        map_res = evaluate_detection_map(pred_batches, target_batches)

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"  Overall TP: {total_tp} | FP: {total_fp} | FN: {total_fn}")
        print(f"  Overall Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        print(f"  mAP@50 (Supervision):    {map_res['map50']:.4f}")
        print(f"  mAP@50:95 (Supervision): {map_res['map50_95']:.4f}")
        print(f"  Latency (ms): Mean={np.mean(latencies):.1f} | Median={np.median(latencies):.1f} | p95={np.percentile(latencies, 95):.1f}")
        print("\n  Per-Class Breakdown (IoU=0.50):")
        print(f"  {'Class':<14} {'GT':>4} {'Pred':>5} {'TP':>4} {'FP':>4} {'FN':>4} {'Precision':>10} {'Recall':>8} {'F1':>8}")
        print("  " + "-" * 65)
        for c in TARGET_CLASSES:
            cd = total_per_class[c]
            print(
                f"  {c:<14} {cd['gt_instances']:>4} {cd['predicted_instances']:>5} "
                f"{cd['tp']:>4} {cd['fp']:>4} {cd['fn']:>4} "
                f"{cd['precision']:>10.4f} {cd['recall']:>8.4f} {cd['f1_score']:>8.4f}"
            )

        results_by_model[model_name] = {
            "model_name": model_name,
            "backend": backend_desc,
            "confidence_threshold": args.conf_threshold,
            "target_classes": TARGET_CLASSES,
            "total_ground_truth": sum(total_per_class[c]["gt_instances"] for c in TARGET_CLASSES),
            "total_predictions": sum(total_per_class[c]["predicted_instances"] for c in TARGET_CLASSES),
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "map50": round(map_res["map50"], 4),
            "map50_95": round(map_res["map50_95"], 4),
            "latency_ms": {
                "mean": round(float(np.mean(latencies)), 2),
                "median": round(float(np.median(latencies)), 2),
                "p95": round(float(np.percentile(latencies, 95)), 2),
                "min": round(float(np.min(latencies)), 2),
                "max": round(float(np.max(latencies)), 2),
            },
            "per_class_counts": total_per_class,
        }

    output_payload = {
        "benchmark_name": "COCO 2017 Validation Subset (20 Images, 8 Classes)",
        "track": "TRACK_A_DETECTOR_EVALUATION",
        "track_description": "Evaluation of actual detector predictions on natural images with verified ground-truth bounding boxes. No oracle boxes used.",
        "evaluation_version": "14b",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": "Official COCO 2017 Validation Set (http://cocodataset.org)",
        "license": "Creative Commons Attribution 4.0 License (CC-BY 4.0)",
        "evaluation_history": [
            {
                "version": "10.0",
                "timestamp": "2026-10-21T04:00:59Z",
                "notes": "Initial Phase 10 evaluation (omitted per-class counts; un-warmed latency)."
            },
            {
                "version": "10a",
                "timestamp": "2026-10-21T04:24:56Z",
                "notes": "Phase 10A corrected evaluation: added documented per-class detection counts, warmup pass."
            },
            {
                "version": "14b",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "notes": "Phase 14B Track A evaluation: added side-by-side comparison of aspect-preserving letterboxing vs direct stretch."
            }
        ],
        "target_classes": TARGET_CLASSES,
        "total_images": len(image_paths),
        "total_ground_truth_objects": sum(len(b.detections) for b in target_batches),
        "hardware": get_hardware_info(),
        "models": results_by_model,
    }

    out_path = os.path.abspath(args.output_file)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\nTrack A Natural-image benchmark evaluation complete! Results saved to: {out_path}")


if __name__ == "__main__":
    main()
