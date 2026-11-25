#!/usr/bin/env python3
"""
Per-image, per-GT-object diagnostic for YOLO26m on data/physical_pilot_v2/manifest_v2.json,
plus parity check against official Ultralytics YOLO('models/yolo26m.onnx').predict().
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from objectmemory.baseline.comparator import calculate_bbox_iou
from objectmemory.benchmarks.manifest import ManifestLoader
from objectmemory.detection.registry import DetectorRegistry
from objectmemory.detection.yolo26 import YOLO26mDetector


def main() -> None:
    manifest = ManifestLoader.load(PROJECT_ROOT / "data/physical_pilot_v2/manifest_v2.json", check_files=True)
    det26: YOLO26mDetector = DetectorRegistry.get("yolo26m")
    det26.ensure_loaded()
    ultra_onnx = YOLO(str(PROJECT_ROOT / "models/yolo26m.onnx"), task="detect")

    rows: List[Dict[str, Any]] = []
    parity_checks: List[Dict[str, Any]] = []

    for seq in manifest.sequences:
        for obs in seq.observations:
            img = cv2.imread(str(obs.image_path))
            batch_025 = det26.detect(img, confidence_threshold=0.25)
            batch_005 = det26.detect(img, confidence_threshold=0.05)

            # Parity check on first image of each sequence
            if obs.observation_id.endswith("_V01") or obs.observation_id == "movement_V05":
                ures = ultra_onnx.predict(img, imgsz=640, conf=0.25, iou=0.45, verbose=False)[0]
                u_dets = sorted(
                    [(ultra_onnx.names[int(c)], round(float(cf), 4)) for cf, c in zip(ures.boxes.conf, ures.boxes.cls)],
                    key=lambda x: (-x[1], x[0]),
                )
                o_dets = sorted(
                    [(d.label, round(float(d.confidence), 4)) for d in batch_025.detections],
                    key=lambda x: (-x[1], x[0]),
                )
                parity_checks.append({
                    "observation": obs.observation_id,
                    "exact_match": u_dets == o_dets,
                    "ultralytics_onnx": u_dets,
                    "yolo26m_adapter": o_dets,
                })

            visible_gt = [
                obj for obj in obs.objects
                if obj.bounding_box is not None and obj.visibility_status in ("CLEARLY_VISIBLE", "PARTIALLY_OCCLUDED")
            ]

            for obj in visible_gt:
                # Best same-class >= 0.25
                same_025 = [
                    (calculate_bbox_iou(d.raw_bbox, obj.bounding_box), d)
                    for d in batch_025.detections
                    if d.label == obj.class_name
                ]
                same_025.sort(key=lambda x: x[0], reverse=True)

                # Best any-class >= 0.25
                any_025 = [
                    (calculate_bbox_iou(d.raw_bbox, obj.bounding_box), d)
                    for d in batch_025.detections
                ]
                any_025.sort(key=lambda x: x[0], reverse=True)

                # Best same-class >= 0.05
                same_005 = [
                    (calculate_bbox_iou(d.raw_bbox, obj.bounding_box), d)
                    for d in batch_005.detections
                    if d.label == obj.class_name
                ]
                same_005.sort(key=lambda x: x[0], reverse=True)

                if same_025 and same_025[0][0] >= 0.50:
                    iou, d = same_025[0]
                    status = "DETECTED"
                    pred = d.label
                    conf = round(float(d.confidence), 4)
                    reason = "OK"
                elif any_025 and any_025[0][0] >= 0.50:
                    iou, d = any_025[0]
                    status = "WRONG_CLASS"
                    pred = d.label
                    conf = round(float(d.confidence), 4)
                    reason = f"Misclassified as {d.label}"
                elif same_005 and same_005[0][0] >= 0.50:
                    iou, d = same_005[0]
                    status = "MISSED_LOW_CONF"
                    pred = f"{d.label} (sub-0.25)"
                    conf = round(float(d.confidence), 4)
                    reason = f"Below frozen 0.25 threshold (conf={conf:.4f}, vis={obj.visibility_status})"
                elif same_025 and same_025[0][0] > 0.0:
                    iou, d = same_025[0]
                    status = "MISSED_LOW_IOU"
                    pred = d.label
                    conf = round(float(d.confidence), 4)
                    reason = f"IoU {iou:.3f} < 0.50 (vis={obj.visibility_status})"
                else:
                    iou = 0.0
                    status = "MISSED_NO_CANDIDATE"
                    pred = "NONE"
                    conf = 0.0
                    reason = f"No candidate >= 0.05 (vis={obj.visibility_status})"

                rows.append({
                    "image": f"{seq.challenge_mode}/{Path(obs.image_path).name}",
                    "sequence": seq.challenge_mode,
                    "gt_object": f"{obj.object_id} ({obj.class_name})",
                    "gt_class": obj.class_name,
                    "prediction": pred,
                    "confidence": conf,
                    "iou": round(float(iou), 3),
                    "status": status,
                    "failure_reason": reason,
                })

    summary = {}
    for r in rows:
        summary[r["status"]] = summary.get(r["status"], 0) + 1

    print("PARITY_ALL_EXACT:", all(p["exact_match"] for p in parity_checks))
    for p in parity_checks:
        if not p["exact_match"]:
            print("PARITY_DIFF:", p)
    print("DIAGNOSTIC_SUMMARY:", summary)
    for r in rows:
        if r["status"] != "DETECTED":
            print(f"  {r['image']} | {r['gt_object']} | pred={r['prediction']} conf={r['confidence']} iou={r['iou']} | {r['status']} | {r['failure_reason']}")

    out_path = PROJECT_ROOT / "research_validation/yolo26m_per_image_diagnostic.json"
    out_path.write_text(json.dumps({"parity_checks": parity_checks, "summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
