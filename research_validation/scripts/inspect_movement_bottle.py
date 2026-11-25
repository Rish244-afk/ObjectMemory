import json
from pathlib import Path
import sys
import cv2
import numpy as np
sys.path.insert(0, "src")

from objectmemory.detection.detector import OpenCVDNNDetector

m = json.load(open("data/physical_pilot_v2/manifest_v2.json", encoding="utf-8"))
mov_seq = next(s for s in m["sequences"] if s["sequence_id"] == "physical_pilot_v2_movement")

det_prod = OpenCVDNNDetector(weights_path="models/yolov8m.onnx", nms_threshold=0.45)
det_low = OpenCVDNNDetector(weights_path="models/yolov8m.onnx", nms_threshold=0.45)

def iou(b1, b2):
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / max(1e-6, a1 + a2 - inter)

for idx, obs in enumerate(mov_seq["observations"], start=1):
    img_p = Path("data/physical_pilot_v2/movement") / f"V0{idx}.jpeg"
    img = cv2.imread(str(img_p))
    gt_bottle = next((o["bounding_box"] for o in obs["objects"] if o["object_id"] == "bottle_01"), None)
    gb = gt_bottle

    res_prod = det_prod.detect(img, confidence_threshold=0.25)
    res_low = det_low.detect(img, confidence_threshold=0.01)

    print(f"\n=== {obs['observation_id']} ({img_p.name}) | GT bottle_01={gb} ===")
    print(f"  Prod (conf>=0.25) [{len(res_prod.detections)}]:")
    for d in res_prod.detections:
        b = [round(d.raw_bbox.x_min,1), round(d.raw_bbox.y_min,1), round(d.raw_bbox.x_max,1), round(d.raw_bbox.y_max,1)]
        ov = iou(b, gb) if gb else 0.0
        print(f"    label={d.label} conf={d.confidence:.4f} bbox={b} iou_with_gt_bottle={ov:.3f}")

    # Check low-conf candidates overlapping GT bottle (IoU >= 0.1)
    cands = []
    for d in res_low.detections:
        b = [round(d.raw_bbox.x_min,1), round(d.raw_bbox.y_min,1), round(d.raw_bbox.x_max,1), round(d.raw_bbox.y_max,1)]
        ov = iou(b, gb) if gb else 0.0
        if ov >= 0.10:
            cands.append((d.label, round(d.confidence, 4), b, round(ov, 3)))
    cands.sort(key=lambda x: x[1], reverse=True)
    print(f"  Low-conf (conf>=0.01) overlapping GT bottle_01 (IoU>=0.10) [{len(cands)}]: {cands[:8]}")
