import cv2
import numpy as np
from pathlib import Path
from objectmemory.detection.detector import OpenCVDNNDetector

img_path = Path("data/workspaces/objectmemory_real_test/images/visit_01.png")
if not img_path.exists():
    print(f"File not found: {img_path}")
    exit(1)

img = cv2.imread(str(img_path))
print(f"Image loaded: {img is not None}, shape={img.shape if img is not None else None}")

det = OpenCVDNNDetector(weights_path="models/yolov8n.onnx", framework="onnx", architecture="yolov8")
print(f"Detector available: {det.is_available()}")

blob = cv2.dnn.blobFromImage(
    img,
    scalefactor=1.0 / 255.0,
    size=(640, 640),
    mean=(0.0, 0.0, 0.0),
    swapRB=True,
    crop=False,
)
det._net.setInput(blob)
out = det._net.forward()
print(f"Raw output ndim: {out.ndim}, shape: {out.shape}, dtype: {out.dtype}")
print(f"Min: {out.min():.4f}, Max: {out.max():.4f}, Mean: {out.mean():.4f}")

# Check predictions breakdown
preds = out[0].T  # Shape: (8400, 84)
print(f"Transposed preds shape: {preds.shape}")

# Look at bounding box coordinates in preds[:, :4]
boxes_xywh = preds[:, :4]
print(f"Boxes min={boxes_xywh.min(axis=0)}, max={boxes_xywh.max(axis=0)}, mean={boxes_xywh.mean(axis=0)}")

# Look at class scores
scores = preds[:, 4:]  # Shape: (8400, 80)
print(f"Scores min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")

# Top 10 max scores overall
flat_scores = scores.flatten()
top_indices = np.argsort(flat_scores)[-10:][::-1]
print("\n--- TOP 10 CLASS SCORES OVERALL ---")
for idx in top_indices:
    row = idx // 80
    col = idx % 80
    score = flat_scores[idx]
    class_name = det.classes[col] if col < len(det.classes) else f"class_{col}"
    box = preds[row, :4]
    print(f"Score: {score:.4f} | Class: {class_name} (id {col}) | Anchor {row} | Box (cx,cy,w,h): {box.round(1)}")

print("\n--- DETECTION TESTS WITH THRESHOLDS ---")
for conf in [0.35, 0.25, 0.15, 0.10, 0.05, 0.01]:
    batch = det.detect(img, confidence_threshold=conf)
    summary = [f"{d.label}: {d.confidence:.3f} bbox={d.raw_bbox.as_tuple()}" for d in batch.detections]
    print(f"Conf {conf:.2f}: {len(batch.detections)} detections -> {summary[:5]}")
