"""
Diagnostic script for inspecting object detection and visual signature extraction.

Usage:
    python tools/diagnose_detection.py --demo [--out demo_detections.png]
    python tools/diagnose_detection.py --image path/to/image.jpg [--weights path/to/model.onnx]
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
import cv2
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root))

from objectmemory.detection.detector import MockObjectDetector, OpenCVDNNDetector
from objectmemory.detection.models import Detection
from objectmemory.detection.pipeline import DetectionPipeline
from objectmemory.detection.signature import VisualSignatureExtractor
from objectmemory.domain.models import BoundingBox


def run_synthetic_detection_demo(out_path: Optional[str] = None) -> None:
    """Demonstrates detection, signature extraction, and coordinate mapping on synthetic bench."""
    print("=" * 70)
    print(" ObjectMemory: Diagnostic Detection & Signature Demo ")
    print("=" * 70)

    from tests.test_registration import _generate_synthetic_workbench
    bench_img = _generate_synthetic_workbench(640, 480, seed=123)

    # Synthetic detections corresponding to workbench items drawn in synthetic generator
    mock_detections = [
        Detection(
            label="digital_multimeter",
            confidence=0.91,
            raw_bbox=BoundingBox(x_min=260.0, y_min=60.0, x_max=360.0, y_max=240.0),
        ),
        Detection(
            label="circuit_board",
            confidence=0.86,
            raw_bbox=BoundingBox(x_min=50.0, y_min=50.0, x_max=220.0, y_max=180.0),
        ),
        Detection(
            label="solder_point",
            confidence=0.72,
            raw_bbox=BoundingBox(x_min=100.0, y_min=70.0, x_max=120.0, y_max=90.0),
        ),
    ]

    detector = MockObjectDetector(mock_detections, latency_simulation_ms=12.5)
    pipeline = DetectionPipeline(detector, VisualSignatureExtractor())

    # Simulate query frame shifted by (+15, +10)
    H_shift = np.array([[1.0, 0.0, -15.0], [0.0, 1.0, -10.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    start = time.perf_counter()
    batch = pipeline.process_frame(
        bench_img,
        homography=H_shift,
        ref_size=(640, 480),
        confidence_threshold=0.5,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    print(f"\nDetector Name:        {batch.detector_name}")
    print(f"Is Synthetic Fixture: {batch.is_synthetic}")
    print(f"Total Detections:     {len(batch.detections)}")
    print(f"Pipeline Latency:     {elapsed_ms:.2f} ms\n")

    vis_img = bench_img.copy()

    for i, det in enumerate(batch.detections):
        rb = det.raw_bbox
        cb = det.canonical_bbox
        print(f"[{i+1}] Label: '{det.label}' | Confidence: {det.confidence:.2%}")
        print(f"    Raw BBox:       ({rb.x_min:.1f}, {rb.y_min:.1f}) -> ({rb.x_max:.1f}, {rb.y_max:.1f})")
        if cb:
            print(f"    Canonical BBox: ({cb.x_min:.1f}, {cb.y_min:.1f}) -> ({cb.x_max:.1f}, {cb.y_max:.1f})")
        if det.signature:
            print(f"    Signature Norm: {np.linalg.norm(det.signature.combined_vector):.4f} (Dim: {len(det.signature.combined_vector)})")

        # Draw raw box in Cyan
        cv2.rectangle(vis_img, (int(rb.x_min), int(rb.y_min)), (int(rb.x_max), int(rb.y_max)), (255, 255, 0), 2)
        cv2.putText(
            vis_img,
            f"{det.label} {det.confidence:.0%}",
            (int(rb.x_min), max(15, int(rb.y_min) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
        )

        # Draw canonical box in Magenta
        if cb:
            cv2.rectangle(vis_img, (int(cb.x_min), int(cb.y_min)), (int(cb.x_max), int(cb.y_max)), (255, 0, 255), 1)

    if out_path:
        cv2.imwrite(out_path, vis_img)
        print(f"\n[Saved] Visualized detections written to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ObjectMemory Detection Diagnostics")
    parser.add_argument("--demo", action="store_true", help="Run synthetic detection demo")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--weights", type=str, default=None, help="Path to ONNX weights")
    parser.add_argument("--out", type=str, default="demo_detections.png", help="Output visualization path")
    args = parser.parse_args()

    if args.demo or not args.image:
        run_synthetic_detection_demo(out_path=args.out)
        return

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"Error: Image not found at {img_path}")
        sys.exit(1)

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Error: Could not decode image at {img_path}")
        sys.exit(1)

    if not args.weights:
        print("Note: No model weights path provided. Use --weights path/to/model.onnx to run DNN inference.")
        print("Running with diagnostic mock detector on image ...")
        h, w = img.shape[:2]
        center_box = BoundingBox(w * 0.25, h * 0.25, w * 0.75, h * 0.75)
        mock = MockObjectDetector([Detection("workbench_roi", 0.85, center_box)])
        pipeline = DetectionPipeline(mock)
        batch = pipeline.process_frame(img)
        print(f"Detected {len(batch.detections)} mock items.")
        return

    weights_path = Path(args.weights)
    detector = OpenCVDNNDetector(weights_path=str(weights_path))
    if not detector.is_available():
        print(f"Error: Detector weights not valid at {weights_path}")
        sys.exit(1)

    pipeline = DetectionPipeline(detector)
    start = time.perf_counter()
    batch = pipeline.process_frame(img)
    latency = (time.perf_counter() - start) * 1000.0

    print(f"Detected {len(batch.detections)} objects in {latency:.2f} ms.")
    for det in batch.detections:
        print(f"- {det.label}: conf={det.confidence:.2%}, bbox={det.raw_bbox}")


if __name__ == "__main__":
    main()
