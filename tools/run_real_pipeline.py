"""
CLI tool for running the ObjectMemory real-image validation and detector integration pipeline.

Features:
1. Ingests real-world workbench visits from structured dataset directory.
2. Runs OpenCV DNN CPU inference (YOLOv8 nano ONNX).
3. Executes planar registration, canonical projection, appearance extraction, and temporal memory engine.
4. Profiles CPU latency, memory footprint, and detection performance.
5. Emits explicit blocked status when manual annotations are missing.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile
import cv2
import numpy as np

# Ensure src is in sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from objectmemory.benchmarks.real_evaluator import RealBenchmarkEvaluator
from objectmemory.datasets.real_dataset import (
    RealVisitAnnotation,
    RealVisitFrame,
    RealWorkbenchDataset,
)
from objectmemory.detection.detector import OpenCVDNNDetector
from objectmemory.persistence.repository import SQLiteMemoryStore
from objectmemory.pipeline.real_pipeline import RealImagePipelineRunner


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ObjectMemory Phase 6: Real-Image Validation and Detector Integration"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/real_workbench"),
        help="Directory containing real dataset manifest and visit images (default: data/real_workbench)",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("models/yolov8n.onnx"),
        help="Path to pretrained ONNX weights (default: models/yolov8n.onnx)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confidence threshold for real object detection (default: 0.35)",
    )
    parser.add_argument(
        "--benchmark-cpu",
        action="store_true",
        help="Run CPU latency and memory benchmark on available real imagery",
    )
    args = parser.parse_args()

    print("=" * 86)
    print(" ObjectMemory: Long-Term Visual Memory for Detecting Persistent Environmental Changes")
    print(" Phase 6: Real-Image Validation and Detector Integration")
    print("=" * 86)

    # 1. Check detector weights
    if not args.weights.exists():
        print(f"\n[ERROR] Detector weights not found at: {args.weights}")
        print("Please ensure yolov8n.onnx is present in the models/ directory.")
        return 1

    print(f"\n[1/4] Initializing CPU Object Detector...")
    detector = OpenCVDNNDetector(
        weights_path=str(args.weights),
        framework="onnx",
        architecture="yolov8",
    )
    info = detector.model_info()
    print(f"      Model:        {info['name']}")
    print(f"      Backend:      {info['backend']}")
    print(f"      Classes:      {info['classes_count']} classes")
    print(f"      Weights:      {args.weights} ({args.weights.stat().st_size / (1024*1024):.2f} MB)")

    # 2. Dataset inspection / template creation
    data_dir = args.data_dir
    manifest_path = data_dir / "manifest.json"

    if not manifest_path.exists():
        print(f"\n[2/4] Dataset directory '{data_dir}' not found or uninitialized.")
        print(f"      Creating empty template with schema specification at: {data_dir}")
        RealWorkbenchDataset.create_empty_template(data_dir)

    dataset = RealWorkbenchDataset.load_from_directory(data_dir)
    print(f"\n[2/4] Loaded real-image dataset: '{dataset.dataset_name}'")
    print(f"      Visits found: {len(dataset.visits)}")
    is_ready, reason = dataset.is_evaluation_ready()
    print(f"      Evaluation status: {'READY' if is_ready else 'BLOCKED_PENDING_ANNOTATIONS'}")
    if not is_ready:
        print(f"      Status note: {reason}")

    # 3. If benchmark-cpu is requested or images exist, demonstrate execution
    print("\n[3/4] Processing Imagery & Profiling CPU Performance...")
    evaluator = RealBenchmarkEvaluator(confidence_threshold=args.conf)

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "real_memory.db"
        with SQLiteMemoryStore(db_path) as store:
            runner = RealImagePipelineRunner(store=store, detector=detector)

            # If real dataset visits have images, evaluate the dataset
            if dataset.visits and all(v.image_path.exists() for v in dataset.visits):
                report = evaluator.evaluate(dataset, runner)
                print("\n[4/4] Pipeline Execution Summary:")
                print(report.format_text_summary())
            else:
                # Check if demo alignment image exists to run a CPU validation pass
                demo_img_path = Path("demo_alignment_overlay.png")
                if demo_img_path.exists():
                    print(f"      No physical dataset frames found; executing CPU inference validation on '{demo_img_path}'...")
                    img = cv2.imread(str(demo_img_path))
                    assert img is not None
                    batch = detector.detect(img, confidence_threshold=args.conf)
                    print(f"      -> Detections found: {len(batch.detections)}")
                    for d in batch.detections:
                        print(f"         - {d.label:<15} conf={d.confidence:.3f} bbox=({d.raw_bbox.x_min:.1f}, {d.raw_bbox.y_min:.1f}, {d.raw_bbox.x_max:.1f}, {d.raw_bbox.y_max:.1f})")
                    print(f"      -> CPU Inference Latency: {batch.latency_ms:.1f} ms")

                print("\n[4/4] Real-Image Evaluation Status:")
                print("=" * 86)
                print(" STATUS: BLOCKED_PENDING_ANNOTATIONS")
                print(" Reason: Real workbench dataset does not contain physical multi-visit images and manual ground-truth annotations.")
                print(f" Manifest location: {manifest_path.resolve()}")
                print("=" * 86)

    return 0


if __name__ == "__main__":
    sys.exit(main())
