#!/usr/bin/env python3
"""
Integrated Production Verification: Pretrained Official YOLOv8m (COCO)
loaded via DetectorRegistry.get() (the production default) under frozen B3 config
on data/physical_pilot_v2/manifest_v2.json.
Saves results to research_validation/larger_detector_experiment/yolov8m_production_integrated_results.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from objectmemory.benchmarks.config_factory import BenchmarkConfigFactory
from objectmemory.benchmarks.manifest import ManifestLoader
from objectmemory.benchmarks.runner import BenchmarkRunner
from objectmemory.detection.registry import DetectorRegistry
from run_yolov8s_b3 import evaluate_detection


def main() -> None:
    # 1. Verify default detector name in registry
    default_name = DetectorRegistry.get_default_name()
    assert default_name == "yolov8m", f"Expected default detector to be 'yolov8m', got '{default_name}'"

    # 2. Instantiate default detector via production registry
    detector = DetectorRegistry.get()  # No arguments -> loads production default
    weights = Path(detector.weights_path)
    assert weights.name == "yolov8m.onnx", f"Expected weights to be yolov8m.onnx, got {weights}"

    # 3. Load frozen physical pilot v2 manifest
    manifest = ManifestLoader.load(PROJECT_ROOT / "data/physical_pilot_v2/manifest_v2.json", check_files=True)

    # 4. Execute frozen B3 benchmark
    runner = BenchmarkRunner(detector=detector, allow_partial=True)
    res = runner.run_config(manifest, BenchmarkConfigFactory.create_config("B3"))
    det_eval = evaluate_detection(manifest, detector, iou_threshold=0.50)

    payload = res.to_dict()
    payload["detector_registry_name"] = default_name
    payload["model_weights"] = str(weights)
    payload["model_size_mb"] = round(weights.stat().st_size / (1024 * 1024), 2)
    payload["detection_evaluation"] = det_eval

    out_json = Path(__file__).resolve().parent / "yolov8m_production_integrated_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    print("Saved production integrated results to:", out_json)
    print("Metrics:", res.metrics)
    print("Detection TP:", det_eval["total_tp"], "/", det_eval["total_gt_visible"], "recall=", det_eval["overall_recall"], "prec=", det_eval["overall_precision"], "mean_ms=", det_eval["mean_inference_ms"])
    for c, st in det_eval["per_class"].items():
        print(f"  {c}: tp={st['tp']}/{st['gt_visible']} recall={st['recall']} preds={st['predictions']} prec={st['precision']}")


if __name__ == "__main__":
    main()
