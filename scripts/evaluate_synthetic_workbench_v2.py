"""
Execute benchmark matrix on synthetic_workbench_v2 dataset (25 observations).
Evaluates supported configurations: B0, B3, A5, and partials A2, A3.
Saves JSON results under research_validation/results/benchmarks/.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from objectmemory.benchmarks.config_factory import BenchmarkConfigFactory
from objectmemory.benchmarks.manifest import ManifestLoader
from objectmemory.benchmarks.runner import BenchmarkRunner
from objectmemory.detection.detector import OpenCVDNNDetector


def main() -> None:
    manifest_path = Path("data/synthetic_workbench_v2/manifest_v2.json")
    results_dir = Path("research_validation/results/benchmarks")
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading manifest from {manifest_path}...")
    manifest = ManifestLoader.load(manifest_path, check_files=True)
    print(f"Loaded {len(manifest.sequences)} sequences ({manifest.total_observations} total observations)")

    weights_path = Path("models/yolov8n.onnx")
    detector = OpenCVDNNDetector(weights_path=str(weights_path))
    runner = BenchmarkRunner(detector=detector, allow_partial=True)

    configs_to_run = ["B0", "B3", "A5", "A2", "A3"]

    for config_id in configs_to_run:
        print(f"\n--- Running Configuration {config_id} ---")
        cfg = BenchmarkConfigFactory.create_config(config_id, allow_partial=True)
        print(f"Config Name: {cfg.name}")
        print(f"Support Status: {cfg.support_status.name}")
        if cfg.notes:
            print(f"Notes: {cfg.notes}")

        result = runner.run_config(manifest, cfg)
        out_file = results_dir / f"{config_id}_synthetic_workbench_v2_result.json"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(result.to_json())
        print(f"Saved result to {out_file}")

        # Print key metrics
        m = result.metrics
        print(f"Execution runtime: {result.runtime_seconds:.2f}s")
        print(f"Detection P/R/F1: {m.detection_precision}/{m.detection_recall}/{m.detection_f1_score}")
        print(f"FRR: {m.false_removal_rate}, FDP: {m.false_discovery_proportion}")
        print(f"Removal Recall: {m.removal_recall}, Confirmation Rate: {m.confirmation_rate}")
        print(f"Identity Consistency: {m.identity_consistency}, Fragmentation: {m.fragmented_identities}")
        print(f"Overall State Accuracy: {m.overall_state_accuracy}")


if __name__ == "__main__":
    main()
