"""
Benchmark runner CLI tool for Phase 5 Multi-Visit Temporal Memory.

Compares Multi-Visit Temporal Memory Engine against the Phase 4 Pairwise Baseline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

# Add src to sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from objectmemory.benchmarks.temporal_dataset import SyntheticTemporalSequenceGenerator
from objectmemory.benchmarks.temporal_evaluator import TemporalBenchmarkEvaluator


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ObjectMemory Phase 5: Multi-Visit Temporal Memory Benchmark"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sequence generation (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/temporal_workbench"),
        help="Directory to save synthetic sequences (default: data/temporal_workbench)",
    )
    args = parser.parse_args()

    print("=" * 78)
    print(" ObjectMemory: Multi-Visit Temporal Memory Benchmark ")
    print("=" * 78)
    print(f"\n[1/3] Generating 10 multi-visit sequences (seed={args.seed})...")

    gen = SyntheticTemporalSequenceGenerator()
    sequences = gen.generate_all_sequences(base_seed=args.seed)
    print(f"      Generated {len(sequences)} controlled sequences (>= 3 visits each).")

    print("\n[2/3] Evaluating Temporal Engine vs Pairwise Baseline...")
    evaluator = TemporalBenchmarkEvaluator()

    with tempfile.TemporaryDirectory() as tmp_dir:
        report = evaluator.evaluate_all_sequences(sequences, Path(tmp_dir))

    print("\n[3/3] Evaluation complete.")
    print("\n" + report.format_text_summary())

    return 0


if __name__ == "__main__":
    sys.exit(main())
