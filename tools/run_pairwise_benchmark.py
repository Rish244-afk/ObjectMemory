"""
CLI tool for running the Phase 4 Independent Pairwise Baseline Benchmark.

Generates the 8 controlled synthetic workbench scenarios, runs registration,
extracts visual signatures, executes deterministic pairwise matching,
and prints the formal evaluation metrics against ground truth.

Usage:
    python tools/run_pairwise_benchmark.py [--save-data] [--persist] [--db path/to/db.sqlite]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root))

from objectmemory.benchmarks.evaluator import PairwiseBenchmarkEvaluator
from objectmemory.benchmarks.synthetic_dataset import SyntheticWorkbenchDatasetGenerator
from objectmemory.domain.models import PairwiseBaselineResult, Session
from objectmemory.domain.states import BaselineChangeType
from objectmemory.persistence.repository import SQLiteMemoryStore


def main() -> None:
    parser = argparse.ArgumentParser(description="ObjectMemory Pairwise Baseline Benchmark Runner")
    parser.add_argument("--save-data", action="store_true", default=True, help="Save synthetic dataset to data/synthetic_workbench/")
    parser.add_argument("--data-dir", type=str, default="data/synthetic_workbench", help="Directory for synthetic dataset")
    parser.add_argument("--persist", action="store_true", help="Persist baseline results to SQLite database")
    parser.add_argument("--db", type=str, default="objectmemory_benchmark.sqlite", help="Path to SQLite database")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for dataset reproducibility")
    args = parser.parse_args()

    print("=" * 78)
    print(" ObjectMemory: Independent Pairwise Baseline Benchmark ")
    print("=" * 78)

    t0 = time.perf_counter()
    generator = SyntheticWorkbenchDatasetGenerator(width=640, height=480)
    print(f"\n[1/3] Generating 8 controlled synthetic workbench scenarios (seed={args.seed})...")
    scenarios = generator.generate_all_scenarios(base_seed=args.seed)

    if args.save_data:
        out_dir = repo_root / args.data_dir
        generator.save_dataset(out_dir, scenarios)
        print(f"      Saved synthetic dataset to: {out_dir}")

    print(f"\n[2/3] Evaluating pairwise baseline across {len(scenarios)} scenarios...")
    evaluator = PairwiseBenchmarkEvaluator()
    report = evaluator.evaluate_all(scenarios)
    elapsed = time.perf_counter() - t0

    print(f"\n[3/3] Benchmark evaluation complete in {elapsed:.2f}s.")
    print()
    print(report.format_text_summary())

    if args.persist:
        print(f"\n[Persistence] Storing results to SQLite database: {args.db}")
        store = SQLiteMemoryStore(repo_root / args.db)
        # Create dummy sessions if needed and save pairwise results
        for s_idx, summary in enumerate(report.scenario_summaries):
            ref_sid = f"bench_{summary.scenario_name}_ref"
            qry_sid = f"bench_{summary.scenario_name}_qry"

            now_iso = "2026-10-10T00:00:00Z"
            if not store.get_session(ref_sid):
                store.add_session(Session(ref_sid, f"Benchmark {summary.scenario_name} Ref", now_iso, "synthetic://ref"))
            if not store.get_session(qry_sid):
                store.add_session(Session(qry_sid, f"Benchmark {summary.scenario_name} Qry", now_iso, "synthetic://qry"))

            for r_idx, rec in enumerate(summary.object_records):
                res_id = f"pw_{summary.scenario_name}_{rec.object_id}_{r_idx}"
                pw_res = PairwiseBaselineResult(
                    result_id=res_id,
                    reference_session_id=ref_sid,
                    query_session_id=qry_sid,
                    entity_id=None,
                    predicted_change=rec.predicted,
                    confidence=rec.match_confidence,
                    details={
                        "object_id": rec.object_id,
                        "ground_truth": rec.ground_truth.value,
                        "is_correct": rec.is_correct,
                        "explanation": rec.explanation,
                        "scenario_name": summary.scenario_name,
                        "is_synthetic": True,
                    },
                )
                store.add_pairwise_result(pw_res)
        store.close()
        print(f"      Stored {report.total_objects} pairwise results successfully.")


if __name__ == "__main__":
    main()
