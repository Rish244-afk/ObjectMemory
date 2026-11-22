#!/usr/bin/env python3
"""
CLI script to validate the Controlled Temporal Benchmark dataset.
Produces a machine-readable validation report at data/controlled_temporal_v1/validation_report.json.
"""

import argparse
import json
import os
import sys

# Ensure src is in python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from objectmemory.datasets.controlled_dataset import ControlledDataset


def main():
    parser = argparse.ArgumentParser(description="Validate Controlled Temporal Benchmark Dataset")
    parser.add_argument(
        "--dataset-dir",
        default="data/controlled_temporal_v1",
        help="Path to controlled dataset directory",
    )
    parser.add_argument(
        "--output-report",
        default=None,
        help="Path to write JSON validation report (default: <dataset-dir>/validation_report.json)",
    )
    args = parser.parse_args()

    dataset_path = os.path.abspath(args.dataset_dir)
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset directory does not exist: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Validating Controlled Temporal Benchmark dataset at: {dataset_path}")
    dataset = ControlledDataset.load(dataset_path)
    report = dataset.validate()

    out_path = dataset.save_validation_report(args.output_report)
    print(f"Validation finished with status: {report['status']}")
    print(f"Images validated: {report['images_validated']}")
    print(f"Annotations validated: {report['annotations_validated']}")
    print(f"Expected events validated: {report['expected_events_validated']}")
    print(f"Sequences count: {report['sequences_count']}")

    if report["errors"]:
        print(f"ERRORS ({len(report['errors'])}):", file=sys.stderr)
        for err in report["errors"]:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"Validation report saved to: {out_path}")
        print("ALL INTEGRITY CHECKS PASSED.")


if __name__ == "__main__":
    main()
