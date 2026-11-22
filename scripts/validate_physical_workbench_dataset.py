#!/usr/bin/env python3
"""
Physical Workbench Benchmark Dataset Validator (Phase 14C).

Validates the structure, schemas, annotations, image files, coordinate bounds,
absence classifications, and ArUco fiducials for the real photographic benchmark.

Usage:
    python scripts/validate_physical_workbench_dataset.py --dataset-dir data/physical_workbench_v1
    python scripts/validate_physical_workbench_dataset.py --dataset-dir data/physical_workbench_v1 --require-images
    python scripts/validate_physical_workbench_dataset.py --dataset-dir data/physical_workbench_v1 --overlay-dir data/physical_workbench_v1/overlays
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import numpy as np

# Ensure src is on python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from objectmemory.datasets.physical_workbench import (
    AbsenceReason,
    DatasetCaptureStatus,
    PhysicalVisibility,
    PhysicalWorkbenchDataset,
    detect_aruco_corner_markers,
)


def render_annotation_overlays(dataset: PhysicalWorkbenchDataset, overlay_dir: str) -> int:
    """Renders ground-truth boxes and ArUco markers on existing images for visual inspection."""
    os.makedirs(overlay_dir, exist_ok=True)
    rendered_count = 0

    colors = {
        "cup": (0, 165, 255),       # Orange
        "book": (255, 100, 0),      # Blue
        "scissors": (0, 0, 255),    # Red
        "bottle": (0, 255, 0),      # Green
        "cell phone": (255, 0, 255), # Magenta
    }

    for seq_id, seq in dataset.sequences.items():
        for visit in seq.visits:
            if not visit.is_image_present():
                continue

            img = visit.load_image()
            if img is None:
                continue

            canvas = img.copy()

            # 1. Overlay ArUco markers if present
            aruco_info = detect_aruco_corner_markers(img)
            if aruco_info.get("markers_detected", 0) > 0:
                for mid, corners in zip(aruco_info["marker_ids"], aruco_info["corners"]):
                    pts = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(canvas, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
                    c_x, c_y = int(np.mean([p[0] for p in corners])), int(np.mean([p[1] for p in corners]))
                    cv2.putText(
                        canvas,
                        f"ArUco {mid}",
                        (c_x - 20, c_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        2,
                    )

            # 2. Overlay ground truth annotations
            for ann in visit.annotations:
                if ann.bbox is not None and ann.visibility != PhysicalVisibility.ABSENT:
                    b = ann.bbox
                    color = colors.get(ann.class_label, (255, 255, 255))
                    x1, y1, x2, y2 = int(b.x_min), int(b.y_min), int(b.x_max), int(b.y_max)
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
                    label_text = f"{ann.object_id} ({ann.class_label})"
                    if ann.visibility == PhysicalVisibility.PARTIALLY_OCCLUDED:
                        label_text += " [OCCLUDED]"
                    cv2.putText(
                        canvas,
                        label_text,
                        (x1, max(15, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )

            out_filename = f"{seq_id}_{visit.visit_id}_overlay.jpg"
            out_path = os.path.join(overlay_dir, out_filename)
            cv2.imwrite(out_path, canvas)
            rendered_count += 1

    return rendered_count


def main():
    parser = argparse.ArgumentParser(description="Validate Physical Workbench Benchmark Dataset")
    parser.add_argument(
        "--dataset-dir",
        default="data/physical_workbench_v1",
        help="Path to physical workbench dataset directory",
    )
    parser.add_argument(
        "--output-report",
        default=None,
        help="Path to save machine-readable validation report JSON",
    )
    parser.add_argument(
        "--overlay-dir",
        default=None,
        help="Optional directory to render visual inspection bounding box overlays",
    )
    parser.add_argument(
        "--require-images",
        action="store_true",
        help="If set, validation fails if any photographic image file is missing",
    )
    args = parser.parse_args()

    dataset_path = os.path.abspath(args.dataset_dir)
    print(f"Loading Physical Workbench Dataset from: {dataset_path}")

    try:
        dataset = PhysicalWorkbenchDataset.load(dataset_path)
    except Exception as e:
        print(f"[FATAL ERROR] Could not load dataset: {e}")
        sys.exit(1)

    report = dataset.validate()

    # If --require-images is set and images are missing, treat as error
    if args.require_images and report["images_missing"] > 0:
        report["errors"].append(
            f"--require-images specified but {report['images_missing']} images are missing from disk."
        )
        report["status"] = "FAILED"

    report_path = dataset.save_validation_report(args.output_report)

    # Render overlays if requested
    overlays_rendered = 0
    if args.overlay_dir:
        overlays_rendered = render_annotation_overlays(dataset, os.path.abspath(args.overlay_dir))

    # Console output
    print("\n" + "=" * 65)
    print(f"  PHYSICAL WORKBENCH DATASET VALIDATION REPORT")
    print("=" * 65)
    print(f"  Dataset Name:        {report['dataset_name']}")
    print(f"  Version:             {report['dataset_version']}")
    print(f"  Status:              {report['status']}")
    print(f"  Capture Status:      {report['capture_status']}")
    print(f"  Total Sequences:     {report['total_sequences']}")
    print(f"  Total Visits:        {report['total_visits']}")
    print(f"  Images Found:        {report['images_found']}")
    print(f"  Images Missing:      {report['images_missing']}")
    print(f"  Total Annotations:   {report['total_annotations']}")
    print(f"  Validation Report:   {report_path}")
    if args.overlay_dir:
        print(f"  Overlays Rendered:   {overlays_rendered} to {args.overlay_dir}")

    if report["warnings"]:
        print(f"\n  Warnings ({len(report['warnings'])}):")
        for w in report["warnings"][:10]:
            print(f"    - {w}")

    if report["errors"]:
        print(f"\n  [ERROR] Validation Errors ({len(report['errors'])}):")
        for err in report["errors"][:10]:
            print(f"    - {err}")
        print("=" * 65)
        sys.exit(1)

    print("=" * 65)
    if report["capture_status"] == DatasetCaptureStatus.AWAITING_PHYSICAL_DATA.value:
        print("\n[NOTE] Dataset schemas and annotations are fully validated.")
        print("Photographic images are currently awaiting physical capture by the user.")
        print("Refer to data/physical_workbench_v1/README.md for instructions.")
    else:
        print(f"\nDataset validation PASSED ({report['capture_status']}).")

    sys.exit(0)


if __name__ == "__main__":
    main()
