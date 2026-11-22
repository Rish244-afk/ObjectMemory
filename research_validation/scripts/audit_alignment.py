#!/usr/bin/env python3
"""
Camera and Trajectory Alignment Audit Script for OpenLORIS-Scene Office Pilot.
Evaluates timestamp matching, sampling rates, time offsets, and extrinsic calibrations.
"""

import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def parse_tum_trajectory(gt_file: Path) -> List[Tuple[float, List[float]]]:
    """Parses TUM format trajectory file: timestamp tx ty tz qx qy qz qw."""
    poses = []
    if not gt_file.exists():
        return poses
    with open(gt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 8:
                t = float(parts[0])
                pose_vec = [float(p) for p in parts[1:8]]
                poses.append((t, pose_vec))
    return poses

def parse_tum_image_timestamps(img_assoc_file: Path) -> List[Tuple[float, str]]:
    """Parses TUM format image association file: timestamp filename."""
    images = []
    if not img_assoc_file.exists():
        return images
    with open(img_assoc_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                t = float(parts[0])
                rel_path = parts[1]
                images.append((t, rel_path))
    return images

def find_nearest_pose(t_query: float, poses: List[Tuple[float, List[float]]]) -> Tuple[Optional[Tuple[float, List[float]]], float]:
    """Binary search for nearest timestamp in sorted pose list."""
    if not poses:
        return None, float("inf")
    
    low = 0
    high = len(poses) - 1
    while low <= high:
        mid = (low + high) // 2
        mid_t = poses[mid][0]
        if mid_t < t_query:
            low = mid + 1
        elif mid_t > t_query:
            high = mid - 1
        else:
            return poses[mid], 0.0

    best_pose = None
    min_dt = float("inf")
    for idx in [max(0, high), min(len(poses) - 1, low)]:
        dt = abs(poses[idx][0] - t_query)
        if dt < min_dt:
            min_dt = dt
            best_pose = poses[idx]
    return best_pose, min_dt

def audit_sequence(seq_id: str, seq_dir: Path, gt_file: Path) -> Dict:
    print(f"\n--- Auditing Sequence: {seq_id} ---")
    report = {
        "sequence_id": seq_id,
        "seq_dir": str(seq_dir),
        "gt_file": str(gt_file),
        "image_file_candidates": [],
        "camera_timestamps_found": False,
        "gt_poses_found": False,
        "total_images": 0,
        "total_gt_poses": 0,
        "time_alignment": {},
        "calibration": {},
        "synchronization_status": "PENDING"
    }

    # 1. Locate camera timestamps
    # In OpenLORIS / TUM format, image files are referenced in color.txt / rgb.txt
    img_list_candidates = [
        seq_dir / "color.txt",
        seq_dir / "rgb.txt",
        seq_dir / "images.txt",
    ]
    matched_img_file = None
    for cand in img_list_candidates:
        if cand.exists():
            matched_img_file = cand
            break

    images = []
    if matched_img_file:
        images = parse_tum_image_timestamps(matched_img_file)
        report["camera_timestamps_found"] = True
        report["image_file_candidates"].append(str(matched_img_file))
        report["total_images"] = len(images)
    else:
        # Fallback: check color/ or rgb/ folder directly
        img_dir_candidates = [seq_dir / "color", seq_dir / "rgb"]
        for idir in img_dir_candidates:
            if idir.exists() and idir.is_dir():
                raw_files = sorted(list(idir.glob("*.png")) + list(idir.glob("*.jpg")))
                report["total_images"] = len(raw_files)
                report["image_file_candidates"].append(str(idir))
                # Attempt to parse timestamp from filename (e.g. 1560000083.935123.png)
                for rf in raw_files:
                    try:
                        t = float(rf.stem)
                        images.append((t, str(rf.name)))
                    except ValueError:
                        pass
                if len(images) > 0:
                    report["camera_timestamps_found"] = True
                break

    # 2. Parse GT poses
    poses = parse_tum_trajectory(gt_file)
    if poses:
        report["gt_poses_found"] = True
        report["total_gt_poses"] = len(poses)

    # 3. Check timestamp overlap & matching
    if images and poses:
        img_t0, img_t1 = images[0][0], images[-1][0]
        gt_t0, gt_t1 = poses[0][0], poses[-1][0]
        
        report["time_alignment"] = {
            "image_start_time": img_t0,
            "image_end_time": img_t1,
            "image_duration_sec": round(img_t1 - img_t0, 3),
            "gt_start_time": gt_t0,
            "gt_end_time": gt_t1,
            "gt_duration_sec": round(gt_t1 - gt_t0, 3),
            "clock_difference_start_sec": round(img_t0 - gt_t0, 4),
        }

        # Match nearest pose for all camera frames
        offsets = []
        for img_t, _ in images:
            _, dt = find_nearest_pose(img_t, poses)
            offsets.append(dt)

        max_dt = max(offsets)
        mean_dt = sum(offsets) / len(offsets)
        report["time_alignment"]["max_nearest_pose_dt_sec"] = round(max_dt, 6)
        report["time_alignment"]["mean_nearest_pose_dt_sec"] = round(mean_dt, 6)

        # Check if timestamps overlap
        overlap_start = max(img_t0, gt_t0)
        overlap_end = min(img_t1, gt_t1)
        if overlap_end > overlap_start:
            report["time_alignment"]["temporal_overlap_sec"] = round(overlap_end - overlap_start, 3)
            # Evaluate nearest association
            if mean_dt <= 0.010: # within 10ms
                report["synchronization_status"] = "SYNCHRONIZED_SUB_CENTIMETER"
            elif mean_dt <= 0.050:
                report["synchronization_status"] = "SYNCHRONIZED_COARSE"
            else:
                report["synchronization_status"] = "LARGE_TIMING_OFFSET"
        else:
            report["time_alignment"]["temporal_overlap_sec"] = 0.0
            report["synchronization_status"] = "NO_TEMPORAL_OVERLAP"
    else:
        report["synchronization_status"] = "MISSING_DATA"

    # 4. Check for calibration files in sequence or parent directory
    calib_candidates = list(seq_dir.glob("*.yaml")) + list(seq_dir.parent.glob("*.yaml"))
    report["calibration"]["yaml_files_found"] = [str(c.name) for c in calib_candidates]

    return report

def main():
    if len(sys.argv) < 3:
        print("Usage: python audit_alignment.py <extracted_office_dir> <extracted_gt_dir>")
        sys.exit(1)

    office_dir = Path(sys.argv[1])
    gt_dir = Path(sys.argv[2])
    out_json = office_dir / "alignment_audit_report.json"

    target_seqs = ["office1-1", "office1-2", "office1-3"]
    all_reports = {}

    for seq in target_seqs:
        seq_path = office_dir / seq
        gt_path = gt_dir / "per-sequence" / seq / "groundtruth.txt"
        report = audit_sequence(seq, seq_path, gt_path)
        all_reports[seq] = report

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2)
    print(f"\nAudit complete. Master report written to: {out_json}")

if __name__ == "__main__":
    main()
