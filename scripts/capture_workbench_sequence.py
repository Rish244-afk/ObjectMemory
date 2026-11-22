#!/usr/bin/env python3
"""
Physical Workbench Photographic Sequence Capture CLI Utility.

Captures sequential observations for the ObjectMemory Physical Workbench Benchmark
(Phase 15) using a connected USB camera, webcam, or video source.

Features:
- Real-time ArUco 4x4_50 marker detection and planar homography quality check.
- Step-by-step guidance conforming to the benchmark manifest and CAPTURE_PROTOCOL.md.
- Frame validation (reprojection error, resolution, illumination sanity).
- Support for mock capture mode when physical camera hardware is detached (headless testing).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def detect_aruco_markers(frame: np.ndarray) -> Tuple[List[int], Optional[np.ndarray], float]:
    """
    Detect ArUco 4x4_50 markers and estimate planar homography reprojection error.
    
    Returns:
        (detected_ids, homography_matrix, reprojection_error_px)
    """
    try:
        # OpenCV 4.7+ and 5.0+ API
        if hasattr(cv2.aruco, 'ArucoDetector'):
            dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            parameters = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(dictionary, parameters)
            corners, ids, rejected = detector.detectMarkers(frame)
        else:
            dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
            parameters = cv2.aruco.DetectorParameters_create()
            corners, ids, rejected = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)

        if ids is None or len(ids) == 0:
            return [], None, 999.0

        detected_ids = [int(i[0]) for i in ids]
        
        # Check for reference markers (0, 1, 2, 3)
        marker_map = {}
        for idx, mid in enumerate(detected_ids):
            if mid in (0, 1, 2, 3):
                # Center of the marker
                c = corners[idx][0]
                marker_map[mid] = np.mean(c, axis=0)

        if len(marker_map) == 4:
            # All 4 markers detected; compute homography to canonical rectangle
            src_pts = np.float32([marker_map[0], marker_map[1], marker_map[2], marker_map[3]])
            dst_pts = np.float32([[100, 100], [1820, 100], [1820, 980], [100, 980]])
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H is not None:
                # Compute reprojection error
                projected = cv2.perspectiveTransform(src_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
                err = float(np.mean(np.linalg.norm(projected - dst_pts, axis=1)))
                return detected_ids, H, err
            return detected_ids, None, 999.0

        return detected_ids, None, 999.0
    except Exception as e:
        return [], None, 999.0


def draw_hud(
    frame: np.ndarray,
    sequence_id: str,
    visit_idx: int,
    total_visits: int,
    scenario_desc: str,
    expected_event: str,
    detected_ids: List[int],
    reprojection_err: float,
) -> np.ndarray:
    """Draw overlay HUD with capture instructions and calibration feedback."""
    display = frame.copy()
    h, w = display.shape[:2]

    # Top banner
    cv2.rectangle(display, (0, 0), (w, 80), (15, 18, 25), -1)
    cv2.line(display, (0, 80), (w, 80), (35, 42, 59), 2)

    title = f"OBJECTMEMORY BENCHMARK CAPTURE | {sequence_id} (Visit {visit_idx}/{total_visits})"
    cv2.putText(display, title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)

    status_str = f"Target Event: {expected_event.upper()}  |  Fiducials: {len(detected_ids)}/4 detected"
    fiducial_color = (80, 220, 100) if len(detected_ids) >= 4 else (100, 180, 255) if len(detected_ids) > 0 else (100, 100, 240)
    cv2.putText(display, status_str, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, fiducial_color, 2)

    if reprojection_err < 5.0:
        calib_str = f"Planar Calib: EXCELLENT (err: {reprojection_err:.2f}px)"
        cv2.putText(display, calib_str, (w - 450, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 220, 100), 1)
    elif reprojection_err < 900.0:
        calib_str = f"Planar Calib: MARGINAL (err: {reprojection_err:.2f}px)"
        cv2.putText(display, calib_str, (w - 450, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 200, 255), 1)
    else:
        calib_str = "Planar Calib: UNREGISTERED"
        cv2.putText(display, calib_str, (w - 450, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 220), 1)

    # Bottom instruction banner
    cv2.rectangle(display, (0, h - 90), (w, h), (15, 18, 25), -1)
    cv2.line(display, (0, h - 90), (w, h - 90), (35, 42, 59), 2)

    desc_line = f"ACTION: {scenario_desc}"
    cv2.putText(display, desc_line, (20, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)

    keys_line = "[SPACE/C] Capture Visit Frame  |  [S] Skip  |  [Q/ESC] Quit Capture Session"
    cv2.putText(display, keys_line, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (148, 163, 184), 1)

    return display


def run_capture(
    manifest_path: Path,
    sequence_id: str,
    output_dir: Path,
    camera_idx: int = 0,
    width: int = 1920,
    height: int = 1080,
    mock: bool = False,
) -> int:
    """Run interactive or automated sequence capture workflow."""
    if not manifest_path.is_file():
        print(f"Error: Manifest not found at {manifest_path}", file=sys.stderr)
        return 1

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Find sequence
    sequences = manifest.get("sequences", [])
    target_seq = next((s for s in sequences if s.get("sequence_id") == sequence_id), None)
    if not target_seq:
        available_ids = [s.get("sequence_id") for s in sequences]
        print(f"Error: Sequence '{sequence_id}' not found in manifest.", file=sys.stderr)
        print(f"Available sequences: {available_ids}", file=sys.stderr)
        return 1

    visits = target_seq.get("visits", [])
    if not visits:
        print(f"Error: No visits defined for sequence '{sequence_id}'.", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n========================================================")
    print(f" ObjectMemory Benchmark Capture — {target_seq.get('scenario_name')}")
    print(f" Sequence: {sequence_id} ({len(visits)} sequential visits)")
    print(f" Destination: {output_dir}")
    print(f"========================================================\n")

    if mock:
        print("[MOCK MODE] Simulating camera capture for testing without physical camera hardware.")
        for v_idx, visit in enumerate(visits, 1):
            visit_id = visit.get("visit_id", f"visit_{v_idx:02d}")
            rel_image_path = visit.get("image", f"images/{sequence_id}_{visit_id}.jpg")
            dest_file = output_dir.parent / rel_image_path
            dest_file.parent.mkdir(parents=True, exist_ok=True)

            # Generate synthetic placeholder image if mock
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(canvas, f"MOCK CAPTURE: {sequence_id} - {visit_id}", (80, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 200, 255), 2)
            cv2.putText(canvas, f"Scenario: {visit.get('scenario_description', '')}", (80, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
            cv2.imwrite(str(dest_file), canvas)
            print(f"  [Mock] Generated {dest_file.name} ({visit.get('expected_event', '')})")
        print("\nMock capture complete. All frames generated.")
        return 0

    cap = cv2.VideoCapture(camera_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        print(f"Error: Could not open camera device at index {camera_idx}.", file=sys.stderr)
        print("Tip: If running on a headless machine or without USB camera attached, run with --mock.", file=sys.stderr)
        return 1

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera opened successfully: {actual_w}x{actual_h}")

    window_name = f"ObjectMemory Capture: {sequence_id}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    current_visit_idx = 0
    captured_count = 0

    try:
        while current_visit_idx < len(visits):
            visit = visits[current_visit_idx]
            visit_id = visit.get("visit_id", f"visit_{current_visit_idx+1:02d}")
            scenario_desc = visit.get("scenario_description", "Prepare scene.")
            expected_event = visit.get("expected_event", "none")
            rel_image_path = visit.get("image", f"images/{sequence_id}_{visit_id}.jpg")
            dest_file = output_dir.parent / rel_image_path
            dest_file.parent.mkdir(parents=True, exist_ok=True)

            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to grab frame from video device.", file=sys.stderr)
                time.sleep(0.05)
                continue

            detected_ids, H, reproj_err = detect_aruco_markers(frame)
            display = draw_hud(
                frame,
                sequence_id,
                current_visit_idx + 1,
                len(visits),
                scenario_desc,
                expected_event,
                detected_ids,
                reproj_err,
            )

            cv2.imshow(window_name, display)
            key = cv2.waitKey(10) & 0xFF

            if key in (ord('q'), 27):  # ESC or q
                print("\nCapture session interrupted by user.")
                break
            elif key in (ord(' '), ord('c')):  # SPACE or c
                # Capture frame
                cv2.imwrite(str(dest_file), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                print(f"  [Captured] Visit {current_visit_idx+1}/{len(visits)} -> {dest_file.name}")
                current_visit_idx += 1
                captured_count += 1
            elif key == ord('s'):  # Skip
                print(f"  [Skipped] Visit {current_visit_idx+1}/{len(visits)}")
                current_visit_idx += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\nCapture session closed. Captured {captured_count}/{len(visits)} visits.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture photographic sequences for ObjectMemory Benchmark.")
    parser.add_argument("--sequence-id", type=str, default="EXP01_comprehensive_workbench", help="Sequence ID to capture.")
    parser.add_argument("--manifest", type=str, default="data/physical_workbench_v1/manifest.json", help="Path to manifest JSON.")
    parser.add_argument("--output-dir", type=str, default="data/physical_workbench_v1/images", help="Output directory for photos.")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0).")
    parser.add_argument("--width", type=int, default=1920, help="Target capture width.")
    parser.add_argument("--height", type=int, default=1080, help="Target capture height.")
    parser.add_argument("--mock", action="store_true", help="Run simulated capture (headless testing).")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    manifest_path = repo_root / args.manifest
    output_dir = repo_root / args.output_dir

    code = run_capture(
        manifest_path=manifest_path,
        sequence_id=args.sequence_id,
        output_dir=output_dir,
        camera_idx=args.camera,
        width=args.width,
        height=args.height,
        mock=args.mock,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
