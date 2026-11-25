#!/usr/bin/env python3
"""
Captures one original 32-second recording from the actual connected webcam (device 0),
saves it with full capture metadata, and replays the exact same recording through
both YOLOv8m and YOLO26m to measure live performance and ObjectMemory pipeline behavior.
"""

from __future__ import annotations

import json
import os
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from objectmemory.baseline.comparator import calculate_bbox_iou
from objectmemory.detection.registry import DetectorRegistry
from objectmemory.registration.registrar import PlanarRegistrar


def _proc_stats() -> Dict[str, Any]:
    try:
        import psutil
        p = psutil.Process(os.getpid())
        return {
            "rss_mb": round(p.memory_info().rss / (1024 * 1024), 2),
            "cpu_percent": psutil.cpu_percent(interval=None),
        }
    except Exception:
        return {"rss_mb": None, "cpu_percent": None}


def capture_webcam_recording(out_dir: Path, duration_sec: float = 32.0) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "live_webcam_32s.avi"
    meta_path = out_dir / "live_webcam_32s_meta.json"

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam device 0 could not be opened.")

    # Request 1280x720 if supported, otherwise camera uses native resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if reported_fps <= 1.0 or reported_fps > 120.0:
        reported_fps = 30.0
    backend_name = cap.getBackendName() if hasattr(cap, "getBackendName") else "unknown"

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(video_path), fourcc, reported_fps, (actual_w, actual_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open VideoWriter at {video_path}")

    # Warmup 5 frames for auto-exposure
    for _ in range(5):
        cap.read()

    start_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t_start = time.perf_counter()
    frame_timestamps: List[float] = []
    frame_count = 0

    while (time.perf_counter() - t_start) < duration_sec:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        t_rel = time.perf_counter() - t_start
        writer.write(frame)
        frame_timestamps.append(round(t_rel, 4))
        frame_count += 1

    elapsed = time.perf_counter() - t_start
    end_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    cap.release()
    writer.release()

    actual_fps = frame_count / elapsed if elapsed > 0 else 0.0
    meta = {
        "video_path": str(video_path),
        "device_index": 0,
        "backend": backend_name,
        "resolution": [actual_w, actual_h],
        "reported_fps": round(reported_fps, 2),
        "actual_capture_fps": round(actual_fps, 2),
        "duration_sec": round(elapsed, 2),
        "total_frames": frame_count,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "codec": "MJPG",
        "frame_timestamps_sec": frame_timestamps,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def replay_recording(model_name: str, video_path: Path, meta: Dict[str, Any], sample_hz: float = 1.0) -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to reopen saved recording: {video_path}")

    frames: List[tuple[int, float, np.ndarray]] = []
    timestamps = meta["frame_timestamps_sec"]
    idx = 0
    next_t = 0.0
    step_t = 1.0 / sample_hz
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        t_sec = timestamps[idx] if idx < len(timestamps) else (idx / max(meta["actual_capture_fps"], 1.0))
        if t_sec >= next_t:
            frames.append((idx, t_sec, frame))
            next_t = t_sec + step_t
        idx += 1
    cap.release()

    detector = DetectorRegistry.get(model_name)
    if hasattr(detector, "ensure_loaded"):
        detector.ensure_loaded()
    # Warmup
    if frames:
        _ = detector.detect(frames[0][2], confidence_threshold=0.25)

    registrar = PlanarRegistrar()
    ref_img = frames[0][2] if frames else None

    _ = _proc_stats()
    tracemalloc.start()
    rss_before = _proc_stats()["rss_mb"]

    latencies_ms: List[float] = []
    e2e_latencies_ms: List[float] = []
    cpu_samples: List[float] = []
    per_frame_summaries: List[Dict[str, Any]] = []
    class_counts: Dict[str, int] = {}
    class_confs: Dict[str, List[float]] = {}
    class_boxes: Dict[str, List[Any]] = {}
    reg_success = 0
    reg_total = 0

    for f_i, (orig_idx, t_sec, frame) in enumerate(frames):
        t_e2e_0 = time.perf_counter()
        t_det_0 = time.perf_counter()
        batch = detector.detect(frame, confidence_threshold=0.25)
        det_ms = (time.perf_counter() - t_det_0) * 1000.0

        reg_status = "REFERENCE"
        inlier_ratio = 1.0
        if f_i > 0 and ref_img is not None:
            reg_total += 1
            reg = registrar.register(ref_img, frame)
            reg_status = reg.status.value
            inlier_ratio = float(reg.inlier_ratio or 0.0)
            if reg.success:
                reg_success += 1

        e2e_ms = (time.perf_counter() - t_e2e_0) * 1000.0
        latencies_ms.append(det_ms)
        e2e_latencies_ms.append(e2e_ms)
        st = _proc_stats()
        if st["cpu_percent"] is not None:
            cpu_samples.append(st["cpu_percent"])

        dets_summary = []
        for d in batch.detections:
            class_counts[d.label] = class_counts.get(d.label, 0) + 1
            class_confs.setdefault(d.label, []).append(float(d.confidence))
            class_boxes.setdefault(d.label, []).append((f_i, d.raw_bbox))
            dets_summary.append({"label": d.label, "conf": round(float(d.confidence), 4)})

        per_frame_summaries.append({
            "sample_idx": f_i,
            "orig_frame_idx": orig_idx,
            "t_sec": round(t_sec, 2),
            "det_ms": round(det_ms, 1),
            "e2e_ms": round(e2e_ms, 1),
            "reg_status": reg_status,
            "inlier_ratio": round(inlier_ratio, 3),
            "num_detections": len(batch.detections),
            "detections": dets_summary,
        })

    _, peak_py_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = _proc_stats()["rss_mb"]

    mean_det_ms = float(np.mean(latencies_ms)) if latencies_ms else 0.0
    mean_e2e_ms = float(np.mean(e2e_latencies_ms)) if e2e_latencies_ms else 0.0
    processing_fps = 1000.0 / mean_det_ms if mean_det_ms > 0 else 0.0
    e2e_fps = 1000.0 / mean_e2e_ms if mean_e2e_ms > 0 else 0.0

    # Real-time frame drop rate if fed at native capture FPS
    capture_fps = meta["actual_capture_fps"]
    total_native_frames = meta["total_frames"]
    can_process_realtime = int(meta["duration_sec"] * e2e_fps)
    simulated_dropped_frames = max(0, total_native_frames - can_process_realtime)
    simulated_drop_pct = round(100.0 * simulated_dropped_frames / max(1, total_native_frames), 1)

    # Continuity & stability per detected label (appearing in >= 3 sampled frames)
    stability: Dict[str, Any] = {}
    for lbl, entries in class_boxes.items():
        if len(entries) < 3:
            continue
        consec_ious: List[float] = []
        gaps = 0
        for k in range(1, len(entries)):
            prev_i, prev_b = entries[k - 1]
            curr_i, curr_b = entries[k]
            if curr_i - prev_i == 1:
                consec_ious.append(calculate_bbox_iou(prev_b, curr_b))
            else:
                gaps += (curr_i - prev_i - 1)
        confs = class_confs[lbl]
        stability[lbl] = {
            "frames_detected": len(set(i for i, _ in entries)),
            "total_sampled_frames": len(frames),
            "continuity_pct": round(100.0 * len(set(i for i, _ in entries)) / max(1, len(frames)), 1),
            "missed_frame_gaps": gaps,
            "mean_consec_iou": round(float(np.mean(consec_ious)), 4) if consec_ious else None,
            "conf_mean": round(float(np.mean(confs)), 4),
            "conf_std": round(float(np.std(confs)), 4),
        }

    return {
        "detector_name": model_name,
        "sampled_frames": len(frames),
        "sample_hz": sample_hz,
        "mean_det_latency_ms": round(mean_det_ms, 2),
        "p95_det_latency_ms": round(float(np.percentile(latencies_ms, 95)), 2) if latencies_ms else 0.0,
        "mean_e2e_latency_ms": round(mean_e2e_ms, 2),
        "processing_fps_det_only": round(processing_fps, 2),
        "processing_fps_e2e": round(e2e_fps, 2),
        "realtime_capture_fps": round(capture_fps, 2),
        "simulated_realtime_dropped_frames": simulated_dropped_frames,
        "simulated_realtime_drop_pct": simulated_drop_pct,
        "registration_success_rate": f"{reg_success}/{reg_total}",
        "mean_cpu_percent": round(float(np.mean(cpu_samples)), 1) if cpu_samples else None,
        "rss_before_mb": rss_before,
        "rss_after_mb": rss_after,
        "tracemalloc_peak_mb": round(peak_py_bytes / (1024 * 1024), 2),
        "class_detection_counts": class_counts,
        "stability_by_class": stability,
        "sample_timeline": per_frame_summaries,
    }


def main() -> None:
    rec_dir = PROJECT_ROOT / "research_validation/webcam_recordings"
    print("Capturing 32s webcam recording from device 0...")
    meta = capture_webcam_recording(rec_dir, duration_sec=32.0)
    print(f"Captured {meta['total_frames']} frames @ {meta['actual_capture_fps']} fps, res={meta['resolution']}")

    video_path = Path(meta["video_path"])
    comparison = {
        "recording_metadata": meta,
        "models": {},
    }
    for name in ("yolov8m", "yolo26m"):
        print(f"=== Replaying {meta['video_path']} through {name} ===")
        res = replay_recording(name, video_path, meta, sample_hz=1.0)
        comparison["models"][name] = res
        print(
            f"  det_ms={res['mean_det_latency_ms']} e2e_ms={res['mean_e2e_latency_ms']} "
            f"det_fps={res['processing_fps_det_only']} drop_pct={res['simulated_realtime_drop_pct']}% "
            f"reg={res['registration_success_rate']} counts={res['class_detection_counts']}"
        )

    out_json = PROJECT_ROOT / "research_validation/webcam_yolov8m_vs_yolo26m_comparison.json"
    out_json.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print("Saved:", out_json)


if __name__ == "__main__":
    main()
