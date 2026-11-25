"""
ETH ASL Lounge Evaluation Script for ObjectMemory (Frozen Baseline).

Executes:
1. Smoke test on observation_01 sample frames.
2. Intra-observation registration & temporal stability test (observation_01 sampled frames).
3. Cross-observation evaluation across observation_01 .. observation_10:
   - Baseline keyframe evaluation (frame_00000.png of each observation).
   - Pose-matched keyframe evaluation (selecting the frame in each observation whose
     T_G_C translation is closest to observation_01/frame_00000.png).
4. Saves machine-readable JSON results and qualitative visual comparisons under
   research_validation/results/eth_asl_lounge/.

Does NOT modify src/objectmemory/, thresholds, or existing benchmark results.
Object-lifecycle metrics (FRR, RR, Movement P/R, IC) are reported as UNAVAILABLE
because ETH ASL Lounge provides no 2D bounding-box or object-ID ground truth.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from objectmemory.detection.detector import OpenCVDNNDetector
from objectmemory.detection.models import Detection
from objectmemory.detection.pipeline import is_bbox_boundary_clipped
from objectmemory.detection.signature import VisualSignatureExtractor
from objectmemory.persistence.repository import SQLiteMemoryStore
from objectmemory.registration.registrar import PlanarRegistrar
from objectmemory.benchmarks.runner import transform_query_bbox_to_reference
from objectmemory.temporal.engine import TemporalMemoryEngine
from objectmemory.temporal.models import TemporalEngineConfig, VisitInput


def run_sequence_eval(
    seq_name: str,
    frame_items: List[Tuple[str, str, Path]],
    detector: OpenCVDNNDetector,
    registrar: PlanarRegistrar,
    sig_ext: VisualSignatureExtractor,
    vis_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    store = SQLiteMemoryStore(":memory:")
    cfg = TemporalEngineConfig()
    engine = TemporalMemoryEngine(store=store, config=cfg, signature_extractor=sig_ext)

    ref_img = None
    per_visit: List[Dict[str, Any]] = []
    status_counts: Dict[str, int] = {}
    class_counts: Dict[str, int] = {}
    inlier_ratios: List[float] = []
    reproj_errors: List[float] = []

    for idx, (obs_id, ts_iso, img_path) in enumerate(frame_items):
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Failed to read {img_path}")
        h_img, w_img = img.shape[:2]
        if ref_img is None:
            ref_img = img
            reg = registrar.register(img, img)
        else:
            reg = registrar.register(ref_img, img)

        st_val = reg.status.value
        status_counts[st_val] = status_counts.get(st_val, 0) + 1
        if reg.success and reg.inlier_ratio is not None:
            inlier_ratios.append(float(reg.inlier_ratio))
        if reg.success and reg.reprojection_stats is not None:
            reproj_errors.append(float(reg.reprojection_stats.mean))

        is_reg_ok = bool(
            reg.success
            and reg.homography is not None
            and (reg.inlier_ratio is None or reg.inlier_ratio >= cfg.min_registration_inlier_ratio)
        )
        raw_dets = detector.detect(img).detections
        ref_w, ref_h = ref_img.shape[1], ref_img.shape[0]
        proc_dets: List[Detection] = []
        for det in raw_dets:
            class_counts[det.label] = class_counts.get(det.label, 0) + 1
            cbox = (
                transform_query_bbox_to_reference(det.raw_bbox, reg.homography, (ref_w, ref_h))
                if (is_reg_ok and reg.homography is not None)
                else None
            )
            sig = sig_ext.extract(img, det.raw_bbox)
            clipped = is_bbox_boundary_clipped(det.raw_bbox, (w_img, h_img), margin=cfg.border_margin_px)
            proc_dets.append(
                Detection(
                    label=det.label,
                    confidence=det.confidence,
                    raw_bbox=det.raw_bbox,
                    canonical_bbox=cbox,
                    signature=sig,
                    metadata={"is_clipped": clipped},
                )
            )

        rep = engine.process_visit(
            VisitInput(
                session_id=obs_id,
                session_name=f"{seq_name}_{obs_id}",
                timestamp=ts_iso,
                image=img,
                registration=reg,
                detections=proc_dets,
            )
        )

        # Save qualitative side-by-side comparison for first 3 cross-obs pairs or failures
        if vis_dir is not None and idx in (1, 2, 5, 9) and ref_img is not None:
            vis_dir.mkdir(parents=True, exist_ok=True)
            vis_q = img.copy()
            for d in raw_dets:
                b = d.raw_bbox
                cv2.rectangle(vis_q, (int(b.x_min), int(b.y_min)), (int(b.x_max), int(b.y_max)), (0, 220, 120), 2)
                cv2.putText(vis_q, f"{d.label} {d.confidence:.2f}", (int(b.x_min), max(15, int(b.y_min)-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 120), 2)
            banner = f"{obs_id} | reg={st_val} | inliers={reg.num_inliers} ({reg.inlier_ratio:.2f})"
            cv2.putText(vis_q, banner, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)
            pair = np.hstack([cv2.resize(ref_img, (640, 360)), cv2.resize(vis_q, (640, 360))])
            cv2.imwrite(str(vis_dir / f"{seq_name}_{obs_id}.jpg"), pair, [cv2.IMWRITE_JPEG_QUALITY, 90])

        entities = store.list_entities()
        state_hist: Dict[str, int] = {}
        for e in entities:
            state_hist[e.current_state.value] = state_hist.get(e.current_state.value, 0) + 1

        per_visit.append({
            "observation_id": obs_id,
            "image_path": str(img_path.name),
            "reg_success": bool(reg.success),
            "reg_status": st_val,
            "num_inliers": int(reg.num_inliers),
            "inlier_ratio": round(float(reg.inlier_ratio), 4) if reg.inlier_ratio is not None else None,
            "reproj_mean_px": round(float(reg.reprojection_stats.mean), 3) if reg.reprojection_stats else None,
            "detections_count": len(raw_dets),
            "detected_classes": [d.label for d in raw_dets],
            "active_entities_total": len(entities),
            "entity_state_counts": state_hist,
            "state_transitions_count": rep.state_transitions_count,
        })

    n_transitions = max(0, len(frame_items) - 1)
    succ_transitions = sum(1 for v in per_visit[1:] if v["reg_success"])
    return {
        "sequence_name": seq_name,
        "frames_evaluated": len(frame_items),
        "transitions_attempted": n_transitions,
        "transitions_succeeded": succ_transitions,
        "registration_success_rate": (succ_transitions / n_transitions) if n_transitions > 0 else None,
        "registration_status_counts": status_counts,
        "mean_inlier_ratio_when_success": (sum(inlier_ratios) / len(inlier_ratios)) if inlier_ratios else None,
        "mean_reproj_error_px_when_success": (sum(reproj_errors) / len(reproj_errors)) if reproj_errors else None,
        "total_detections": sum(class_counts.values()),
        "detected_class_counts": class_counts,
        "unavailable_metrics": [
            "false_removal_rate",
            "false_discovery_proportion",
            "removal_recall",
            "movement_precision",
            "movement_recall",
            "identity_consistency",
            "overall_state_accuracy",
        ],
        "unavailable_reason": "ETH ASL Lounge provides 3D meshes and poses only; no 2D bounding boxes, persistent object IDs, or lifecycle labels.",
        "per_visit": per_visit,
    }


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    man_path = root / "research_validation/ETH_ASL_DATASET_MANIFEST.json"
    if not man_path.exists():
        raise FileNotFoundError(f"Missing {man_path}; run prepare_eth_asl_lounge.py first.")
    manifest = json.loads(man_path.read_text(encoding="utf-8"))

    out_dir = root / "research_validation/results/eth_asl_lounge"
    vis_dir = out_dir / "visual_comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = OpenCVDNNDetector(weights_path=str(root / "models/yolov8n.onnx"))
    registrar = PlanarRegistrar()
    sig_ext = VisualSignatureExtractor()

    # 1. Smoke test on observation_01 (first 2 frames)
    obs01_dir = root / manifest["observations"][0]["extracted_dir"]
    smoke_items = [
        ("smoke_00", "2026-11-25T00:00:00Z", obs01_dir / "rgb/frame_00000.png"),
        ("smoke_01", "2026-11-25T00:00:01Z", obs01_dir / "rgb/frame_00001.png"),
    ]
    smoke_res = run_sequence_eval("smoke_test", smoke_items, detector, registrar, sig_ext)

    # 2. Intra-observation test (10 frames spaced by 150 frames in observation_01)
    obs01_meta = json.loads((obs01_dir / "metadata/frames_and_poses.json").read_text(encoding="utf-8"))
    intra_items = []
    for k in range(0, min(len(obs01_meta["frames"]), 1500), 150):
        finfo = obs01_meta["frames"][k]
        intra_items.append((f"obs01_f{k:05d}", f"2026-11-25T00:{k//60:02d}:{k%60:02d}Z", obs01_dir / finfo["rel_path"]))
    intra_res = run_sequence_eval("intra_observation_01", intra_items, detector, registrar, sig_ext, vis_dir=vis_dir)

    # 3. Cross-observation evaluation: first frame of each observation_01 .. observation_10
    cross_first_items = []
    for idx, obs in enumerate(manifest["observations"]):
        odir = root / obs["extracted_dir"]
        cross_first_items.append((obs["observation_id"], f"2026-11-25T01:{idx:02d}:00Z", odir / "rgb/frame_00000.png"))
    cross_first_res = run_sequence_eval("cross_obs_first_frames", cross_first_items, detector, registrar, sig_ext, vis_dir=vis_dir)

    # 4. Cross-observation pose-matched evaluation (nearest T_G_C translation to obs01 frame_00000)
    ref_t = np.array(obs01_meta["T_G_C"][0]["translation"])
    cross_pose_items = []
    for idx, obs in enumerate(manifest["observations"]):
        odir = root / obs["extracted_dir"]
        ometa = json.loads((odir / "metadata/frames_and_poses.json").read_text(encoding="utf-8"))
        poses = ometa["T_G_C"]
        frames = ometa["frames"]
        best_f_idx = 0
        if poses and frames:
            best_dist = float("inf")
            limit = min(len(poses), len(frames))
            for k in range(limit):
                d = float(np.linalg.norm(np.array(poses[k]["translation"]) - ref_t))
                if d < best_dist:
                    best_dist = d
                    best_f_idx = k
        cross_pose_items.append((obs["observation_id"], f"2026-11-25T02:{idx:02d}:00Z", odir / frames[best_f_idx]["rel_path"]))
    cross_pose_res = run_sequence_eval("cross_obs_pose_matched", cross_pose_items, detector, registrar, sig_ext, vis_dir=vis_dir)

    full_out = {
        "dataset": "ETH_ASL_Change_Detection_Lounge",
        "smoke_test": smoke_res,
        "intra_observation_01": intra_res,
        "cross_observation_first_frames": cross_first_res,
        "cross_observation_pose_matched": cross_pose_res,
    }
    out_json = out_dir / "eth_asl_lounge_evaluation_results.json"
    out_json.write_text(json.dumps(full_out, indent=2), encoding="utf-8")
    print("Saved ETH ASL Lounge evaluation to", out_json)
    print("Smoke test reg success:", smoke_res["registration_success_rate"])
    print("Intra-obs01 reg success:", intra_res["registration_success_rate"], f"({intra_res['transitions_succeeded']}/{intra_res['transitions_attempted']})")
    print("Cross-obs (first frames) reg success:", cross_first_res["registration_success_rate"], f"({cross_first_res['transitions_succeeded']}/{cross_first_res['transitions_attempted']})")
    print("Cross-obs (pose-matched) reg success:", cross_pose_res["registration_success_rate"], f"({cross_pose_res['transitions_succeeded']}/{cross_pose_res['transitions_attempted']})")


if __name__ == "__main__":
    main()
