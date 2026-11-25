"""
Final Experimental Completion Runner (Tasks 2, 3, and 4).

Strictly read-only with respect to:
- src/objectmemory/ (no source or threshold modifications)
- research_validation/results/benchmarks/*.json (no overwriting frozen results)

Writes all new outputs strictly to:
  research_validation/results/final_completion/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from objectmemory.benchmarks.config_factory import BenchmarkConfig, BenchmarkConfigFactory, ConfigurationSupportStatus
from objectmemory.benchmarks.manifest import DatasetManifest, ManifestLoader
from objectmemory.benchmarks.runner import BenchmarkRunner
from objectmemory.detection.detector import OpenCVDNNDetector
from objectmemory.domain.states import EpistemicState
from objectmemory.temporal.models import TemporalEngineConfig


def compute_task2_missing_metrics(
    runner: BenchmarkRunner,
    manifest: DatasetManifest,
    dataset_label: str,
) -> Dict[str, Any]:
    """
    Task 2: Post-processing evaluation on frozen B3 configuration to extract:
    1. Standalone Movement Precision and Recall (visit-level & event-level).
    2. Physical Removal Confirmation Latency (from PENDING_REMOVAL physical disappearance onset).
    3. Synthetic S02 return_to_baseline HRA verification.
    """
    b3_cfg = BenchmarkConfigFactory.create_config("B3")
    seq_details: Dict[str, Any] = {}

    mov_tp_visits = 0
    mov_fp_visits = 0
    mov_fn_visits = 0

    mov_events_gt = 0
    mov_events_tp = 0

    removal_latencies: List[Dict[str, Any]] = []
    hra_checks: List[Dict[str, Any]] = []

    for seq in manifest.sequences:
        store_dummy = None
        # Execute via internal _execute_temporal_sequence? Wait: we want exact per-visit snapshots
        # Let's run runner._execute_temporal_sequence AND also inspect per-visit predictions cleanly
        # by re-running the exact same sequence loop in read-only mode so we can inspect every visit!
        from objectmemory.detection.pipeline import is_bbox_boundary_clipped
        from objectmemory.benchmarks.runner import transform_query_bbox_to_reference
        from objectmemory.persistence.repository import SQLiteMemoryStore
        from objectmemory.temporal.engine import TemporalMemoryEngine
        from objectmemory.temporal.models import VisitInput
        from objectmemory.baseline.comparator import calculate_bbox_iou

        store = SQLiteMemoryStore(":memory:")
        temp_cfg = b3_cfg.get_isolated_temporal_config()
        engine = TemporalMemoryEngine(
            store=store,
            config=temp_cfg,
            signature_extractor=runner.signature_extractor,
        )
        detector = runner._get_detector()
        canon_ref_img = None
        visit_entity_states: Dict[int, Dict[str, str]] = {}
        obs_frame_matches: Dict[int, Dict[str, str]] = {}
        v1_entities = []

        for v_idx, obs in enumerate(seq.observations, start=1):
            img = runner._load_observation_image(obs)
            h_img, w_img = img.shape[:2]
            if canon_ref_img is None:
                canon_ref_img = img
                reg = runner.registrar.register(img, img)
            else:
                reg = runner.registrar.register(canon_ref_img, img)

            is_reg_ok = bool(
                reg.success
                and reg.homography is not None
                and (reg.inlier_ratio is None or reg.inlier_ratio >= temp_cfg.min_registration_inlier_ratio)
            )
            raw_dets = detector.detect(img).detections
            ref_w, ref_h = canon_ref_img.shape[1], canon_ref_img.shape[0]
            processed_dets = []
            for det in raw_dets:
                cbox = (
                    transform_query_bbox_to_reference(det.raw_bbox, reg.homography, (ref_w, ref_h))
                    if (is_reg_ok and reg.homography is not None)
                    else None
                )
                sig = runner.signature_extractor.extract(img, det.raw_bbox)
                clipped = is_bbox_boundary_clipped(det.raw_bbox, (w_img, h_img), margin=temp_cfg.border_margin_px)
                from objectmemory.detection.models import Detection
                processed_dets.append(
                    Detection(
                        label=det.label,
                        confidence=det.confidence,
                        raw_bbox=det.raw_bbox,
                        canonical_bbox=cbox,
                        signature=sig,
                        metadata={"is_clipped": clipped},
                    )
                )

            v_in = VisitInput(
                session_id=obs.observation_id,
                session_name=f"{seq.sequence_id}_{obs.observation_id}",
                timestamp=obs.timestamp,
                image=img,
                registration=reg,
                detections=processed_dets,
                metadata={"sequence_id": seq.sequence_id},
            )
            rep = engine.process_visit(v_in)

            # Frame matches for HRA
            cur_matches: Dict[str, str] = {}
            active_anns = [a for a in obs.objects if a.bounding_box is not None]
            for ann in active_anns:
                best_idx, best_iou = None, 0.0
                for d_idx, d in enumerate(processed_dets):
                    if d.label.lower() == ann.class_name.lower():
                        iou = calculate_bbox_iou(ann.bounding_box, d.raw_bbox)
                        if iou > best_iou:
                            best_iou, best_idx = iou, d_idx
                if best_idx is not None and best_iou >= 0.50:
                    md = processed_dets[best_idx]
                    for ent in store.list_entities():
                        if ent.label.lower() == ann.class_name.lower() and ent.canonical_bbox is not None:
                            if md.canonical_bbox is not None and calculate_bbox_iou(md.canonical_bbox, ent.canonical_bbox) >= 0.30:
                                cur_matches[ann.object_id] = ent.entity_id
                                break
                            elif ent.last_seen_session_id == obs.observation_id:
                                cur_matches[ann.object_id] = ent.entity_id
                                break
            obs_frame_matches[v_idx] = cur_matches

            # Capture exact entity state snapshot
            snap_states = {}
            for ent in store.list_entities():
                snap_states[ent.entity_id] = ent.current_state.value
            visit_entity_states[v_idx] = snap_states

        gt_to_sys, _ = runner._compute_stable_bipartite_mapping(seq, [], store.list_entities())

        # Analyze Movement P/R, Removal Physical Latency, and S02 Return-to-Baseline
        seq_mov_events: Dict[str, Dict[str, bool]] = {}
        rem_onset_v: Dict[str, int] = {}
        rem_confirm_v: Dict[str, int] = {}

        for v_idx, obs in enumerate(seq.observations, start=1):
            st_map = visit_entity_states[v_idx]
            for ann in obs.objects:
                gid = ann.object_id
                sys_id = gt_to_sys.get(gid)
                pred_st = st_map.get(sys_id, "UNKNOWN") if sys_id else "UNKNOWN"
                gt_st = ann.object_state

                # Movement Visit-level P/R
                if gt_st == EpistemicState.CONFIRMED_MOVED.value:
                    if gid not in seq_mov_events:
                        seq_mov_events[gid] = {"gt": True, "tp": False}
                    if pred_st == EpistemicState.CONFIRMED_MOVED.value:
                        mov_tp_visits += 1
                        seq_mov_events[gid]["tp"] = True
                    else:
                        mov_fn_visits += 1
                elif gt_st in (EpistemicState.CONFIRMED_PRESENT.value, "STATIONARY"):
                    if pred_st == EpistemicState.CONFIRMED_MOVED.value:
                        mov_fp_visits += 1

                # Physical Removal Latency: onset = first PENDING_REMOVAL or CONFIRMED_REMOVED
                if gt_st in (EpistemicState.PENDING_REMOVAL.value, EpistemicState.CONFIRMED_REMOVED.value):
                    if gid not in rem_onset_v:
                        rem_onset_v[gid] = v_idx
                if pred_st == EpistemicState.CONFIRMED_REMOVED.value and gid in rem_onset_v and gid not in rem_confirm_v:
                    rem_confirm_v[gid] = v_idx

                # Explicit check for S02 return_to_baseline (cup_01 at visit 5) or annotated return
                is_return = (
                    "return" in (ann.evidence_notes or "").lower()
                    or ann.event_id == "return_to_baseline"
                    or (seq.sequence_id == "S02_relocation_return" and v_idx == 5 and gid == "cup_01")
                )
                if is_return and sys_id is not None:
                    matched_sys = obs_frame_matches.get(v_idx, {}).get(gid)
                    hra_checks.append({
                        "sequence_id": seq.sequence_id,
                        "visit_idx": v_idx,
                        "object_id": gid,
                        "expected_sys_id": sys_id,
                        "matched_sys_id": matched_sys,
                        "reassociated": (matched_sys == sys_id),
                        "predicted_state": pred_st,
                        "gt_state": gt_st,
                    })

        for gid, info in seq_mov_events.items():
            mov_events_gt += 1
            if info["tp"]:
                mov_events_tp += 1

        for gid, onset_v in rem_onset_v.items():
            conf_v = rem_confirm_v.get(gid)
            removal_latencies.append({
                "sequence_id": seq.sequence_id,
                "object_id": gid,
                "physical_disappearance_visit": onset_v,
                "system_confirmed_visit": conf_v,
                "inclusive_visits_to_confirm": (conf_v - onset_v + 1) if conf_v is not None else None,
                "additional_visits_after_disappearance": (conf_v - onset_v) if conf_v is not None else None,
            })

    mov_p = mov_tp_visits / (mov_tp_visits + mov_fp_visits) if (mov_tp_visits + mov_fp_visits) > 0 else None
    mov_r = mov_tp_visits / (mov_tp_visits + mov_fn_visits) if (mov_tp_visits + mov_fn_visits) > 0 else None
    mov_f1 = (2 * mov_p * mov_r / (mov_p + mov_r)) if (mov_p and mov_r and (mov_p + mov_r) > 0) else None

    confirmed_lats = [r["inclusive_visits_to_confirm"] for r in removal_latencies if r["inclusive_visits_to_confirm"] is not None]
    mean_inclusive_lat = sum(confirmed_lats) / len(confirmed_lats) if confirmed_lats else None

    hra_num = sum(1 for h in hra_checks if h["reassociated"])
    hra_den = len(hra_checks)

    return {
        "dataset": dataset_label,
        "movement_metrics": {
            "visit_level_tp": mov_tp_visits,
            "visit_level_fp": mov_fp_visits,
            "visit_level_fn": mov_fn_visits,
            "movement_precision_visits": mov_p,
            "movement_recall_visits": mov_r,
            "movement_f1_visits": mov_f1,
            "event_level_tp": mov_events_tp,
            "event_level_gt": mov_events_gt,
            "movement_recall_events": (mov_events_tp / mov_events_gt) if mov_events_gt > 0 else None,
        },
        "physical_removal_latency": {
            "events": removal_latencies,
            "mean_inclusive_visits_from_disappearance": mean_inclusive_lat,
            "mean_additional_visits_after_disappearance": (mean_inclusive_lat - 1.0) if mean_inclusive_lat is not None else None,
        },
        "historical_retrieval_verification": {
            "events": hra_checks,
            "hra_numerator": hra_num,
            "hra_denominator": hra_den,
            "hra_accuracy": (hra_num / hra_den) if hra_den > 0 else None,
        },
    }


def run_single_param_sensitivity(
    runner: BenchmarkRunner,
    manifest: DatasetManifest,
    param_name: str,
    values: List[float],
) -> List[Dict[str, Any]]:
    """
    Task 3: Evaluates changing exactly one TemporalEngineConfig parameter at a time
    in an isolated BenchmarkConfig without modifying production defaults.
    """
    rows: List[Dict[str, Any]] = []
    for val in values:
        kwargs = {param_name: val}
        temp_cfg = TemporalEngineConfig(**kwargs)
        b_cfg = BenchmarkConfig(
            config_id=f"SENS_{param_name}_{val}",
            name=f"Sensitivity {param_name}={val}",
            description=f"Single-parameter sweep: {param_name}={val}",
            support_status=ConfigurationSupportStatus.SUPPORTED,
            pipeline_type="FULL_TEMPORAL_MEMORY",
            temporal_config=temp_cfg,
            enabled_safeguards=("sensitivity_sweep",),
            disabled_safeguards=(),
            notes="Isolated read-only sensitivity sweep.",
            effective_parameters={param_name: val},
        )
        res = runner.run_config(manifest, b_cfg)
        m = res.metrics
        rows.append({
            "parameter": param_name,
            "value": val,
            "is_b3_default": (val == getattr(TemporalEngineConfig(), param_name)),
            "overall_state_accuracy": m.overall_state_accuracy,
            "false_removal_rate": m.false_removal_rate,
            "false_discovery_proportion": m.false_discovery_proportion,
            "removal_recall_visits": m.removal_recall,
            "event_removal_recall": m.event_removal_recall,
            "confirmation_rate": m.confirmation_rate,
            "unsupported_movement_rate": m.unsupported_movement_rate,
            "identity_consistency": m.identity_consistency,
            "fragmented_identities": m.fragmented_identities,
        })
    return rows


def run_task4_a5_ambiguous_distractor(runner: BenchmarkRunner, out_dir: Path) -> Dict[str, Any]:
    """
    Task 4: Constructs an isolated 3-visit synthetic ambiguous-distractor benchmark
    using two identical cup cutouts from synthetic_workbench_v2/assets/cup.png
    to test whether A5 (ambiguity_margin=0.0) differs from B3 (ambiguity_margin=0.12).
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from generate_synthetic_workbench_v2 import generate_wood_tabletop, composite_asset

    ds_dir = out_dir / "a5_ambiguous_distractor_dataset"
    img_dir = ds_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    cup_rgba = cv2.imread("data/synthetic_workbench_v2/assets/cup.png", cv2.IMREAD_UNCHANGED)
    if cup_rgba is None:
        return {"status": "BLOCKED", "reason": "cup.png asset not found"}

    # Visit 1: Two identical cups separated by 180px (x=200 and x=380, y=260)
    # Visit 2: cup_01 removed; cup_02 shifted slightly left to x=290 (equidistant between old cup_01 and cup_02)
    # Visit 3: cup_02 stays at x=290
    observations = []
    placements = [
        (1, [("cup_01", 180, 260, "CONFIRMED_PRESENT", "CLEARLY_VISIBLE"), ("cup_02", 420, 260, "CONFIRMED_PRESENT", "CLEARLY_VISIBLE")]),
        (2, [("cup_01", None, None, "PENDING_REMOVAL", "ABSENT"), ("cup_02", 300, 260, "CONFIRMED_MOVED", "CLEARLY_VISIBLE")]),
        (3, [("cup_01", None, None, "CONFIRMED_REMOVED", "ABSENT"), ("cup_02", 300, 260, "CONFIRMED_PRESENT", "CLEARLY_VISIBLE")]),
    ]

    for v_idx, objs in placements:
        canvas = generate_wood_tabletop(width=1080, height=720, seed=900 + v_idx)
        gt_objs = []
        ts = f"2026-11-25T12:00:{v_idx*10:02d}Z"
        for oid, px, py, st, vis in objs:
            if px is not None and py is not None:
                canvas, bbox = composite_asset(canvas, cup_rgba, px, py)
            else:
                bbox = None
            gt_objs.append({
                "object_id": oid,
                "class_name": "cup",
                "bounding_box": bbox,
                "visibility_status": vis,
                "object_state": st,
                "timestamp": ts,
            })
        img_rel = f"images/visit_{v_idx:02d}.png"
        cv2.imwrite(str(ds_dir / img_rel), canvas)
        observations.append({
            "observation_id": f"visit_{v_idx:02d}",
            "timestamp": ts,
            "image_path": img_rel,
            "objects": gt_objs,
        })

    manifest_dict = {
        "manifest_version": "2.0",
        "dataset_name": "a5_ambiguous_distractor_probe",
        "provenance": "SYNTHETIC_GENERATED",
        "sequences": [{
            "sequence_id": "A5_ambiguous_cup_pair",
            "environment_id": "synthetic_ambiguity_probe",
            "dataset_name": "a5_ambiguous_distractor_probe",
            "observations": observations,
        }],
    }
    manifest_path = ds_dir / "manifest_v2.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_dict, f, indent=2)

    manifest = ManifestLoader.load(manifest_path, check_files=True)
    b3_res = runner.run_config(manifest, BenchmarkConfigFactory.create_config("B3"))
    a5_res = runner.run_config(manifest, BenchmarkConfigFactory.create_config("A5"))

    return {
        "status": "EXECUTED",
        "manifest_path": str(manifest_path),
        "B3": {
            "overall_state_accuracy": b3_res.metrics.overall_state_accuracy,
            "identity_consistency": b3_res.metrics.identity_consistency,
            "id_switches": b3_res.metrics.id_switches,
            "false_removal_rate": b3_res.metrics.false_removal_rate,
            "spurious_system_tracks": b3_res.metrics.spurious_system_tracks,
        },
        "A5": {
            "overall_state_accuracy": a5_res.metrics.overall_state_accuracy,
            "identity_consistency": a5_res.metrics.identity_consistency,
            "id_switches": a5_res.metrics.id_switches,
            "false_removal_rate": a5_res.metrics.false_removal_rate,
            "spurious_system_tracks": a5_res.metrics.spurious_system_tracks,
        },
    }


def main() -> None:
    out_dir = Path("research_validation/results/final_completion")
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = OpenCVDNNDetector(weights_path="models/yolov8n.onnx")
    runner = BenchmarkRunner(detector=detector, allow_partial=True)

    synth_manifest = ManifestLoader.load(Path("data/synthetic_workbench_v2/manifest_v2.json"), check_files=True)
    pilot_manifest = ManifestLoader.load(Path("data/real_workbench/manifest_v2.json"), check_files=True)

    print("Running Task 2: Missing Metrics...")
    task2 = {
        "synthetic_workbench_v2": compute_task2_missing_metrics(runner, synth_manifest, "synthetic_workbench_v2"),
        "desk_pilot_01": compute_task2_missing_metrics(runner, pilot_manifest, "desk_pilot_01"),
    }
    with open(out_dir / "task2_missing_metrics.json", "w", encoding="utf-8") as f:
        json.dump(task2, f, indent=2)

    print("Running Task 3: Sensitivity Analysis...")
    vac_vals = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.85]
    mov_vals = [15.0, 25.0, 35.0, 50.0, 65.0, 100.0]
    task3 = {
        "synthetic_workbench_v2": {
            "vacancy_similarity_max": run_single_param_sensitivity(runner, synth_manifest, "vacancy_similarity_max", vac_vals),
            "movement_displacement_threshold": run_single_param_sensitivity(runner, synth_manifest, "movement_displacement_threshold", mov_vals),
        },
        "desk_pilot_01": {
            "vacancy_similarity_max": run_single_param_sensitivity(runner, pilot_manifest, "vacancy_similarity_max", vac_vals),
            "movement_displacement_threshold": run_single_param_sensitivity(runner, pilot_manifest, "movement_displacement_threshold", mov_vals),
        },
    }
    with open(out_dir / "task3_sensitivity_analysis.json", "w", encoding="utf-8") as f:
        json.dump(task3, f, indent=2)

    print("Running Task 4: A5 Ambiguous-Distractor Probe...")
    task4 = run_task4_a5_ambiguous_distractor(runner, out_dir)
    with open(out_dir / "task4_a5_association_probe.json", "w", encoding="utf-8") as f:
        json.dump(task4, f, indent=2)

    print("Completed Tasks 2, 3, and 4! Outputs written to", out_dir)


if __name__ == "__main__":
    main()
