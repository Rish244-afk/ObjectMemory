import shutil
import tempfile
from pathlib import Path
import sys
sys.path.insert(0, "src")

from objectmemory.api.workspace_manager import WorkspaceManager

scenarios = ["stationary", "movement", "removal", "addition", "occlusion", "return_to_baseline"]
tmp = Path(tempfile.mkdtemp())
try:
    mgr = WorkspaceManager(tmp)
    for sc in scenarios:
        ws_id = f"diag_{sc}"
        mgr.create_workspace(ws_id, f"Diag {sc}")
        img_dir = Path("data/physical_pilot_v2") / sc
        imgs = sorted(img_dir.glob("V0*.jpeg"))
        for idx, img_p in enumerate(imgs, start=1):
            mgr.process_image_observation(
                workspace_id=ws_id,
                image_bytes=img_p.read_bytes(),
                session_name=f"{sc} {img_p.stem}",
                timestamp=f"2026-11-25T10:0{idx}:00Z",
                confidence_threshold=0.25,
                detector_name="yolov8m",
            )
        comp = mgr.compare_sessions(ws_id, "visit_01", "visit_05")
        print(f"\n=== {sc.upper()} (visit_01 -> visit_05) ===")
        print(f"reg={comp['registration_status']} inlier={comp['registration_inlier_ratio']:.3f} ref_dets={comp['ref_detections_count']} q_dets={comp['query_detections_count']} warn={comp['cross_sequence_warning']}")
        for it in comp["items"]:
            print(f"  {it['entity_id']} ({it['label']}, uncertain={it['class_uncertain']}): change={it['status_change']} | {it['prior_state']}->{it['new_state']} | disp={round(it['displacement_px'],1) if it['displacement_px'] is not None else None} | ref_box={it['ref_bbox'] is not None} ref_raw={it['ref_raw_bbox'] is not None} q_box={it['query_bbox'] is not None} q_raw={it['query_raw_bbox'] is not None}")
        if sc == "return_to_baseline":
            comp3 = mgr.compare_sessions(ws_id, "visit_01", "visit_03")
            moved3 = [(it["entity_id"], it["label"], it["status_change"], round(it["displacement_px"], 1) if it["displacement_px"] else None) for it in comp3["items"] if it["status_change"] == "CONFIRMED_MOVED"]
            print(f"  [return_to_baseline visit_01->visit_03 MOVED]: {moved3}")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
