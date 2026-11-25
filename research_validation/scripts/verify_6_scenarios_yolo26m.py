#!/usr/bin/env python3
"""
Verifies all 6 physical-pilot scenarios (stationary, movement, removal, addition,
occlusion, return_to_baseline) through the full WorkspaceManager pipeline with YOLO26m,
checking explicit movement_status reporting (MOVED, NOT_MOVED, REMOVED, ADDED, NOT_DETECTED).
"""

from __future__ import annotations

import json
import shutil
import tempfile
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from objectmemory.api.workspace_manager import WorkspaceManager

SCENARIOS = ["stationary", "movement", "removal", "addition", "occlusion", "return_to_baseline"]


def main() -> None:
    tmp = tempfile.mkdtemp()
    report = {}
    try:
        mgr = WorkspaceManager(base_dir=Path(tmp))
        for sc in SCENARIOS:
            ws = f"verify26_{sc}"
            mgr.create_workspace(ws, ws)
            sc_dir = PROJECT_ROOT / "data/physical_pilot_v2" / sc
            for i in range(1, 6):
                img_path = sc_dir / f"V0{i}.jpeg"
                mgr.process_image_observation(
                    ws,
                    img_path.read_bytes(),
                    session_name=f"V0{i}",
                    timestamp=f"2026-11-25T10:0{i}:00Z",
                    detector_name="yolo26m",
                )
            pairs = [("visit_01", "visit_05")]
            if sc == "return_to_baseline":
                pairs.insert(0, ("visit_01", "visit_03"))
            report[sc] = {}
            for r_id, q_id in pairs:
                comp = mgr.compare_sessions(ws, r_id, q_id)
                items = [
                    {
                        "entity_id": it["entity_id"],
                        "label": it["label"],
                        "movement_status": it["movement_status"],
                        "status_change": it["status_change"],
                        "evidence_source": it["evidence_source"],
                        "displacement_px": round(it["displacement_px"], 1) if it["displacement_px"] is not None else None,
                    }
                    for it in comp["items"]
                ]
                report[sc][f"{r_id}->{q_id}"] = {
                    "registration_status": comp["registration_status"],
                    "ref_dets": comp["ref_detections_count"],
                    "q_dets": comp["query_detections_count"],
                    "items": items,
                }
                print(f"=== {sc.upper()} ({r_id}->{q_id}) reg={comp['registration_status']} ===")
                for it in items:
                    print(f"  {it['entity_id']} ({it['label']}): {it['movement_status']} [{it['status_change']}] disp={it['displacement_px']} src={it['evidence_source']}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    out_path = PROJECT_ROOT / "research_validation/yolo26m_6_scenarios_reporting.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
