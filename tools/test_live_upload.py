import json
import urllib.request
from pathlib import Path

def run_live_audit():
    # 1. Health check
    with urllib.request.urlopen("http://127.0.0.1:8000/api/health") as resp:
        health = json.loads(resp.read().decode())
    print("Health:", health["status"], "| Detector available:", health["detector_available"])

    # 2. Query Detectors endpoint
    with urllib.request.urlopen("http://127.0.0.1:8000/api/detectors") as resp:
        detectors_data = json.loads(resp.read().decode())
    print(f"Registered Detectors ({len(detectors_data['detectors'])}):", [d["name"] for d in detectors_data["detectors"]])

    # 3. Create or reset workspace
    ws_id = "live_audit_bench"
    req_ws = urllib.request.Request(
        "http://127.0.0.1:8000/api/workspaces",
        data=json.dumps({"workspace_id": ws_id, "name": "Live Audit Bench"}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req_ws) as resp:
            ws = json.loads(resp.read().decode())
            print("Created workspace:", ws["workspace"]["workspace_id"])
    except urllib.error.HTTPError as err:
        print("Workspace status:", err.code, "(already exists or created)")

    # 4. Upload Visit 1 (Reference Image)
    img_path = Path("data/real_workbench/visits/visit_01.png")
    assert img_path.exists(), f"Image {img_path} must exist"
    img_bytes = img_path.read_bytes()

    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{img_path.name}"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    ).encode("utf-8") + img_bytes + (
        f"\r\n--{boundary}\r\n"
        f'Content-Disposition: form-data; name="session_name"\r\n\r\nVisit 1 Reference\r\n'
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="confidence_threshold"\r\n\r\n0.25\r\n'
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="detector_name"\r\n\r\nyolov8n\r\n'
        f"--{boundary}--\r\n"
    ).encode("utf-8")

    upload_req = urllib.request.Request(
        f"http://127.0.0.1:8000/api/workspaces/{ws_id}/observations",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST"
    )

    with urllib.request.urlopen(upload_req) as resp:
        obs1 = json.loads(resp.read().decode())

    print("\n--- Observation 1 (Visit 1 Reference) ---")
    print("Session ID:", obs1["session_id"])
    print("Is Reference Frame:", obs1["is_reference_frame"])
    print("Registration Status:", obs1["registration"]["status"])
    print(f"Raw Detections Count: {len(obs1['detections'])}")
    for d in obs1["detections"]:
        raw = d["raw_bbox"]
        print(f"  * {d['label']} (conf={d['confidence']:.3f}) bbox=[{raw['x_min']:.1f}, {raw['y_min']:.1f}, {raw['x_max']:.1f}, {raw['y_max']:.1f}]")
    print("Active Beliefs Count:", obs1["total_active_entities"])
    print("Total Latency:", f"{obs1['total_latency_ms']:.1f}ms")

    # 5. Upload Visit 2
    img2_path = Path("data/real_workbench/visits/visit_02.png")
    assert img2_path.exists(), f"Image {img2_path} must exist"
    img2_bytes = img2_path.read_bytes()

    body2 = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{img2_path.name}"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    ).encode("utf-8") + img2_bytes + (
        f"\r\n--{boundary}\r\n"
        f'Content-Disposition: form-data; name="session_name"\r\n\r\nVisit 2 Query\r\n'
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="confidence_threshold"\r\n\r\n0.25\r\n'
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="detector_name"\r\n\r\nyolov8n\r\n'
        f"--{boundary}--\r\n"
    ).encode("utf-8")

    upload_req2 = urllib.request.Request(
        f"http://127.0.0.1:8000/api/workspaces/{ws_id}/observations",
        data=body2,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST"
    )

    with urllib.request.urlopen(upload_req2) as resp:
        obs2 = json.loads(resp.read().decode())

    print("\n--- Observation 2 (Visit 2 Query) ---")
    print("Session ID:", obs2["session_id"])
    print("Is Reference Frame:", obs2["is_reference_frame"])
    print(f"Registration: status={obs2['registration']['status']}, inlier_ratio={obs2['registration']['inlier_ratio']:.3f}, inliers={obs2['registration']['inliers_count']}")
    print(f"Raw Detections Count: {len(obs2['detections'])}")
    for d in obs2["detections"]:
        raw = d["raw_bbox"]
        print(f"  * {d['label']} (conf={d['confidence']:.3f}) bbox=[{raw['x_min']:.1f}, {raw['y_min']:.1f}, {raw['x_max']:.1f}, {raw['y_max']:.1f}]")
    print("Belief Updates Count:", len(obs2["belief_updates"]))
    for bu in obs2["belief_updates"]:
        print(f"  * Entity {bu['entity_id']} ({bu['label']}): {bu['prior_state']} -> {bu['new_state']} (transition={bu['transition_occurred']}) | {bu['explanation']}")
    print("Events Count:", len(obs2["events"]))
    print("Total Active Entities:", obs2["total_active_entities"])
    print("Total Latency:", f"{obs2['total_latency_ms']:.1f}ms")

    # 6. Verify Session Detail Inspection (with label verification)
    detail_url = f"http://127.0.0.1:8000/api/workspaces/{ws_id}/observations/{obs1['session_id']}"
    with urllib.request.urlopen(detail_url) as resp:
        sess_detail = json.loads(resp.read().decode())
    print("\n--- Session 1 Detail Inspection ---")
    print("Session Name:", sess_detail["session"]["session_name"])
    print("Observations Count:", len(sess_detail["observations"]))
    for o in sess_detail["observations"]:
        print(f"  * Observation {o['observation_id']}: entity_id={o['entity_id']}, label='{o.get('label')}', conf={o['detector_confidence']}")

if __name__ == "__main__":
    run_live_audit()
