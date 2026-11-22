"""
Integration verification script for ObjectMemory Studio API.
"""

import sys
from pathlib import Path
import requests

BASE = 'http://127.0.0.1:8000'

def main():
    print("Testing live FastAPI backend at", BASE)

    # 1. Health
    r = requests.get(f'{BASE}/api/health')
    assert r.status_code == 200, f"Health check failed: {r.text}"
    health_data = r.json()
    print("[OK] 1. Health check OK:", health_data['status'], "| Detector:", health_data['detector_available'])

    # 2. Workspace creation
    ws_payload = {
        'workspace_id': 'studio_demo',
        'name': 'Studio Workbench Demo',
        'description': 'Integration verification workspace'
    }
    r = requests.post(f'{BASE}/api/workspaces', json=ws_payload)
    if r.status_code == 201:
        print("[OK] 2. Workspace created:", r.json()['workspace']['name'])
    elif r.status_code == 400:
        print("[OK] 2. Workspace already exists, reusing.")
    else:
        raise AssertionError(f"Workspace creation error: {r.text}")

    # 3. Ingest visit 1
    p1 = Path('data/real_workbench/visits/visit_01.png')
    assert p1.exists(), f"File {p1} not found"
    with open(p1, 'rb') as f:
        r = requests.post(
            f'{BASE}/api/workspaces/studio_demo/observations',
            files={'image': ('visit_01.png', f, 'image/png')},
            data={'session_name': 'Baseline Visit 1', 'timestamp': '2026-10-20T10:00:00Z', 'confidence_threshold': 0.35}
        )
    assert r.status_code == 201, f"Visit 1 ingest failed: {r.text}"
    v1_data = r.json()
    print(f"[OK] 3. Ingest Visit 1 OK: session={v1_data['session_id']}, is_ref={v1_data['is_reference_frame']}, detections={len(v1_data['detections'])}, entities={v1_data['total_active_entities']}")

    # 4. Ingest visit 2
    p2 = Path('data/real_workbench/visits/visit_02.png')
    assert p2.exists(), f"File {p2} not found"
    with open(p2, 'rb') as f:
        r = requests.post(
            f'{BASE}/api/workspaces/studio_demo/observations',
            files={'image': ('visit_02.png', f, 'image/png')},
            data={'session_name': 'Follow-up Visit 2', 'timestamp': '2026-10-20T11:00:00Z', 'confidence_threshold': 0.35}
        )
    assert r.status_code == 201, f"Visit 2 ingest failed: {r.text}"
    v2_data = r.json()
    inlier_pct = v2_data['registration']['inlier_ratio'] * 100
    print(f"[OK] 4. Ingest Visit 2 OK: session={v2_data['session_id']}, reg_inliers={inlier_pct:.1f}%, updates={len(v2_data['belief_updates'])}")

    # 5. Scene Comparison
    r = requests.get(f'{BASE}/api/workspaces/studio_demo/compare')
    assert r.status_code == 200, f"Comparison failed: {r.text}"
    comp = r.json()
    print(f"[OK] 5. Scene Comparison OK: compared {comp['reference_session_id']} vs {comp['query_session_id']} ({len(comp['items'])} items)")

    # 6. Entities
    r = requests.get(f'{BASE}/api/workspaces/studio_demo/entities')
    assert r.status_code == 200, f"Entities list failed: {r.text}"
    ents = r.json()
    print(f"[OK] 6. Entities List OK: {len(ents)} persistent entities found")
    first_ent_id = ents[0]['entity_id']

    # 7. Entity detail
    r = requests.get(f'{BASE}/api/workspaces/studio_demo/entities/{first_ent_id}')
    assert r.status_code == 200, f"Entity detail failed: {r.text}"
    ent_detail = r.json()
    print(f"[OK] 7. Entity Detail OK: {ent_detail['entity']['entity_id']} has {len(ent_detail['observations'])} observations")

    # 8. Events
    r = requests.get(f'{BASE}/api/workspaces/studio_demo/events')
    assert r.status_code == 200, f"Events failed: {r.text}"
    evs = r.json()
    print(f"[OK] 8. Change Events OK: {len(evs)} events recorded")

    # 9. Diagnostics
    r = requests.get(f'{BASE}/api/workspaces/studio_demo/diagnostics')
    assert r.status_code == 200, f"Diagnostics failed: {r.text}"
    diag = r.json()
    print(f"[OK] 9. Diagnostics OK: model={diag['detector_info']['architecture']}, reg_health={diag['registration_health']['status']}")

    # 10. Image streaming
    r = requests.get(f'{BASE}/api/workspaces/studio_demo/images/visit_01.png')
    assert r.status_code == 200, f"Image streaming failed: {r.text}"
    assert len(r.content) > 1000, "Image content too small"
    print(f"[OK] 10. Image streaming OK: {len(r.content)} bytes received")

    print("\nALL 10 LIVE BACKEND INTEGRATION CHECKS PASSED SUCCESSFULLY!")

if __name__ == '__main__':
    main()
