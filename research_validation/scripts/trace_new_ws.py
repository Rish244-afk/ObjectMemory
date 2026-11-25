import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, "src")

from objectmemory.persistence.repository import SQLiteMemoryStore

store = SQLiteMemoryStore(Path("data/workspaces/new/memory.db"))
print("=== SESSIONS IN WORKSPACE new ===")
for s in store.list_sessions(order_asc=True):
    meta_reg = (s.metadata or {}).get("registration", {})
    print(f"{s.session_id} | {s.session_name} | ts={s.timestamp} | img={s.image_path} | reg={meta_reg.get('status')} inlier={meta_reg.get('inlier_ratio')}")

print("\n=== RAW DETECTIONS PER SESSION ===")
for s in store.list_sessions(order_asc=True):
    raws = store.list_raw_detections_for_session(s.session_id)
    print(f"  {s.session_id} ({len(raws)}):", [(r.label, round(r.confidence, 3), [round(r.raw_bbox.x_min), round(r.raw_bbox.y_min), round(r.raw_bbox.x_max), round(r.raw_bbox.y_max)]) for r in raws])

print("\n=== OBSERVATIONS PER SESSION ===")
for s in store.list_sessions(order_asc=True):
    obs = store.list_observations_for_session(s.session_id)
    print(f"  {s.session_id} ({len(obs)}):", [(o.entity_id, o.visibility_status.value, round(o.detector_confidence or 0, 3), [round(o.canonical_bbox.x_min), round(o.canonical_bbox.y_min), round(o.canonical_bbox.x_max), round(o.canonical_bbox.y_max)] if o.canonical_bbox else None) for o in obs])

print("\n=== ENTITIES ===")
for e in store.list_entities():
    print(f"  {e.entity_id} | label={e.label} | state={e.current_state.value} | created_in={e.metadata.get('created_in_session')} | last={e.last_seen_session_id} | init_bbox={e.metadata.get('initial_canonical_bbox')}")

print("\n=== CHANGE EVENTS ===")
for row in store.conn.execute("SELECT * FROM change_events ORDER BY timestamp ASC").fetchall():
    d = dict(row)
    print(f"  {d['entity_id']} | {d['from_session_id']} -> {d['to_session_id']} | {d['prior_state']} -> {d['new_state']} | disp={d['displacement_distance']} | {d['explanation']}")
