import json

d = json.load(open('research_validation/physical_pilot_v2_track_b_oracle_corrected_results.json', encoding='utf-8'))
mismatches = []
for s_id, s_data in d['sequences'].items():
    el = s_data['entity_level_evaluation']
    for gid, erec in el['per_entity_results'].items():
        for ev in erec['evaluations']:
            if not ev['is_match']:
                mismatches.append(f"{s_id} {gid} v{ev['visit_index']} exp={ev['expected_state']} obs={ev['observed_state']} expl={ev['explanation']}")

print(f"Total mismatches: {len(mismatches)} / 183")
for m in mismatches:
    print(m)
