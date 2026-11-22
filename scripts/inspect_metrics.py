import json
import glob
from pathlib import Path

files = sorted(glob.glob("research_validation/results/benchmarks/*.json"))
for f in files:
    fname = Path(f).name
    with open(f) as fp:
        d = json.load(fp)
    m = d.get("metrics", {})
    print(f"=== {fname} ===")
    print(f"  Config: {d.get('config_id')} ({d.get('config_name')})")
    print(f"  Dataset: {d.get('dataset_name')} | Provenance: {d.get('dataset_provenance')} | Genuine: {d.get('is_genuine_physical')}")
    print(f"  Observations: {m.get('total_observations_processed')} | Entity Evaluations: {m.get('total_entity_evaluations')}")
    print(f"  Det: P={m.get('detection_precision')}, R={m.get('detection_recall')}, F1={m.get('detection_f1_score')}")
    print(f"  FRR: {m.get('false_removal_rate')} ({m.get('false_removal_count')}/{m.get('false_removal_denominator')})")
    print(f"  FDP: {m.get('false_discovery_proportion')} ({m.get('false_discovery_count')}/{m.get('false_discovery_denominator')})")
    print(f"  Removal Recall: {m.get('removal_recall')} ({m.get('removal_recall_count')}/{m.get('removal_recall_denominator')})")
    print(f"  Event Removal Recall: {m.get('event_removal_recall')} ({m.get('event_removal_recall_count')}/{m.get('event_removal_recall_denominator')})")
    print(f"  Confirmation Rate: {m.get('confirmation_rate')} ({m.get('confirmed_change_events_count')}/{m.get('total_eligible_change_events')})")
    print(f"  Identity Consistency: {m.get('identity_consistency')} ({m.get('identity_consistency_numerator')}/{m.get('identity_consistency_denominator')})")
    print(f"  Fragmentation: {m.get('fragmented_identities')} | ID Switches: {m.get('id_switches')}")
    print(f"  Overall State Acc: {m.get('overall_state_accuracy')} ({m.get('correct_state_evaluations')}/{m.get('total_entity_evaluations')})")
