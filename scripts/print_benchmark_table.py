import json

files = [
    ('Real Pilot', 'B0', 'research_validation/results/benchmarks/B0_real_pilot_result.json'),
    ('Real Pilot', 'B3', 'research_validation/results/benchmarks/B3_real_pilot_result.json'),
    ('Real Pilot', 'A5', 'research_validation/results/benchmarks/A5_real_pilot_result.json'),
    ('Real Pilot', 'A2*', 'research_validation/results/benchmarks/A2_real_pilot_result.json'),
    ('Real Pilot', 'A3*', 'research_validation/results/benchmarks/A3_real_pilot_result.json'),
    ('Synth v2', 'B0', 'research_validation/results/benchmarks/B0_synthetic_workbench_v2_result.json'),
    ('Synth v2', 'B3', 'research_validation/results/benchmarks/B3_synthetic_workbench_v2_result.json'),
    ('Synth v2', 'A5', 'research_validation/results/benchmarks/A5_synthetic_workbench_v2_result.json'),
    ('Synth v2', 'A2*', 'research_validation/results/benchmarks/A2_synthetic_workbench_v2_result.json'),
    ('Synth v2', 'A3*', 'research_validation/results/benchmarks/A3_synthetic_workbench_v2_result.json'),
    ('OpenLORIS', 'B0', 'research_validation/results/benchmarks/B0_openloris_result.json'),
    ('OpenLORIS', 'B3', 'research_validation/results/benchmarks/B3_openloris_result.json'),
    ('OpenLORIS', 'A5', 'research_validation/results/benchmarks/A5_openloris_result.json'),
]

print(f"{'Dataset':12s} | {'Cfg':4s} | {'Det P/R/F1':14s} | {'FRR':5s} | {'FDP':5s} | {'RR':5s} | {'CR':5s} | {'IC':5s} | {'Frag':4s} | {'StateAcc':8s}")
print("-" * 88)

for dname, cid, path in files:
    with open(path) as f:
        data = json.load(f)
    m = data.get('metrics', {})
    dp = f"{m.get('detection_precision'):.2f}" if m.get('detection_precision') is not None else 'N/A'
    dr = f"{m.get('detection_recall'):.2f}" if m.get('detection_recall') is not None else 'N/A'
    df1 = f"{m.get('detection_f1_score'):.2f}" if m.get('detection_f1_score') is not None else 'N/A'
    det = f"{dp}/{dr}/{df1}"
    frr = f"{m.get('false_removal_rate'):.2f}" if m.get('false_removal_rate') is not None else 'N/A'
    fdp = f"{m.get('false_discovery_proportion'):.2f}" if m.get('false_discovery_proportion') is not None else 'N/A'
    rr = f"{m.get('removal_recall'):.2f}" if m.get('removal_recall') is not None else 'N/A'
    cr = f"{m.get('confirmation_rate'):.2f}" if m.get('confirmation_rate') is not None else 'N/A'
    ic = f"{m.get('identity_consistency'):.2f}" if m.get('identity_consistency') is not None else 'N/A'
    frag = str(m.get('fragmented_identities', 'N/A'))
    sa = f"{m.get('overall_state_accuracy'):.2f}" if m.get('overall_state_accuracy') is not None else 'N/A'
    print(f"{dname:12s} | {cid:4s} | {det:14s} | {frr:>5s} | {fdp:>5s} | {rr:>5s} | {cr:>5s} | {ic:>5s} | {frag:>4s} | {sa:>8s}")
