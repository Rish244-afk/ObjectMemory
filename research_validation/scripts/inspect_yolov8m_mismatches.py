import json

d = json.load(open('research_validation/larger_detector_experiment/yolov8m_production_integrated_results.json', encoding='utf-8'))
for s_id, s in d['sequence_reports'].items():
    c = s['correct_state_evaluations']
    t = s['entity_evaluations']
    print(f"{s_id}: {c}/{t} ({c/t:.2%}) frr={s['frr_num']}/{s['frr_den']} ic={s['ic_num']}/{s['ic_den']} uninst={s['uninstantiated_gt_entities']}")

print("--- Detection Evaluation ---")
det = d['detection_evaluation']
print(f"Total TP: {det['total_tp']} / {det['total_gt_visible']} (Recall: {det['overall_recall']:.2%}, Prec: {det['overall_precision']:.2%}, Mean ms: {det['mean_inference_ms']:.2f})")
for cls, st in det['per_class'].items():
    print(f"  {cls}: tp={st['tp']}/{st['gt_visible']} (rec={st['recall']:.2%}) preds={st['predictions']} (prec={st['precision']:.2%})")
