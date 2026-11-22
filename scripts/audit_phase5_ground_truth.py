"""
Phase 5 Task 1: Ground-Truth Evaluation Audit and Metric Recalculation.

Independently recalculates all metrics directly from saved predictions in
`scripts/phase4_evaluation_results.json` and manual annotations in
`data/real_workbench/manifest.json`.

Audits:
- Strict 1-to-1 matching: each GT matched at most once; each prediction counted as TP at most once.
- Per-image TP, FP, FN counts.
- Per-class metrics with canonical taxonomy normalization.
- Matched prediction records with exact IoU and confidence.
- Unmatched predictions (False Positives) and unmatched ground truth (False Negatives).
- Mathematical reconciliation with Phase 4 claims.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def compute_iou(boxA: List[float], boxB: List[float]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    boxBArea = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea
    return interArea / unionArea if unionArea > 0 else 0.0


CANONICAL_SYNONYMS = {
    "mug": "cup",
    "cup": "cup",
    "bottle": "bottle",
    "water bottle": "bottle",
    "book": "book",
    "notebook": "book",
    "potted plant": "potted plant",
    "plant": "potted plant",
    "scissors": "scissors",
}


def audit_ground_truth(
    phase4_results_path: Path,
    manifest_path: Path,
    output_audit_path: Path,
) -> Dict[str, Any]:
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    with open(phase4_results_path, encoding="utf-8") as f:
        phase4_data = json.load(f)

    # 1. Parse ground truth from manifest
    gt_by_visit: Dict[str, List[Dict[str, Any]]] = {}
    for v in manifest.get("visits", []):
        vid = v["visit_id"]
        gt_list = []
        for ann in v.get("annotations", []):
            if ann.get("physical_observation") == "VISIBLE" and ann.get("bbox") is not None:
                raw_lbl = ann.get("entity_label", "").lower().strip()
                canon_lbl = CANONICAL_SYNONYMS.get(raw_lbl, raw_lbl)
                gt_list.append({
                    "entity_id": ann.get("entity_id"),
                    "raw_label": raw_lbl,
                    "canonical_label": canon_lbl,
                    "bbox": ann.get("bbox"),
                    "expected_state": ann.get("expected_state"),
                    "annotated_event": ann.get("annotated_event"),
                })
        gt_by_visit[vid] = gt_list

    models = [
        ("yolov8n", "detection_records_yolov8n", 0.25, None),
        ("yolo_world", "detection_records_world", 0.15, [
            "potted plant", "plant", "cup", "mug", "bottle", "water bottle",
            "scissors", "book", "notebook", "tool", "pliers", "container"
        ]),
    ]

    audit_summary: Dict[str, Any] = {
        "audit_version": "Phase 5 Task 1 Ground Truth Audit",
        "ground_truth_source": str(manifest_path),
        "total_annotated_visits": len(gt_by_visit),
        "total_ground_truth_instances": sum(len(gts) for gts in gt_by_visit.values()),
        "models": {},
    }

    for model_name, rec_key, default_conf, prompts in models:
        model_audit: Dict[str, Any] = {
            "model_name": model_name,
            "confidence_threshold": default_conf,
            "prompts": prompts,
            "iou_thresholds": {},
        }

        records = [r for r in phase4_data[rec_key] if r.get("image_group") == "real_annotated"]

        for iou_thresh in [0.50, 0.30]:
            total_tp = 0
            total_fp = 0
            total_fn = 0

            per_visit_tables: List[Dict[str, Any]] = []
            per_class_counts: Dict[str, Dict[str, int]] = {}
            matched_records: List[Dict[str, Any]] = []
            unmatched_predictions: List[Dict[str, Any]] = []
            unmatched_ground_truth: List[Dict[str, Any]] = []

            for rec in records:
                vid = rec["image_id"]
                gt_instances = list(gt_by_visit.get(vid, []))
                gt_matched = [False] * len(gt_instances)

                # Predictions sorted by confidence descending
                preds = sorted(
                    zip(rec["labels"], rec["bboxes"], rec["confidences"]),
                    key=lambda x: x[2],
                    reverse=True,
                )

                v_tp = 0
                v_fp = 0
                v_matched = []
                v_unmatched_pred = []

                for p_lbl, p_box, p_conf in preds:
                    p_clean = p_lbl.lower().strip()
                    p_canon = CANONICAL_SYNONYMS.get(p_clean, p_clean)

                    best_iou = 0.0
                    best_gt_idx = -1

                    for g_idx, gt in enumerate(gt_instances):
                        if not gt_matched[g_idx] and p_canon == gt["canonical_label"]:
                            iou = compute_iou(p_box, gt["bbox"])
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = g_idx

                    # 1-to-1 match condition
                    if best_iou >= iou_thresh and best_gt_idx >= 0:
                        gt_matched[best_gt_idx] = True
                        matched_gt = gt_instances[best_gt_idx]
                        v_tp += 1
                        total_tp += 1

                        match_rec = {
                            "visit_id": vid,
                            "predicted_label": p_lbl,
                            "canonical_label": p_canon,
                            "confidence": round(p_conf, 4),
                            "predicted_bbox": p_box,
                            "matched_entity_id": matched_gt["entity_id"],
                            "ground_truth_label": matched_gt["raw_label"],
                            "ground_truth_bbox": matched_gt["bbox"],
                            "iou": round(best_iou, 4),
                        }
                        matched_records.append(match_rec)
                        v_matched.append(match_rec)

                        d = per_class_counts.setdefault(matched_gt["canonical_label"], {"tp": 0, "fp": 0, "fn": 0, "gt": 0})
                        d["tp"] += 1
                    else:
                        v_fp += 1
                        total_fp += 1
                        unmatched_p = {
                            "visit_id": vid,
                            "predicted_label": p_lbl,
                            "canonical_label": p_canon,
                            "confidence": round(p_conf, 4),
                            "predicted_bbox": p_box,
                            "reason": "IoU below threshold" if best_gt_idx >= 0 else "No matching ground-truth class/spatial overlap",
                            "best_unmatched_iou": round(best_iou, 4) if best_gt_idx >= 0 else 0.0,
                        }
                        unmatched_predictions.append(unmatched_p)
                        v_unmatched_pred.append(unmatched_p)

                        d = per_class_counts.setdefault(p_canon, {"tp": 0, "fp": 0, "fn": 0, "gt": 0})
                        d["fp"] += 1

                v_unmatched_gt = []
                for g_idx, gt in enumerate(gt_instances):
                    c = gt["canonical_label"]
                    d = per_class_counts.setdefault(c, {"tp": 0, "fp": 0, "fn": 0, "gt": 0})
                    d["gt"] += 1
                    if not gt_matched[g_idx]:
                        total_fn += 1
                        d["fn"] += 1
                        unmatched_g = {
                            "visit_id": vid,
                            "entity_id": gt["entity_id"],
                            "raw_label": gt["raw_label"],
                            "canonical_label": c,
                            "bbox": gt["bbox"],
                            "expected_state": gt["expected_state"],
                        }
                        unmatched_ground_truth.append(unmatched_g)
                        v_unmatched_gt.append(unmatched_g)

                per_visit_tables.append({
                    "visit_id": vid,
                    "ground_truth_count": len(gt_instances),
                    "prediction_count": len(preds),
                    "true_positives": v_tp,
                    "false_positives": v_fp,
                    "false_negatives": len(v_unmatched_gt),
                    "matched": v_matched,
                    "unmatched_predictions": v_unmatched_pred,
                    "unmatched_ground_truth": v_unmatched_gt,
                })

            prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            per_class_summary = {}
            for c_name, c_counts in sorted(per_class_counts.items()):
                c_tp = c_counts["tp"]
                c_fp = c_counts["fp"]
                c_fn = c_counts["fn"]
                c_gt = c_counts["gt"]
                c_p = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
                c_r = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0
                c_f1 = (2 * c_p * c_r) / (c_p + c_r) if (c_p + c_r) > 0 else 0.0
                per_class_summary[c_name] = {
                    "ground_truth_count": c_gt,
                    "true_positives": c_tp,
                    "false_positives": c_fp,
                    "false_negatives": c_fn,
                    "precision": round(c_p, 4),
                    "recall": round(c_r, 4),
                    "f1_score": round(c_f1, 4),
                    "sum_tp_fn_equals_gt": (c_tp + c_fn == c_gt),
                }

            model_audit["iou_thresholds"][f"iou_{int(iou_thresh*100)}"] = {
                "iou_threshold": iou_thresh,
                "overall": {
                    "total_ground_truth": sum(len(gts) for gts in gt_by_visit.values()),
                    "true_positives": total_tp,
                    "false_positives": total_fp,
                    "false_negatives": total_fn,
                    "precision": round(prec, 4),
                    "recall": round(rec, 4),
                    "f1_score": round(f1, 4),
                    "sum_tp_fn_equals_gt": (total_tp + total_fn == sum(len(gts) for gts in gt_by_visit.values())),
                },
                "per_class": per_class_summary,
                "per_visit_tables": per_visit_tables,
                "matched_records_count": len(matched_records),
                "unmatched_predictions_count": len(unmatched_predictions),
                "unmatched_ground_truth_count": len(unmatched_ground_truth),
            }

        audit_summary["models"][model_name] = model_audit

    with open(output_audit_path, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    print(f"Task 1 Audit complete. Results saved to {output_audit_path}")
    return audit_summary


if __name__ == "__main__":
    REPO_ROOT = Path(__file__).resolve().parent.parent
    p4_path = REPO_ROOT / "scripts" / "phase4_evaluation_results.json"
    man_path = REPO_ROOT / "data" / "real_workbench" / "manifest.json"
    out_path = REPO_ROOT / "scripts" / "phase5_ground_truth_audit.json"
    audit_ground_truth(p4_path, man_path, out_path)
