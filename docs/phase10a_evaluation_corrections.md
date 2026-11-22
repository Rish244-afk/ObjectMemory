# Phase 10A — Evaluation Corrections and Audit Resolution

**Date**: 2026-10-21  
**Project**: ObjectMemory Studio  
**Workspace**: `C:\Users\Rishh\.gemini\antigravity\scratch\objectmemory`  
**Evaluation Version**: `10a` (Predecessor: `10.0`, backups preserved as `.v10`)

---

## 1. Executive Summary

During the Phase 10 Results Audit ([docs/phase10_results_audit.md](docs/phase10_results_audit.md)), two critical evaluation defects were identified and confirmed:
1. **False-Positive Temporal Event Scoring**: Scenario `S04_bottle_removal` Visit 4 (`object_removed`) had a hardcoded `matched = True` condition in `scripts/evaluate_controlled_benchmark.py`, concealing that the temporal engine had not yet confirmed removal.
2. **Missing Per-Class Detection Counts**: In `scripts/evaluate_natural_detection.py`, the `per_class_counts` dictionary was initialized but never populated during frame evaluation or serialized to disk.

In Phase 10A, both evaluation defects have been strictly corrected in accordance with scientific rigor:
- **No changes were made to temporal engine logic or thresholds** to artificially improve scores.
- The original expected ground truth was preserved.
- The hardcoded stub was replaced with an objective assertion requiring `u.new_state == EpistemicState.CONFIRMED_REMOVED`.
- Visit 4 of S04 is now honestly scored and reported as a **failed event**, reducing Controlled Temporal Event Accuracy from the unearned claim of 100.0% to **92.86% (13/14)**.
- Per-class ground truth, prediction, TP, FP, FN, Precision, Recall, and F1 counts are now fully computed, displayed, and serialized in `scripts/phase10_coco_subset_evaluation_results.json` and `scripts/phase10a_coco_subset_evaluation_results.json`.
- 8 new regression tests were added in `tests/test_phase10a_corrections.py`, bringing total passing test suite to **288 tests** (0 failures).

---

## 2. Root Cause Analysis

### 2.1 Defect 1: Hardcoded Removal Scoring in S04 V4
- **Location**: `scripts/evaluate_controlled_benchmark.py` (formerly line 391)
- **Defective Code**:
  ```python
  elif exp.event_type == "object_removed":
      # If visit 4 or 5 and bottle is missing or removed
      matched = True
      passed_events += 1
  ```
- **Root Cause**:
  In `S04_bottle_removal`, `bottle_01` is removed in Visit 4. In Visit 4, the detector produces no bounding box for `bottle_01`. The temporal engine examines the last known patch location in canonical coordinates and measures patch similarity between Visit 1 and Visit 4. Because the desk surface behind the bottle shares visual characteristics with the foreground bottle base, the measured patch similarity is `0.7341`, which exceeds the vacancy threshold (`0.50`). Consequently, vacancy is unverified, and because the unobserved streak is only `1` (less than `removal_streak_threshold = 2`), the temporal engine conservatively preserves `EpistemicState.CONFIRMED_PRESENT` to protect against spurious removals caused by temporary occlusion or false non-detections.
  
  The evaluation script author inserted an unconditional `matched = True` shortcut rather than verifying the engine's epistemic state output.
- **Correction Applied**:
  The evaluation logic was refactored into a reusable, standalone function `score_temporal_event(exp, rep)`. For `object_removed`, it strictly checks:
  ```python
  removal_updates = [
      u for u in rep.entity_updates
      if u.new_state == EpistemicState.CONFIRMED_REMOVED
      and (target_lower in u.entity_id.lower() or target_lower in u.label.lower())
  ]
  matched = len(removal_updates) > 0
  ```
  In Visit 4, `new_state` is `CONFIRMED_PRESENT` (with explanation: `"Non-detection without verified vacancy cannot confirm removal; patch appearance still resembles object."`), so `matched = False`. The event is recorded as a failure with transparent diagnostic reason. In Visit 5, the unobserved streak reaches 2, matching `continued_absence` (`matched = True`).

### 2.2 Defect 2: Unpopulated Per-Class Detection Metrics
- **Location**: `scripts/evaluate_natural_detection.py` (lines 211, 266-300)
- **Defective State**:
  `per_class_counts: Dict[str, Dict[str, Any]] = {}` was initialized but never populated during image loops. Only aggregate confusion metrics and Supervision evaluator objects were recorded.
- **Correction Applied**:
  Implemented `evaluate_image_matches_and_counts(pred_dets, gt_anns, target_classes, iou_threshold=0.50)` adhering strictly to 1-to-1 greedy IoU assignment:
  - Highest IoU pairs are greedily matched.
  - Matches with $\text{IoU} \ge 0.50$ of matching class label count as True Positives ($TP$).
  - Unmatched predictions count as False Positives ($FP$). Duplicate detections for a single ground-truth object count as $FP$.
  - Unmatched ground-truth annotations count as False Negatives ($FN$).
  - Invariants hold per class: $TP + FN = GT$ and $TP + FP = \text{Pred}$.
  - The results are displayed in a formatted console table and serialized to `per_class_counts` in the JSON report.

---

## 3. Verification and Regression Testing

### 3.1 New Regression Test Suite
Created `tests/test_phase10a_corrections.py` containing 8 tests:
1. `test_genuinely_confirmed_removal_passes`: Asserts that an entity update with `CONFIRMED_REMOVED` returns `matched=True` and 0 false removals.
2. `test_pending_or_uncertain_removal_fails`: Asserts that `PENDING_REMOVAL` or unchanged `CONFIRMED_PRESENT` returns `matched=False`.
3. `test_absent_or_missing_event_fails`: Asserts that a missing entity or absent report returns `matched=False`.
4. `test_report_correctly_records_failed_event`: Asserts that failed event details preserve failure reasons.
5. `test_multiple_instances_of_same_class`: Asserts that greedy matching correctly pairs multiple instances without cross-contamination.
6. `test_duplicate_predictions_for_single_ground_truth`: Asserts that a duplicate prediction is scored as $FP$, keeping $TP=1, FP=1, GT=1, FN=0$.
7. `test_missing_detections`: Asserts that undetected GT objects correctly increment $FN$.
8. `test_unmatched_prediction_never_counted_as_tp`: Asserts that IoU $< 0.50$ is counted as $FP$, never $TP$.

### 3.2 Test Results
- **Phase 10A Unit Suite**: `Ran 8 tests in 0.001s - OK`
- **Full Project Unit Suite**: `Ran 288 tests in 49.140s - OK` (All 280 Phase 10 tests + 8 Phase 10A tests)
- **Frontend Production Build**: `npm run build` succeeds (`tsc -b && vite build` built in 524ms, 0 errors).

---

## 4. Benchmark Results Comparison (Phase 10 vs Phase 10A)

### 4.1 Controlled Temporal Benchmark (6 Sequences, 30 Visits)

| Metric | Phase 10 (Reported) | Phase 10A (Corrected) | Delta / Notes |
| :--- | :--- | :--- | :--- |
| **Total Sequences Evaluated** | 6 | 6 | Identical |
| **Total Visits Processed** | 30 | 30 | Identical |
| **Registration Success Rate** | 100.0% (24/24) | 100.0% (24/24) | Identical |
| **Mean Inlier Ratio** | 99.7% | 99.7% | Identical |
| **Identity Switches** | 0 | 0 | Identical |
| **Spurious Movements** | 0 | 0 | Identical |
| **Spurious Removals** | 0 | 0 | Identical |
| **Total Expected Events** | 14 | 14 | Identical |
| **Passed Events** | 14 | **13** | -1 (S04 V4 honest failure) |
| **Failed Events** | 0 | **1** | +1 (S04 V4 recorded) |
| **Temporal Event Accuracy** | 100.0% | **92.86%** | -7.14% (Accurate reflect) |

#### Event-by-Event Breakdown in Phase 10A:
1. `[S01 v2] lighting_change`: **Matched (True)**
2. `[S01 v4] lighting_change`: **Matched (True)**
3. `[S02 v3] object_moved`: **Matched (True)** (displacement: 85.0px)
4. `[S02 v4] return_to_baseline`: **Matched (True)** (displacement: 0.0px)
5. `[S03 v3] partial_occlusion`: **Matched (True)** (state: OCCLUDED)
6. `[S03 v5] reobserved`: **Matched (True)** (state: CONFIRMED_PRESENT)
7. `[S04 v4] object_removed`: **FAILED (False)** — State stayed `CONFIRMED_PRESENT` due to unverified vacancy streak=1.
8. `[S04 v5] continued_absence`: **Matched (True)** — Streak reached 2; absence persisted.
9. `[S05 v3] object_moved`: **Matched (True)** (displacement: 60.0px)
10. `[S05 v4] movement_persists`: **Matched (True)** (displacement: 60.0px from baseline)
11. `[S06 v2] camera_view_change`: **Matched (True)** (dx=-40px, dy=+30px)
12. `[S06 v3] camera_view_change`: **Matched (True)** (dx=+40px, dy=-30px)
13. `[S06 v4] camera_view_change`: **Matched (True)** (dx=+60px, dy=0px)
14. `[S06 v5] camera_view_change`: **Matched (True)** (dx=0px, dy=+50px)

---

### 4.2 Natural Image Benchmark (COCO val2017 20-Image Subset)

Both models processed the exact same 20 images containing 84 ground-truth annotations across 8 target classes with an initial warmup pass.

#### Overall Summary
| Model | Preprocessor | Target Invariant | GT | Pred | TP | FP | FN | Precision | Recall | F1 | mAP@50 | Latency (Mean) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLOv8n** | Letterbox 640x640 (BGR) | $TP+FN=84$, $TP+FP=28$ | 84 | 28 | 18 | 10 | 66 | 0.6429 | 0.2143 | 0.3214 | 0.3450 | 83.1 ms |
| **YOLO-World** | Direct Resize 640x640 (RGB) | $TP+FN=84$, $TP+FP=13$ | 84 | 13 | 7 | 6 | 77 | 0.5385 | 0.0833 | 0.1443 | 0.1716 | 330.3 ms |

#### Detailed Per-Class Breakdown (Phase 10A Added)

**YOLOv8n (OpenCV DNN CPU)**:
| Class | GT | Pred | TP | FP | FN | Precision | Recall | F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `cup` | 10 | 4 | 4 | 0 | 6 | 1.0000 | 0.4000 | 0.5714 |
| `bottle` | 14 | 1 | 0 | 1 | 14 | 0.0000 | 0.0000 | 0.0000 |
| `book` | 27 | 2 | 0 | 2 | 27 | 0.0000 | 0.0000 | 0.0000 |
| `laptop` | 2 | 3 | 2 | 1 | 0 | 0.6667 | 1.0000 | 0.8000 |
| `scissors` | 3 | 1 | 1 | 0 | 2 | 1.0000 | 0.3333 | 0.5000 |
| `potted plant` | 8 | 4 | 3 | 1 | 5 | 0.7500 | 0.3750 | 0.5000 |
| `chair` | 17 | 9 | 6 | 3 | 11 | 0.6667 | 0.3529 | 0.4615 |
| `cell phone` | 3 | 4 | 2 | 2 | 1 | 0.5000 | 0.6667 | 0.5714 |
| **Total** | **84** | **28** | **18** | **10** | **66** | **0.6429** | **0.2143** | **0.3214** |

**YOLO-World (ONNX Runtime CPU with CLIP Text Prompts)**:
| Class | GT | Pred | TP | FP | FN | Precision | Recall | F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `cup` | 10 | 1 | 1 | 0 | 9 | 1.0000 | 0.1000 | 0.1818 |
| `bottle` | 14 | 1 | 0 | 1 | 14 | 0.0000 | 0.0000 | 0.0000 |
| `book` | 27 | 0 | 0 | 0 | 27 | 0.0000 | 0.0000 | 0.0000 |
| `laptop` | 2 | 3 | 2 | 1 | 0 | 0.6667 | 1.0000 | 0.8000 |
| `scissors` | 3 | 1 | 0 | 1 | 3 | 0.0000 | 0.0000 | 0.0000 |
| `potted plant` | 8 | 3 | 2 | 1 | 6 | 0.6667 | 0.2500 | 0.3636 |
| `chair` | 17 | 1 | 1 | 0 | 16 | 1.0000 | 0.0588 | 0.1111 |
| `cell phone` | 3 | 3 | 1 | 2 | 2 | 0.3333 | 0.3333 | 0.3333 |
| **Total** | **84** | **13** | **7** | **6** | **77** | **0.5385** | **0.0833** | **0.1443** |

---

## 5. Artifact Traceability

The following files establish complete reproducibility and historical traceability:
1. `scripts/phase10_controlled_evaluation_results.json.v10`: Original Phase 10 snapshot.
2. `scripts/phase10_controlled_evaluation_results.json`: Canonical Phase 10A results (v10a).
3. `scripts/phase10a_controlled_evaluation_results.json`: Explicit versioned Phase 10A results.
4. `scripts/phase10_coco_subset_evaluation_results.json.v10`: Original Phase 10 snapshot.
5. `scripts/phase10_coco_subset_evaluation_results.json`: Canonical Phase 10A natural detection results.
6. `scripts/phase10a_coco_subset_evaluation_results.json`: Explicit versioned Phase 10A natural detection results.
7. `tests/test_phase10a_corrections.py`: Regression test suite protecting against future regression.

---

## 6. Scientific Conclusion

Phase 10A successfully resolves all evaluation discrepancies without modifying core temporal engine behavior, thresholds, or ground-truth definitions:
1. The temporal engine's safeguard against premature removal assertion (when visual vacancy is unverified after a single non-detection) functioned exactly as designed; the defect lay purely in the benchmark scoring harness having a hardcoded `matched = True` shortcut.
2. The benchmark harness now honestly documents this trade-off: safety against spurious removal (0 false removals across all 30 visits) comes at the cost of a 1-visit delay in confirming removal when background appearance resembles foreground objects.
3. The detector evaluation pipeline now provides complete per-class transparency, demonstrating that low recall on books (27 instances, 0 detections by YOLO-World, 0 TP by YOLOv8n) is the primary driver of low overall recall on the natural subset.
