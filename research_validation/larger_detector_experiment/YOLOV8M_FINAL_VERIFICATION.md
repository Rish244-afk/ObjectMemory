# ObjectMemory — YOLOv8m Final Scientific & Integration Verification

**Dataset:** `data/physical_pilot_v2/manifest_v2.json` (6 sequences, 30 observations, 183 annotations)  
**Verified Artifacts:**
- `research_validation/larger_detector_experiment/run_yolov8m_b3.py`
- `research_validation/larger_detector_experiment/yolov8m_b3_results.json`
- `research_validation/larger_detector_experiment/LARGER_DETECTOR_EXPERIMENT_REPORT.md`

---

## 1. Verification of the 89.07% (`163 / 183`) Result

1. **Unchanged B3 Formula:** `BenchmarkRunner.run_config(manifest, BenchmarkConfigFactory.create_config("B3"))` was executed unchanged (`support_status = SUPPORTED`). `overall_state_accuracy = 163 / 183 = 0.8907103825136612` (`89.07%`).
2. **All 183 Cases Accounted For:** Sum of `entity_evaluations` across the 6 `sequence_reports` is `30 + 30 + 30 + 33 + 30 + 30 = 183`; sum of `correct_state_evaluations` is `25 + 24 + 25 + 32 + 28 + 29 = 163`.
3. **177 Detection-Recall Denominator:** `183` total annotations minus `4` `UNOBSERVED` `bottle_01` entries (`removal/V02`–`V05`) minus `2` `HEAVILY_OCCLUDED` `book_01` entries (`occlusion/V02`–`V03`) = **`177`** visible GT bounding boxes. `YOLOv8m` matched `160 / 177` (**90.40%** recall) out of `168` target-class predictions (**95.24%** precision).
4. **Preserved Thresholds & Parameters:** `OpenCVDNNDetector` used production defaults (`conf=0.25`, `nms=0.45`), and `effective_parameters` in `yolov8m_b3_results.json` matches frozen `B3` verbatim (`min_registration_inlier_ratio=0.2`, `movement_displacement_threshold=35.0`, `vacancy_visits_for_removal=2`, `iou_match_threshold=0.4`, etc.).
5. **No GT Leakage / No Fine-Tuning:** `YOLOv8m` ran strictly on `cv2.imread` images using official Ultralytics COCO `yolov8m.pt` exported to ONNX (`(1, 3, 640, 640)` → `(1, 84, 8400)`, `COCO_CLASSES_80`) with zero fine-tuning.
6. **11 Spurious / Duplicate Identities:** `1 + 1 + 2 + 3 + 2 + 2 = 11` across the 6 sequences, driven primarily by `8` extra `book` false positives (`36` `book` predictions vs `28` GT, e.g., detecting the printed booklet under the plant) plus transient unmatched tracks.

---

## 2. Explanation of the Oracle Discrepancy (`89.07%` vs `76.50%`)

Both `YOLOv8m B3` (`BenchmarkRunner`) and `Track B Oracle` (`scripts/evaluate_physical_workbench.py:evaluate_physical_track_b`) use the **same `TemporalMemoryEngine` and `PlanarRegistrar`**, yet `Track B` scored **76.50% (`140/183`)** while `YOLOv8m B3` scored **89.07% (`163/183`)**.

**Exact Root Cause (`evaluate_physical_workbench.py:382-396`):**
- In `evaluate_sequence_entity_level`, `Track B` looks up each entity's Visit-1 (`V01`) state **only** in `visit_reports[1].entity_updates`:
  - `V01` newly instantiated entities are stored directly in `store` (`CONFIRMED_PRESENT`), not returned as update transitions in `v_rep.entity_updates`.
  - Consequently, `observed` is `None` for every `V01` entity and falls back to `EpistemicState.INSUFFICIENT_EVIDENCE` (`"Historical state record missing for visit"`).
  - Across 6 sequences × 6 `V01` objects = **36 Visit-1 evaluations**, `Track B` marked all 36 as incorrect (`INSUFFICIENT_EVIDENCE != CONFIRMED_PRESENT`), capping `Track B` at `147 / 183 = 80.33%` before any other error (`-7` in `movement`/`removal`/`addition` → `140 / 183 = 76.50%`).
- `BenchmarkRunner` (`YOLOv8m B3`) reads `store.list_entities()` after `V01`, scoring Visit-1 instantiated entities as `CONFIRMED_PRESENT`.
- **Conclusion:** The `89.07%` `YOLOv8m B3` score is **100% valid under the unchanged `BenchmarkRunner` B3 benchmark**; the lower `76.50%` `Track B` score is an artifact of `evaluate_sequence_entity_level`'s Visit-1 lookup.

---

## 3. Change Detection Validation

- **Why `removal_recall` is `0/3`:** `YOLOv8m` missed `bottle_01` in `removal/V01` (and `stationary`/`movement`, `12/26` overall), leaving `bottle_01` in `uninstantiated_gt_entities` for `removal/` (`0/3` removal recall, `0/1` event removal recall).
- **Validity of remaining scenarios:** `stationary` (`25/30`), `movement` (`24/30`), `addition` (`32/33`, `banana_01` `3/3`), `occlusion` (`28/30`), and `return_to_baseline` (`29/30`, `cr=1/1`, `hra=2/2`, `0` ID switches) are all valid under the unchanged B3 pipeline.
- **Remaining limitations:** (1) `bottle_01` overhead recall is `46.15%` (`12/26`); (2) `book` over-detection (`36` preds vs `28` GT) creates `11` spurious/duplicate tracks; (3) `895.29 ms/frame` CPU latency (~0.9s/visit).

---

## 4. Production Integration Readiness

- `DetectorRegistry` (`src/objectmemory/detection/registry.py:38, 175-210`) keeps `_default_name = "yolov8n"` and `WorkspaceManager._get_detector` (`workspace_manager.py:74-84`) resolves `detector_name` via `DetectorRegistry.get(key)`.
- **Smallest safe integration path (without changing the default):**
  1. Place `yolov8m.onnx` at `models/yolov8m.onnx`.
  2. Register `"yolov8m"` in `DetectorRegistry.reset_to_defaults()` with `is_default=False` (retaining `"yolov8n"` as `is_default=True` baseline/fallback).
  3. `/api/detectors` and the dashboard `<select>` will automatically expose `"yolov8m"` as a selectable option while preserving `"yolov8n"` as the default.

---

## Final Conclusion

- **89.07% (`163/183`) is valid** under the unchanged `BenchmarkRunner` B3 benchmark.
- **The Oracle discrepancy is explained** by `evaluate_sequence_entity_level` (`evaluate_physical_workbench.py:382-396`) scoring all 36 Visit-1 entities as `INSUFFICIENT_EVIDENCE`.
- **YOLOv8m is ready for controlled production integration** as a configurable option in `DetectorRegistry` (`is_default=False`), keeping `YOLOv8n` as the default baseline.
