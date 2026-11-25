# ObjectMemory — Final Dashboard Correctness Report

**Document ID:** `OM-DASHBOARD-CORRECTNESS-01`  
**Date:** November 25, 2026  
**System Version:** 2.0 (`YOLOv8m` Production Default + ETH ASL Lounge Supplementary Integration)  

---

## 1. Screenshot Defects & Reproduction Procedure

Two dashboard defects were observed in **Scene Comparison** for workspace `new` (`new2`, 10 visits) when comparing `Visit 01 (V01)` (`visit_01`) against `Visit 10 (V05)` (`visit_10`) with the `MOVED` filter active:
1. **Cross-Sequence Default Comparison:** `SceneComparisonView` defaulted to `sList[0]` (`visit_01`) and `sList[length - 1]` (`visit_10`), silently comparing Scenario 1 (`visit_01`–`visit_05`, `stationary`) against Scenario 2 (`visit_06`–`visit_10`, `movement`), because two separate 5-visit scenarios were uploaded into the same workspace.
2. **Contradictory `0 objs` Count vs. Visible Detections:** Both Reference and Query canvas headers displayed `0 objs` whenever `MOVED` (or any empty filter) was active, even though `visit_01` had 6 raw detections and `visit_10` had 5 raw detections. Additionally, edge-clipped objects whose `canonical_bbox` was `None` (e.g., `potted plant` at `y_min = 1`) were silently omitted from comparison overlays.
3. **Missing Intermediate Movements & Generic `MISSING_IN_QUERY`:** `WorkspaceManager.compare_sessions()` only inspected `change_events` attached to `query_id`, causing entities that moved in intermediate visits (and stayed moved at `query_id`) to report `UNCHANGED` with `displacement_px = None`, disappearing from the `MOVED` filter (`0 of 6`). Removed entities also reported generic `MISSING_IN_QUERY` instead of `CONFIRMED_REMOVED` / `PENDING_REMOVAL`.
4. **`hair drier` Classification on Dark Bottle:** In `visit_06` (`movement/V01.jpeg`), YOLOv8m predicted `hair drier` (confidence `0.502`, bbox `[1138, 119, 1545, 465]`) on the dark bottle/tumbler, and then failed to detect it in `visit_07`–`visit_10`.

**Reproduction:** Open `http://127.0.0.1:8000/`, select workspace `new2` (`new`), and open **Scene Comparison**. Previously, it defaulted to `visit_01 → visit_10`; selecting `MOVED` yielded `0 of 6` rows and `0 objs` on both canvases.

---

## 2. Exact Root Causes, Files Changed & Fixes Implemented

| Component / File | Root Cause | Fix Implemented |
| :--- | :--- | :--- |
| `src/objectmemory/api/workspace_manager.py` (`compare_sessions`, `update_entity_label`) | Only checked `query_events` on `query_id`, omitted intermediate `CONFIRMED_MOVED` transitions, omitted pair centroid displacement, returned `MISSING_IN_QUERY` for removed entities, lacked visit raw-detection counts, lacked cross-sequence detection, and had no user label-confirmation method. | Computed `state_at_ref` & `state_at_query`, surfaced intermediate `CONFIRMED_MOVED` & `CONFIRMED_REMOVED`/`PENDING_REMOVAL`, computed canonical pair displacement, added `ref_detections_count`/`query_detections_count`, added `ref_raw_bbox`/`query_raw_bbox`, added `cross_sequence_warning` on `V01` resets, surfaced `class_uncertain` & `raw_label`, and added `update_entity_label()` preserving `raw_detections`. |
| `src/objectmemory/api/schemas.py` (`ComparisonDiffItem`, `ComparisonResponse`, `EntityLabelUpdate`) | Lacked raw detection counts, raw bbox fallbacks, class uncertainty fields, cross-sequence warning, and label update schema. | Added backward-compatible fields and `EntityLabelUpdate`. |
| `src/objectmemory/api/app.py` | Missing entity label confirmation endpoint; `list_entities` needed `ValueError -> 400` preserved. | Added `PATCH /api/workspaces/{workspace_id}/entities/{entity_id}/label` and verified `list_entities` 400 handling. |
| `frontend/src/components/BoundingBoxCanvas.tsx` | Displayed `{validBoxes.length} objs` (which became `0 objs` when filtered). | Added `totalDetectionsCount` & `filterActive` props to display exact visit counts (e.g., `5 dets (0 shown)` or `6 dets`). |
| `frontend/src/views/SceneComparisonView.tsx` | Defaulted to `sList[0] → sList[-1]` across `V01` resets; dropped boxes with `canonical_bbox == null`; lacked cross-sequence warning, class uncertainty badge, and label confirmation UI. | Defaults to the latest contiguous `V01` sequence (`visit_06 → visit_10`), renders `cross_sequence_warning` banner, falls back to dashed `[raw]` bbox when `canonical_bbox` is `null`, shows `uncertain` badge + `raw: <label>`, and adds inline `edit` label confirmation. |
| `frontend/src/types/api.ts`, `frontend/src/api/client.ts` | Missing TypeScript interfaces and `confirmEntityLabel()` client call. | Added types and `api.confirmEntityLabel()`. |

---

## 3. Before-and-After Comparison Behavior

- **Before (`new2`, `visit_01 → visit_10`, `MOVED` filter):** Defaulted silently to cross-sequence `visit_01 → visit_10`, showed `0 objs` on both canvases, and hid intermediate movements as `UNCHANGED` (`disp: null`).
- **After (`new2`, default `visit_06 → visit_10`):**
  - Automatically selects the latest contiguous sequence (`Visit 06 (V01) → Visit 10 (V05)`).
  - Reference and Query canvas headers display `1600×1200px • 5 dets` (and `5 dets (0 shown)` when `MOVED` filter is active — never `0 objs`).
  - Manually selecting `visit_01 → visit_10` displays the red banner: *"Warning: Cross-Sequence Comparison Detected — Selected visits appear to span across two separate sequences ('visit_06: Visit 06 (V01)' restarts at V01)."*
  - Entities that moved in an intermediate visit (`visit_02`) and stayed moved at `visit_03` now surface as `CONFIRMED_MOVED` with exact canonical displacement (`200.0 px` in `test_comparison_correctness.py`, `85.9 px` in `addition` `visit_01 → visit_05`).

---

## 4. Object Classification, Identity & Raw YOLOv8m Movement Evidence

### 4.1 Exact Raw YOLOv8m Predictions on `data/physical_pilot_v2/movement/` (`V01`–`V05`)
Running the unchanged production `OpenCVDNNDetector(weights_path="models/yolov8m.onnx", nms_threshold=0.45)` at both production threshold (`conf >= 0.25`) and diagnostic low-confidence threshold (`conf >= 0.01`) on `movement/V01.jpeg`–`V05.jpeg` yielded:

| Visit | GT `bottle_01` BBox | Production (`conf >= 0.25`) | Sub-Threshold Candidate (`conf >= 0.01`, IoU $\ge 0.10$) | Exact Pipeline Stage |
| :--- | :--- | :--- | :--- | :--- |
| **V01** | `[1095, 43, 1412, 461]` (upper-right) | `hair drier` (`conf=0.5020`, `[1138, 119, 1545, 465]`, IoU=`0.522`) | Same (`0.5020`) | Passed conf & NMS; misclassified as `hair drier` |
| **V02** | `[1085, 445, 1435, 755]` (lower-right) | **0 overlapping detections** (5 other objects) | `hair drier` (`conf=0.1586`, `[1108, 477, 1449, 746]`, IoU=`0.784`) | **Dropped by confidence filter (`0.1586 < 0.25`)** |
| **V03** | `[1085, 445, 1435, 755]` (lower-right) | **0 overlapping detections** (5 other objects) | `hair drier` (`conf=0.1665`, `[1106, 469, 1444, 739]`, IoU=`0.801`) | **Dropped by confidence filter (`0.1665 < 0.25`)** |
| **V04** | `[1085, 445, 1435, 755]` (lower-right) | **0 overlapping detections** (5 other objects) | `bottle` (`conf=0.1453`, `[1105, 470, 1443, 738]`, IoU=`0.799`) | **Dropped by confidence filter (`0.1453 < 0.25`)** |
| **V05** | `[1085, 445, 1435, 755]` (lower-right) | **0 overlapping detections** (5 other objects) | `hair drier` (`conf=0.1780`, `[1106, 474, 1446, 743]`, IoU=`0.794`) | **Dropped by confidence filter (`0.1780 < 0.25`)** |

### 4.2 Why `CONFIRMED_REMOVED` Occurred & Scientific Boundary
- Because the frozen B3 confidence threshold (`0.25`) filters out all `V02`–`V05` candidates (`0.1453`–`0.1780 < 0.25`), `TemporalMemoryEngine` receives **zero detections** for `bottle_01` across 4 consecutive reliably registered visits (`inlier_ratio = 0.931`). Meanwhile, the object's original upper-right canonical patch (`[1138, 119, 1545, 465]`) is genuinely vacant because the object moved to the lower-right. Under the frozen 3-visit absence policy, the engine transitions `ent_visit_01_5` (`hair drier`) to `PENDING_REMOVAL` at `V03` and `CONFIRMED_REMOVED` at `V04`.
- **No prediction was fabricated or threshold lowered:** Lowering the confidence threshold below `0.25` would violate frozen B3 thresholds and introduce false positives elsewhere. Therefore, `bottle_01` movement in `physical_pilot_v2/movement` remains a **known pretrained COCO detector limitation** (`0.1780 < 0.25` in `V05`), rather than a fabricated movement event.
- **Dashboard Honesty & Label Confirmation:**
  1. `raw_detections` preserves `'hair drier'` untouched.
  2. Low-confidence (`< 0.55`) or cross-visit label-disagreement entities are flagged with `class_uncertain = True` (`uncertain` badge in UI).
  3. Operators can optionally confirm the semantic label via `PATCH .../entities/{id}/label`, updating `entity.label` while retaining `raw: hair drier` visibly in the table.

---

## 5. Scenario-Wise Results (All 6 Physical-Pilot Scenarios via `WorkspaceManager` + `YOLOv8m`)

Evaluated independently using `data/physical_pilot_v2/` (`V01.jpeg`–`V05.jpeg`) through `WorkspaceManager.process_image_observation()` and `compare_sessions("visit_01", "visit_05")`:

| Scenario | Registration (`V01→V05`) | Raw Dets (`V01 / V05`) | Actual `compare_sessions` Outcome & Failures |
| :--- | :---: | :---: | :--- |
| **A. Stationary** | `SUCCESS` (`0.798`) | `6 / 4` | 4 entities `UNCHANGED` (`1.6–4.0 px` drift); `potted plant` and `hair drier` missed in `V05` (`MISSING_IN_QUERY`, 0 false removals). |
| **B. Movement** | `SUCCESS` (`0.931`) | `5 / 5` | 4 entities `UNCHANGED` (`2.1–15.3 px`); dark bottle misclassified as `hair drier` (`uncertain=True`) in `V01` & missed after relocation (`CONFIRMED_REMOVED`); `potted plant` added in `V02` (`ADDED`, `q_raw=True`). **Failure:** `bottle_01` relocation not detected by YOLOv8m. |
| **C. Removal** | `SUCCESS` (`0.940`) | `6 / 5` | `potted plant` shows `q_raw=True` fallback; `bottle_01` missed in `V01` by YOLOv8m so removal is not detected (0 false removals). |
| **D. Addition** | `SUCCESS` (`0.909`) | `7 / 7` | `ent_visit_03_1 (banana)` correctly reported as `ADDED`; `ent_visit_02_1 (book)` surfaces `CONFIRMED_MOVED` (`85.9 px` disp). |
| **E. Occlusion** | `SUCCESS` (`1.000`) | `7 / 7` | Temporarily occluded `ent_visit_01_2 (book)` recovered to `CONFIRMED_PRESENT` (`0.0 px` disp); `vase` returned to `CONFIRMED_PRESENT`. |
| **F. Return to Baseline** | `SUCCESS` (`0.531`) | `8 / 8` | At `visit_01 → visit_03`: `ent_visit_01_6 (bottle)` reported `CONFIRMED_MOVED` (`842.1 px`). At `visit_01 → visit_05`: returned to `CONFIRMED_PRESENT` (`26.7 px`, same entity ID, 0 duplicates). |

---

## 6. Test & Verification Summary

- **Targeted & Backend Regression (`pytest -o pythonpath=src tests/test_comparison_correctness.py tests/test_api_backend.py tests/test_eth_registration_api.py tests/test_detector_registry.py`):** **33 / 33 passed**.
- **Frontend TypeScript & Build (`tsc -b && vite build`):** **Passed** (`dist/assets/index-CYjiMHLe.js`, 411.39 kB).
- **Live Browser E2E Verification (`http://127.0.0.1:8000/`, screenshot `final_step_36.png`):**
  - Confirmed `new2` defaults to `visit_06 → visit_10`.
  - Confirmed panel headers display `5 dets` and `5 dets (0 shown)` under `MOVED` filter (no `0 objs`).
  - Confirmed `visit_01 → visit_10` triggers the red Cross-Sequence Comparison Detected banner.
  - Confirmed **ETH Registration** tab loads all 10 sessions (`observation_01`–`observation_10`) with 0 console/network errors.
- **YOLOv8m & Persistence:** `yolov8m` remains the default detector (`GET /api/detectors`), `yolov8n` remains selectable, and all existing workspaces persist across server restarts.

---

## 7. Status Categorization & Demonstration-Readiness Verdict

- **Fixed and Verified:**
  - `SceneComparisonView` default sequence selection & cross-sequence warning banner.
  - `BoundingBoxCanvas` accurate detection counts (`X dets` / `X dets (Y shown)`) eliminating `0 objs`.
  - Intermediate `CONFIRMED_MOVED` persistence and `CONFIRMED_REMOVED`/`PENDING_REMOVAL` status reporting in `compare_sessions()`.
  - `raw_bbox` fallback rendering (`[raw]` dashed boxes) when `canonical_bbox` is `null`.
  - Class uncertainty flagging (`uncertain` badge + raw prediction preservation) and user-confirmable entity label endpoint (`PATCH .../label`).
  - ETH Registration supplementary view preservation (verified zero regression).
- **Known Scientific Limitations (Reproduced, Not Fabricated):**
  - Pretrained COCO `yolov8m` struggles on the transparent/dark bottle (`bottle_01` missed in `removal/V01`, misclassified as `hair drier` in `movement/V01` and undetected in `movement/V02–V05`). Improving this requires domain-specific detector fine-tuning rather than dashboard or threshold changes.

**Demonstration-Readiness Verdict:** **READY FOR ACADEMIC DEMONSTRATION.** The dashboard is internally consistent, truthful about detector uncertainty and registration boundaries, and cleanly demonstrates Stationary, Addition, Occlusion, Return-to-Baseline, and ETH Registration scenarios, while transparently documenting the pretrained COCO `bottle` limitation in Physical Pilot v2.
