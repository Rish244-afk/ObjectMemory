# Detection Inconsistency and Movement/No-Movement Reporting Fix Report

## 1. Root Cause of Inconsistent YOLO26m Detection
Per-image evaluation of all 30 images (`177` visible ground-truth objects) in `data/physical_pilot_v2/manifest_v2.json` via `research_validation/scripts/diagnose_yolo26m_missed.py` established:

1. **Exact ONNX Parity (`PARITY_ALL_EXACT: True`)**: `YOLO26mDetector` (`ONNX Runtime 1.31.0`) matched `YOLO('models/yolo26m.onnx').predict(img, imgsz=640, conf=0.25, iou=0.45)` identically on all representative images (`stationary_V01`, `movement_V01`, `movement_V05`, `removal_V01`, `addition_V01`, `occlusion_V01`, `return_to_baseline_V01`).
2. **No Preprocessing/Decoding/Coordinate Bug**: Letterboxing, RGB/BGR conversion, `[0, 1]` normalization, `(1, 84, 8400)` decoding, `COCO_CLASSES_80` class mapping, `inverse_transform_cxcywh`, and NMS (`0.45`) are verified exact.
3. **Genuine Model Confidence Limitations**: All `16` missed detections (`161/177` detected) are caused by raw YOLO26m confidence falling below the frozen `0.25` threshold:
   - `13` **`MISSED_LOW_CONF`** (IoU `0.686`–`0.982`, confidence `0.0655`–`0.2451`):
     - `cell phone` (`9`): `stationary/V01` (`0.2451`), `V02` (`0.1310`), `V03` (`0.2266`), `V05` (`0.1344`), `movement/V01` (`0.0747`), `V02` (`0.2131`), `addition/V04` (`0.1896`), `V05` (`0.2309`), `occlusion/V03` (`0.2089`).
     - `bottle` (`2`): `movement/V04` (`0.0655`), `return_to_baseline/V02` (`0.1690`).
     - `cup` (`1`): `stationary/V05` (`0.1699`).
     - `book` (`1`): `movement/V01` (`0.2285`).
   - `3` **`MISSED_NO_CANDIDATE`** (`< 0.05` confidence): `bottle_01` in `movement/V02`, `movement/V03`, `movement/V05`.

*Full 177-row per-image diagnostic table saved in `research_validation/yolo26m_per_image_diagnostic.json`.*

## 2. Detection Changes Made
- Preserved frozen B3 thresholds (`0.25` confidence, `0.45` NMS) without manufacturing detections or altering `YOLOv8m`/`YOLOv8n`.
- Verified `YOLO26mDetector` parity and documented low-confidence misses as an authentic model limitation.

## 3. Root Cause of Missing `MOVED` / `NOT MOVED` Reporting
Tracing `compare_sessions` (`src/objectmemory/api/workspace_manager.py`) and `SceneComparisonView.tsx` revealed four defects:
1. **Ambiguous `UNCHANGED` Status**: Stationary objects present in both visits with reliable canonical bounding boxes returned `status_change = "UNCHANGED"` rather than an explicit `NOT_MOVED` comparison state.
2. **Missing Explicit Uncertainty & Epistemic States**: Temporary misses (`PENDING_REMOVAL` / `MISSING_IN_QUERY`), candidate movements (`PENDING_MOVEMENT`), ambiguous identities (`AMBIGUOUS_IDENTITY`), and unreliable registration (`INSUFFICIENT_EVIDENCE`) were not mapped to explicit comparison states.
3. **Registration Failure Defaulted to `inlier_ratio = 1.0`**: When registration failed completely (`query_obs` empty), `qualities` was empty and `compare_sessions` defaulted `inlier_ratio` to `1.0` (`registration_status: "SUCCESS"`), hiding `REGISTRATION_UNRELIABLE`.
4. **Stale Frontend Comparison & Missing Positions**: `SceneComparisonView.tsx` only loaded sessions on `workspace.workspace_id` change (missing `workspace.total_sessions`), did not show `NOT MOVED` / `UNCERTAIN` filter pills, and did not display Ref/Query bounding box coordinates or `Not comparable` when displacement was unavailable.

## 4. Backend & Frontend Changes
- **`src/objectmemory/api/schemas.py`**: Added `movement_status: str = "NOT_MOVED"` to `ComparisonDiffItem`.
- **`src/objectmemory/api/workspace_manager.py`**:
  - Fixed `compare_sessions` registration reliability check to inspect `query_events` and `query_raws` when `query_obs` is empty.
  - Populated `movement_status` with `MOVED`, `NOT_MOVED`, `PENDING_MOVEMENT`, `INSUFFICIENT_EVIDENCE`, `IDENTITY_AMBIGUOUS`, `REGISTRATION_UNRELIABLE`, `NOT_DETECTED`, `REMOVED`, or `ADDED` while preserving `status_change` for backward compatibility.
- **`frontend/src/types/api.ts`, `frontend/src/components/StatusBadge.tsx`, `frontend/src/views/SceneComparisonView.tsx`**:
  - Added `movement_status` support, badge styling for all 9 states, `NOT_MOVED` and `UNCERTAIN` filters, Ref/Query bounding box display (`Ref: [...]` / `Qry: [...]` / `Qry: Not detected`), `Not comparable` displacement label, and `workspace.total_sessions` dependency for automatic refresh.

## 5. Six-Scenario Verification with YOLO26m (`verify_6_scenarios_yolo26m.py`)
- **`stationary` (`V01->V05`, `reg=SUCCESS`)**: `scissors`, `book` (`ent_visit_01_4`), `bottle`, `cup` (`SUB_THRESHOLD_RECOVERY`) reported `NOT_MOVED` (`2.7–5.7 px`); `book` (`ent_visit_01_2`) `NOT_DETECTED`; `potted plant` `INSUFFICIENT_EVIDENCE` (canonical box clipped at frame boundary); `cell phone` `ADDED`.
- **`movement` (`V01->V05`, `reg=SUCCESS`)**: `scissors`, `cup`, `book`, `vase` reported `NOT_MOVED` (`3.0–28.1 px`); `bottle` reported `REMOVED` under YOLO26m because YOLO26m emitted `< 0.05` confidence on the relocated dark bottle in V02/V03/V05 (whereas YOLOv8m with `SUB_THRESHOLD_RECOVERY` recovered it as `MOVED`).
- **`removal` (`V01->V05`, `reg=SUCCESS`)**: `bottle` `REMOVED`; stationary objects `NOT_MOVED` (`0.7–9.4 px`).
- **`addition` (`V01->V05`, `reg=SUCCESS`)**: `banana` `ADDED`; `vase` `MOVED` (`42.1 px`); stationary items `NOT_MOVED` (`0.6–21.1 px`).
- **`occlusion` (`V01->V05`, `reg=SUCCESS`)**: all visible returning objects `NOT_MOVED` (`0.0 px`).
- **`return_to_baseline`**:
  - `V01->V03`: `bottle` `MOVED` (`841.7 px`), `vase` `MOVED` (`46.0 px`), stationary items `NOT_MOVED` (`1.0–5.8 px`).
  - `V01->V05`: `bottle` returned to `NOT_MOVED` (`26.9 px`), stationary items `NOT_MOVED` (`1.9–11.7 px`).

## 6. Test Results
- **Frontend (`npm run build`: `tsc -b && vite build`)**: Passed (`0` errors, `1.30s`).
- **Pytest (`tests/test_movement_reporting.py`, `tests/test_yolo26m.py`, `tests/test_detector_registry.py`, `tests/test_identity_and_recovery.py`, `tests/test_comparison_correctness.py`, `tests/test_api_backend.py`, `tests/test_eth_registration_api.py`)**: **52 passed, 0 failed** (verified in latest run).

## 7. Remaining Limitations
- `YOLO26m` misses `bottle_01` at `< 0.05` confidence in `movement/V02, V03, V05` under frozen `0.25` thresholds; sub-threshold recovery cannot recover a `< 0.05` miss without human confirmation (`HUMAN_CONFIRMED`).
- Corner-clipped objects (`potted plant`) whose canonical box projects outside frame bounds honestly report `INSUFFICIENT_EVIDENCE` rather than `NOT_MOVED`.
