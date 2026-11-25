# ObjectMemory — Final Project Stabilization Report

## 1. Final Project Status
- **Title**: *ObjectMemory: Long-Term Visual Memory for Detecting Persistent Environmental Changes*
- **Status**: **Feature-Complete & Stabilized**
- **Production Default Detector**: `YOLOv8m` (`models/yolov8m.onnx`)
- **Supported Alternative Detectors**: `YOLOv8n` (`models/yolov8n.onnx`), `YOLO26m` (`models/yolo26m.onnx`, `ONNX Runtime 1.31.0`), plus `YOLO-World` open-vocabulary (`models/yolov8s-worldv2.onnx`)

## 2. Seven-Stage Architecture Status
All seven stages operate end-to-end with frozen B3 thresholds (`confidence_threshold=0.25`, `nms_threshold=0.45`, `iou=0.50`) preserved:
1. **Image Acquisition**: Single upload, batch sequence, and live webcam stream (`WorkspaceOverviewView.tsx`).
2. **Object Detection**: `DetectorRegistry` (`yolov8m` default, `yolov8n`, `yolo26m`, `yolo-world`).
3. **Image Registration**: SIFT + homography (`PlanarRegistrar`) with inlier-ratio reliability gating (`>= 0.20`).
4. **Object Association**: `CrossVisitEntityAssociator` using canonical spatial proximity + `VisualSignatureExtractor` cosine similarity + multi-label raw-class compatibility.
5. **Evidence-Gated Temporal Reasoning**: `TemporalMemoryEngine` with 2–3 visit vacancy/unobserved confirmation gates, boundary-clipping guards, and return-to-baseline detection.
6. **Persistent Visual Memory**: `SQLiteMemoryStore` persisting sessions, raw detections, observations, entities, confirmed semantic labels, and auditable state transitions.
7. **Evidence Display & Change Reporting**: `SpatialMemoryExplorerView.tsx`, `SceneComparisonView.tsx`, `EntityHistoryView.tsx`, `BoundingBoxCanvas.tsx`, `StatusBadge.tsx`, and `EthRegistrationView.tsx`.

## 3. Defects Found & Classification
1. **HIGH (Fixed prior to final pass, verified here)**: Spatial Memory omitted `CONFIRMED_REMOVED`, `INSUFFICIENT_EVIDENCE`, `AMBIGUOUS_IDENTITY`, and `PENDING_MOVEMENT` in `getStateColors` and did not refresh on `total_sessions`/`total_entities`/`total_events`. Fixed in `SpatialMemoryExplorerView.tsx`, `BoundingBoxCanvas.tsx`, and `StatusBadge.tsx`.
2. **HIGH (Fixed prior to final pass, verified here)**: `compare_sessions` returned `UNCHANGED` instead of explicit `NOT_MOVED` and defaulted `inlier_ratio` to `1.0` when `query_obs` was empty after registration failure. Fixed in `workspace_manager.py` (`movement_status`) and `SceneComparisonView.tsx`.
3. **MEDIUM (Fixed in this pass)**: Switching from a workspace with more visits (e.g., `visit_09`) to one with fewer visits (e.g., `live_person_verify`, 4 visits) triggered one transient `GET .../observations/visit_09 -> 404` console error before `fetchSessions()` resolved. Fixed by resetting `selectedSessionId` and `selectedSessionDetail` at the start of the `workspace.workspace_id` `useEffect` in `WorkspaceOverviewView.tsx`.

## 4. Exact Files Modified in Final Stabilization
- `frontend/src/views/WorkspaceOverviewView.tsx` (cleared `selectedSessionId` and `selectedSessionDetail` on `workspace.workspace_id` switch; persisted `selectedDetector` in `localStorage`; added webcam overlapping-tick guard and live epistemic badges).
- Earlier stabilization deliverables verified in this pass: `src/objectmemory/detection/yolo26.py`, `src/objectmemory/detection/registry.py`, `src/objectmemory/api/schemas.py`, `src/objectmemory/api/workspace_manager.py`, `frontend/src/views/SpatialMemoryExplorerView.tsx`, `frontend/src/views/SceneComparisonView.tsx`, `frontend/src/components/BoundingBoxCanvas.tsx`, `frontend/src/components/StatusBadge.tsx`, `tests/test_yolo26m.py`, `tests/test_movement_reporting.py`, `tests/test_spatial_memory_status.py`.

## 5. Live-Monitoring Workflow Results (`live_person_verify`)
Reproduced using genuine saved webcam frames from `research_validation/webcam_recordings/live_webcam_32s.avi` (`1280x720`, `29.99 fps`) through `http://127.0.0.1:8000/api` with `yolov8m`:
- **V01 (frame 642, person visible)**: Detected `person` (`conf=0.597`), `reg=SUCCESS (1.0)`, entity `ent_visit_01_1` created as `CONFIRMED_PRESENT` in SQLite.
- **V02 (frame 825, person left)**: `dets=[]`, `reg=SUCCESS (0.793)`, entity retained (`CONFIRMED_PRESENT`, not prematurely removed); `compare V01->V02` returned `NOT_DETECTED` (`MISSING_IN_QUERY`).
- **V03 (frame 856, person absent)**: `dets=[]`, `reg=SUCCESS (0.786)`, entity transitioned to `PENDING_REMOVAL` (grey); `compare V01->V03` returned `NOT_DETECTED` (`PENDING_REMOVAL`).
- **V04 (frame 886, person absent)**: `dets=[]`, `reg=SUCCESS (0.803)`, entity transitioned to `CONFIRMED_REMOVED` (red); `compare V01->V04` returned `REMOVED` (`CONFIRMED_REMOVED`).

### Person Disappearance Support & Limitations
- `person` (COCO class `0`) is supported as a standard persistent entity by `YOLOv8m`, `YOLOv8n`, and `YOLO26m` with no person-specific hardcoding.
- Because ObjectMemory models persistent scene entities with an evidence-gated 2–3 visit absence rule, a person leaving the frame transitions `CONFIRMED_PRESENT` → `NOT_DETECTED` (1st miss) → `PENDING_REMOVAL` (2nd miss) → `CONFIRMED_REMOVED` (3rd reliable miss), rather than instant 1-frame removal.

## 6. Spatial Memory & Scene Comparison Correctness (Live Browser E2E)
Verified via browser automation on `http://127.0.0.1:8000/`:
- **`live_person_verify`**: Spatial Memory rendered `ent_visit_01_1 (person)` with **rose/red dashed stroke** (`#f43f5e`) + diagonal striped fill (`CONFIRMED_REMOVED`), NOT green; Scene Comparison (`V01` vs `V04`) displayed **`REMOVED`** (red badge) and `Qry: Not detected`.
- **`pilot_movement_verify`**: Spatial Memory rendered `ent_visit_01_5` (`hair drier`) in **amber** (`#f59e0b`, `CONFIRMED_MOVED`, dashed prior box + vector `Δ 326.2px`) and stationary items (`scissors`, `book`, `cell phone`, `cup`) in **green** (`#10b981`, `CONFIRMED_PRESENT`); Scene Comparison (`V01` vs `V05`) displayed **`MOVED`** (amber badge, `326.2 px`) and **`NOT MOVED`** (green badge, `2.1–15.3 px`) with Ref/Qry boxes.

## 7. Detector Integration Status
Verified `yolov8m`, `yolov8n`, and `yolo26m` through `WorkspaceManager` on `data/physical_pilot_v2/return_to_baseline` (`V01`–`V03`):
- `yolov8m` (default): `8` entities, states `['CONFIRMED_MOVED', 'CONFIRMED_PRESENT']`, comparison `['INSUFFICIENT_EVIDENCE', 'MOVED', 'NOT_DETECTED', 'NOT_MOVED']`.
- `yolov8n` (alternative): `4` entities, states `['CONFIRMED_PRESENT']`, comparison `['ADDED', 'NOT_MOVED']`.
- `yolo26m` (alternative): `8` entities, states `['CONFIRMED_MOVED', 'CONFIRMED_PRESENT']`, comparison `['INSUFFICIENT_EVIDENCE', 'MOVED', 'NOT_MOVED']`.

## 8. Database, API & Health Verification
- `GET /api/health` -> `HEALTHY`, `detector_available: True`, `default: yolov8`.
- `GET /api/detectors` -> `yolov8m` (`is_default=True`), `yolov8n`, `yolo26m`, `mock`, `yolo-world` all `is_available=True`.
- `GET /api/eth-lounge/sessions` -> `10` ETH ASL Lounge sessions intact.

## 9. Complete Test & Build Results
- **Full Backend Pytest Suite (`python -m pytest -o pythonpath=src`)**: **427 passed, 1 skipped, 0 failed** (`272.59s`), covering unit, integration, synthetic, physical-pilot, movement reporting, Spatial Memory, YOLO26m, identity/recovery, API backend, and ETH registration tests.
- **Frontend Production Build (`cd frontend; npm run build`, runs `tsc -b && vite build`)**: Passed with `0` errors.

## 10. Known Limitations
- CPU inference + SIFT registration runs at ~0.87–1.34 E2E FPS (`YOLOv8m` / `YOLO26m`), so live auto-capture uses interval sampling (5s/10s/30s) with overlap skipping rather than 30 FPS continuous ingestion.
- `YOLO26m` has lower `cell phone` recall (`70.0%`) and higher `book` false positives (`50` preds vs `28` GT) on the physical pilot, while `YOLOv8m` misclassifies the dark bottle as `hair drier` (`5` frames).
- Real webcam replay comparison has no manual bounding-box ground truth.

## 11. Exact Launch & Demonstration Instructions
1. **Start Backend & Built Frontend** (from `C:\Users\Rishh\.gemini\antigravity\scratch\objectmemory`):
   ```powershell
   $env:PYTHONPATH = "src"
   python -m uvicorn objectmemory.api.app:app --host 127.0.0.1 --port 8000
   ```
2. **Open Dashboard**: Navigate to `http://127.0.0.1:8000/`.
3. **Demonstrate**:
   - Select **YOLOv8m** (default) or **YOLO26m** in **Overview & Upload** (works for Upload, Batch Sequence, and Live Webcam Monitor).
   - Inspect **Scene Comparison** for explicit **`MOVED`** (amber), **`NOT MOVED`** (green), **`REMOVED`** (red), **`NOT DETECTED`** (grey), and Ref/Qry canonical coordinates.
   - Inspect **Spatial Memory** for the 5-color canonical map (`Present` green, `Moved / Pending` amber, `Removed` red dashed, `Insufficient / Not Detected` grey, `Ambiguous` purple) and displacement vectors.
   - Inspect **ETH Registration** for the supplementary 10-session 3D/registration view.
