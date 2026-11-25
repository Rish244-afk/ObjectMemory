# Movement Detection, Persistent Entity Identity, and User-Confirmed Labels — Verification Report

## 1. Executive Summary

This report documents the root-cause diagnosis, implementation, and verification of persistent entity identity, user-confirmed semantic labels, evidence-gated sub-threshold movement recovery, and human-assisted location confirmation in **ObjectMemory**.

All fixes preserve the seven-stage architecture, `YOLOv8m` production default (`YOLOv8n` alternative), frozen B3 thresholds (`confidence_threshold=0.25`, `nms_threshold=0.45`), historical benchmark baselines, and the ETH ASL Lounge registration integration.

---

## 2. Problem Reproduction (`physical_pilot_v2/movement`)

In the physical-pilot `movement` sequence (`data/physical_pilot_v2/movement/V01.jpeg`–`V05.jpeg`):

1. **V01 (`visit_01`)**: `YOLOv8m` detects the dark bottle-like object (`bottle_01`) in the upper-right workspace as `hair drier` with `confidence=0.5020` (`[1138, 119, 1545, 465]`, IoU `0.522` with ground-truth `bottle_01`).
2. **V02–V05 (`visit_02`–`visit_05`)**: The object is relocated to the lower-right workspace (`[1085, 445, 1435, 755]`). At the frozen production threshold (`0.25`), `YOLOv8m` emits no accepted detection for this object (`raw_detections_count = 5` in V05), causing the temporal engine to mark the entity `PENDING_REMOVAL` and ultimately `CONFIRMED_REMOVED` instead of `CONFIRMED_MOVED`.
3. **Sub-threshold evidence audit**: Inspecting raw `YOLOv8m` outputs below `0.25` reveals consistent sub-threshold candidates overlapping the lower-right ground-truth box, with high `VisualSignatureExtractor` cosine similarity to the V01 entity crop:
   - **V02**: `hair drier`, `conf=0.1586`, `[1108, 477, 1449, 746]`, IoU `0.784`, cosine similarity `0.9161`
   - **V03**: `hair drier`, `conf=0.1665`, `[1106, 469, 1444, 739]`, IoU `0.801`, cosine similarity `0.9181`
   - **V04**: `bottle`, `conf=0.1453`, `[1105, 470, 1443, 738]`, IoU `0.799`, cosine similarity `0.9254`
   - **V05**: `hair drier`, `conf=0.1780`, `[1106, 474, 1446, 743]`, IoU `0.794`, cosine similarity `0.9324`

---

## 3. Implemented Changes

### 3.1 Persistent User-Confirmed Semantic Labels & Identity Continuity
- **`src/objectmemory/api/schemas.py`**: Extended `EntitySummary` and `ComparisonDiffItem` with `raw_label`, `raw_confidence`, `confirmed_label`, `label_provenance`, `label_confirmed_at`, `observed_labels`, `class_uncertain`, and `evidence_source`. Added `ConfirmLocationRequest` and `ConfirmLocationResponse`.
- **`src/objectmemory/api/app.py`**: Added `DELETE /api/workspaces/{workspace_id}/entities/{entity_id}/label` and `POST /api/workspaces/{workspace_id}/observations/{session_id}/confirm-location`.
- **`src/objectmemory/api/workspace_manager.py`**:
  - `update_entity_label` (`PATCH`) sets `confirmed_label` and `label_provenance="USER_CONFIRMED"` while preserving `original_raw_label` and `raw_predicted_labels`.
  - `clear_entity_label` (`DELETE`) removes `confirmed_label` and restores `original_raw_label` (`label_provenance="MODEL_PREDICTED"`).
  - `_format_entity_summary` exposes raw vs. user-confirmed fields and `class_uncertain`.
- **`src/objectmemory/temporal/associator.py`**: Updated `CrossVisitEntityAssociator.associate` category compatibility matching to check `ent.label`, `ent.metadata["raw_predicted_labels"]`, and `ent.metadata["original_raw_label"]`, ensuring cross-visit identity continuity even after user label confirmation or raw class fluctuation (`hair drier` vs. `bottle`).

### 3.2 Evidence-Gated Sub-Threshold Movement Recovery (`SUB_THRESHOLD_RECOVERY`)
- **`src/objectmemory/api/workspace_manager.py` (`_attempt_subthreshold_recovery`)**:
  - Leaves frozen `0.25` confidence and `0.45` NMS thresholds unchanged and never mutates `raw_detections`.
  - Only activates when planar registration is reliable, an existing entity is unmatched in the current visit, and a sub-threshold candidate (`0.12 <= conf < 0.25`) satisfies:
    - IoU `< 0.20` with all accepted (`>= 0.25`) detections,
    - `VisualSignatureExtractor` cosine similarity `>= 0.85` (clamped to `[0.0, 1.0]`),
    - ambiguity margin `>= 0.05` over the second-best candidate, and
    - competing unmatched entity similarity `< 0.80`.
  - Records recovery observations and transition events with `evidence_source="SUB_THRESHOLD_RECOVERY"`.

### 3.3 Human-Assisted Location Confirmation (`HUMAN_CONFIRMED`) & Comparison Ordering
- **`src/objectmemory/api/workspace_manager.py` (`confirm_entity_location`)**:
  - Persists a user-supplied bounding box in a specific visit with `evidence_source="HUMAN_CONFIRMED"` without altering `raw_detections`.
  - Gates movement claims on planar registration reliability: if registration is unreliable, records `INSUFFICIENT_EVIDENCE` (`registration_reliable=False`, `displacement_px=None`).
- **`src/objectmemory/api/workspace_manager.py` (`compare_sessions`)**:
  - Fixed `change_events` ordering from `ORDER BY timestamp ASC, event_id ASC` to `ORDER BY timestamp ASC, rowid ASC` so `HUMAN_CONFIRMED` and `SUB_THRESHOLD_RECOVERY` events on the same session timestamp cleanly supersede earlier automatic `PENDING_REMOVAL` events.

### 3.4 Frontend Dashboard Integration
- **`frontend/src/types/api.ts` & `frontend/src/api/client.ts`**: Added typed support for semantic label provenance, `evidence_source`, `clearEntityLabel`, and `confirmEntityLocation`.
- **`frontend/src/views/EntityHistoryView.tsx`**: Displays `Model prediction` vs. `User-confirmed identity`, class-uncertainty badge, and Confirm / Revise / Clear controls.
- **`frontend/src/views/SceneComparisonView.tsx`**: Displays `RECOVERY-ASSISTED` (`SUB_THRESHOLD_RECOVERY`) and `HUMAN-CONFIRMED` (`HUMAN_CONFIRMED`) badges, dashed overlay styling, inline label clearing, and the Human-Assisted Location Confirmation workflow.

---

## 4. Verification Results (Strictly Separated by Evidence Mode)

### 4.1 Automatic Detector Only (`YOLOv8m`, Frozen `conf >= 0.25`)
- **Physical-pilot `movement` (`V01->V05`)**: `ref_detections_count=5`, `query_detections_count=5`. Without recovery or human confirmation, `bottle_01` (`hair drier`, `0.5020` in V01) has no `>= 0.25` detection in `V02–V05` and transitions to `CONFIRMED_REMOVED`.
- **Frozen Benchmarks (Preserved Untouched)**:
  - `YOLOv8n` B3: `30.60%`
  - `YOLO-World` explicit 7-class B3: `31.69%`
  - `YOLOv8m` B3: `89.07%` (`163/183`)
  - Corrected Track B Oracle: `96.17%` (`176/183`)

### 4.2 Recovery-Assisted (`SUB_THRESHOLD_RECOVERY`)
- **Live FastAPI Verification (`pilot_movement_verify`, `V01->V05`)**:
  - `ref_dets=5`, `q_dets=5` (raw detections untouched)
  - `ent_visit_01_1` (`scissors`): `UNCHANGED` (`MODEL_DETECTION`, `2.7 px`)
  - `ent_visit_01_2` (`book`): `UNCHANGED` (`MODEL_DETECTION`, `4.6 px`)
  - `ent_visit_01_3` (`cell phone`): `UNCHANGED` (`MODEL_DETECTION`, `2.1 px`)
  - `ent_visit_01_4` (`cup`): `UNCHANGED` (`MODEL_DETECTION`, `15.3 px`)
  - `ent_visit_01_5` (`hair drier`): `CONFIRMED_MOVED` (`SUB_THRESHOLD_RECOVERY`, `326.2 px`)
  - `ent_visit_02_1` (`potted plant`): `ADDED` (`MODEL_DETECTION`)
- **Six-Scenario Regression (`research_validation/scripts/diagnose_6_scenarios.py`, exit code 0)**:
  - `STATIONARY` (`V01->V05`): 6 `UNCHANGED`, 0 false removals (`0.9–13.3 px`)
  - `MOVEMENT` (`V01->V05`): `ent_visit_01_5` `CONFIRMED_MOVED` (`326.2 px`)
  - `REMOVAL` (`V01->V05`): genuine removal preserved, 0 false removals
  - `ADDITION` (`V01->V05`): `banana` `ADDED`, `ent_visit_02_1` (`book`) `CONFIRMED_MOVED` (`85.9 px`)
  - `OCCLUSION` (`V01->V05`): `book` recovered to `CONFIRMED_PRESENT` (`0.0 px`)
  - `RETURN_TO_BASELINE`: `V01->V03` `ent_visit_01_6` (`bottle`) `CONFIRMED_MOVED` (`842.1 px`), `V01->V05` `CONFIRMED_PRESENT` (`26.7 px`)

### 4.3 Human-Confirmed (`HUMAN_CONFIRMED`) & Automated Test Suites
- **Frontend Build (`frontend/npm run build`)**: `tsc -b && vite build` passed (`0` errors, built in `1.25s`).
- **Pytest Suite (`tests/test_identity_and_recovery.py`, `tests/test_comparison_correctness.py`, `tests/test_api_backend.py`, `tests/test_eth_registration_api.py`, `tests/test_detector_registry.py`)**: **42 passed, 0 failed** (`67.01s`), covering:
  - Semantic label confirm, revise, clear, and backend-restart persistence while preserving raw predictions (`hair drier`) and `class_uncertain=True`.
  - Sub-threshold recovery (`CONFIRMED_MOVED`, `SUB_THRESHOLD_RECOVERY`, `query_detections_count=1`).
  - Ambiguous sub-threshold candidate rejection (`margin < 0.05` does not emit `CONFIRMED_MOVED`).
  - Genuine removal preservation when no recovery candidate exists (`CONFIRMED_REMOVED`).
  - Human-confirmed location persistence across restart (`HUMAN_CONFIRMED`, `CONFIRMED_MOVED`).
  - Unreliable registration gating on `confirm-location` (`registration_reliable=False`, `INSUFFICIENT_EVIDENCE`, `displacement_px=None`).
