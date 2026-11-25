# ObjectMemory — Live End-to-End Verification Report

**Project Root:** `C:\Users\Rishh\.gemini\antigravity\scratch\objectmemory`  
**Verification Date:** `2026-11-25`

---

## 1. Environment & Startup Result
- **Python Environment:** `C:\Users\Rishh\AppData\Local\Programs\Python\Python311\python.exe` (Python `3.11.9`). No separate `.venv` existed; `pytest 9.2.0` was installed into this environment alongside existing `fastapi 0.151.0`, `uvicorn 0.54.0`, `httpx 0.28.1`, `opencv`, `onnxruntime`, and `tokenizers`.
- **Frontend Build:** `frontend/dist/index.html` and bundled assets were present and served directly by FastAPI at `http://127.0.0.1:8000/`.
- **Startup:** `python -m uvicorn objectmemory.api.app:app --host 127.0.0.1 --port 8000` (`PYTHONPATH=src`) started cleanly and served HTTP `200 OK`.

---

## 2. Dashboard Workflow Result (Live Browser Verification)
Verified in headless browser at `http://127.0.0.1:8000` with **0 console errors** and **0 HTTP `>= 400` API responses**:
- **Workspace Creation:** Created workspace `e2e_verify_ui` (`POST /api/workspaces` → HTTP `201`), opened automatically.
- **Detector Selection & Configuration:** Detector `<select>` exposed `yolov8n`, `yolov8`, `mock`, `yolo-world`, and `yolov8-world`. Selecting `yolo-world` switched default confidence to `0.15` and revealed the custom prompts input (`potted plant, book, bottle, scissors, cell phone, cup, banana`).
- **Single-Image Upload:** Uploaded `removal/V01.jpeg` (`POST /observations` → HTTP `201`, `1072.1 ms`, `100.0%` registration inliers). Created `visit_01 • Baseline` with `ent_visit_01_1` (`bottle`, 37%, `CONFIRMED_PRESENT`) and `ent_visit_01_2` (`cell phone`, 28%, `CONFIRMED_PRESENT`).
- **Batch Multi-Image Upload:** Uploaded `removal/V02.jpeg` and `removal/V03.jpeg` together (`POST /observations/batch` → HTTP `201`). Created `visit_02` (`66.6%` inliers, `cell phone` 29% + `potted plant` 15.6%) and `visit_03` (`90.5%` inliers, `cell phone` 29%, triggering `PENDING_REMOVAL` on `bottle`).
- **Webcam / Live Monitor Tab:** Rendered cleanly (`Camera is currently offline` / `Start Camera Stream` button; hardware stream not started in headless browser).

---

## 3. Six-Scenario Physical Pilot Processing Result
Processed `data/physical_pilot_v2/manifest_v2.json` (6 sequences × 5 chronological observations = 30 visits, 183 annotations; original images, annotations, and synthetic ordering timestamps preserved unchanged):
- **Registration:** `24/24` query visits succeeded (`AlignmentStatus.SUCCESS`, inliers `0.509`–`1.000`).
- **Frozen `YOLOv8n` B3 (`research_validation/physical_pilot_v2_b3_results.json`):** `56/183` (30.60%) state accuracy, `48/179` (26.82%) identity consistency, `6/180` false removals, `0/3` removal recall, `0` ID switches.
- **Explicit 7-Class `YOLOWorld` B3 (`research_validation/physical_pilot_v2_yoloworld_target_classes_results.json`):** `58/183` (31.69%) state accuracy, `54/179` (30.17%) identity consistency, `2/180` false removals, `2/3` removal recall (`1/1` event removal recall), `2/2` historical retrieval accuracy, `0` spurious tracks, `0` ID switches.
- **Track B Oracle (`research_validation/physical_pilot_v2_track_b_oracle_results.json`):** `140/183` (76.50%) entity-visit accuracy (`24/24 = 100%` on `V02`–`V05` for `stationary`, `occlusion`, `return_to_baseline`), `0` ID switches.

---

## 4. Persistence Across Restart Result
Verified on workspace `e2e_verify_ui` across a full backend process kill and restart:

| Metric / State | Before Restart | After Restart |
| :--- | :---: | :---: |
| `total_sessions` | `3` | `3` |
| `total_entities` | `3` | `3` |
| `total_events` | `4` | `4` |
| `reference_session_id` | `visit_01` | `visit_01` |
| `ent_visit_01_1` (`bottle`) | `PENDING_REMOVAL` | `PENDING_REMOVAL` |
| `ent_visit_01_2` (`cell phone`) | `CONFIRMED_PRESENT` | `CONFIRMED_PRESENT` |
| `ent_visit_02_1` (`potted plant`) | `CONFIRMED_PRESENT` | `CONFIRMED_PRESENT` |
| Reference image (`GET .../images/...`) | — | HTTP `200` (`1,892,782` bytes) |
| Comparison endpoint (`visit_01` → `visit_03`) | — | HTTP `200` |

---

## 5. Frontend / Backend Integration Result
All core views (`Entities & Beliefs`, `Change Timeline`, `Scene Comparison`, `Spatial Memory`, `Diagnostics`) rendered live backend data including `PENDING_REMOVAL` / `PROVISIONAL` uncertainty badges. **Observed minor UI/backend presentation mismatches:**
1. **Top Navbar Overflow:** At `1280px` width, `Diagnostics` and `Evaluation Lab` overflow horizontally offscreen.
2. **Batch Visit Auto-Naming:** Batch-uploaded `V02.jpeg` and `V03.jpeg` were named `Visit 01 (V02)` and `Visit 02 (V03)` (1-indexed per batch rather than workspace-relative `Visit 2` / `Visit 3`).
3. **Pending Removal Absence Streak:** `evt_visit_03_ent_visit_01_1_pending_removal` rendered `Absence Streak: 0 visit(s)` while its explanation text stated `"Object not detected for 2 consecutive reliably registered visits"`.
4. **Timeline Filter Label:** Entity filter dropdown displayed `All Entities (4)` (4 events count instead of 3 entities).
5. **Scene Comparison (`visit_01` → `visit_02`) State:** Displayed `ent_visit_01_1` (`bottle`) with its latest state (`PENDING_REMOVAL` from `visit_03`) rather than its state as of `visit_02`.
6. **Diagnostics Detector Card:** Always displayed the default `yolov8` (`models\yolov8n.onnx`) engine info even when `yolo-world` was used for ingestion.

---

## 6. Full Test-Suite Result
Executed `python -m pytest -v --tb=short` (`PYTHONPATH=src`) across all `38` test files in `tests/`:
- **Passed:** `400` (`+ 39 subtests passed`)
- **Failed:** `0`
- **Skipped:** `1` (`tests/test_benchmark_manifest.py::TestBenchmarkManifestLoader::test_17_symlink_escaping_dataset_root`, skipped on Windows where unprivileged symlinks are unavailable)
- **Collection Errors:** `0`
- **Runtime:** `153.50s` (`0:02:33`)

---

## 7. Actual Failures & Causes
- **0 automated test failures** (`400 passed, 0 failed`).
- **0 backend HTTP/runtime errors** during live UI & restart verification.
- **6 minor frontend presentation/UX mismatches** listed in Section 5 (non-blocking for core pipeline execution).

---

## 8. Remaining Tasks Before Final Project Submission
1. *(Optional UI polish)* Fix the 6 minor frontend presentation quirks in Section 5 (navbar overflow, batch visit numbering, comparison query-visit state lookup, timeline entity count, absence streak display, active detector display) if demoing the live UI.
2. *(Dataset completeness)* Note the 2 duplicate-image groups in `physical_pilot_v2` (`removal/V01 == addition/V02 == occlusion/V04` and `addition/V01 == occlusion/V01 == occlusion/V05`) as a documented limitation in the final report/poster, or recapture those 5 frames if independent captures are required.
3. *(Deliverables)* Finalize the end-semester report and poster using the verified benchmark tables (`YOLOv8n B3`, `YOLOWorld 7-class B3`, `Track B Oracle`, `ETH ASL Lounge`, and `synthetic_workbench_v2`).
