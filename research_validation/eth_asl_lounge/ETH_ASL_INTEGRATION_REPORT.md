# ETH ASL Change Detection (Lounge) — Supplementary Integration Report

**Document ID:** `OM-ETH-ASL-INTEGRATION-01`  
**Date:** November 25, 2026  
**Evaluation Role:** Supplementary 2D Planar Registration Evaluation  
**Primary Object-Lifecycle Benchmark:** Physical Pilot v2 (`data/physical_pilot_v2/manifest_v2.json`, 89.07% YOLOv8m B3)

---

## 1. Executive Summary

An audit of the repository confirmed that the **ETH ASL Change Detection (`lounge`)** dataset was previously preprocessed and evaluated offline via standalone scripts (`scripts/prepare_eth_asl_lounge.py`, `scripts/evaluate_eth_asl_lounge.py`), but had **no FastAPI or React dashboard integration**.

In accordance with ObjectMemory's architectural boundaries, a **supplementary ETH ASL Lounge registration evaluation** capability has been added via `/api/eth-lounge/*` (`src/objectmemory/api/eth_registration.py`) and the **ETH Registration** tab in the React dashboard (`frontend/src/views/EthRegistrationView.tsx`). This integration strictly evaluates 2D planar registration (`PlanarRegistrar` with unchanged default thresholds) without altering the seven-stage pipeline, YOLOv8m production default, or physical-pilot benchmark.

---

## 2. Dataset Structure (Verified on Disk)

- **Location:** `data/eth_asl_lounge/` (excluded from Git due to ~19.6 GB size; manifest tracked at `research_validation/ETH_ASL_DATASET_MANIFEST.json`).
- **Sessions:** 10 multi-week hand-held RGB-D scans (`extracted/observation_01` through `observation_10`).
- **RGB Frames:** **21,084** lossless `1280×720` PNG frames (`bgr8` source encoding, 0 corrupt/missing frames).
- **3D Meshes & Poses:** 10 aligned TSDF `.ply` meshes (`raw/complete_mesh/observation_0.ply`–`9.ply`) and per-session `metadata/frames_and_poses.json` containing `T_G_C`, `T_G_D`, and `tf` topic records.
- **Ground-Truth Boundary:** ETH ASL Lounge provides **no 2D object bounding boxes, persistent 2D object IDs, or epistemic lifecycle labels**. Consequently, object-lifecycle metrics (state accuracy, removal recall, identity consistency) are **UNAVAILABLE** and are never computed or claimed on this dataset.

---

## 3. Previous Offline Registration Experiments

Evaluated via `scripts/evaluate_eth_asl_lounge.py` and recorded in `research_validation/results/eth_asl_lounge/eth_asl_lounge_evaluation_results.json`:

| Experiment | Frames Evaluated | Succeeded / Attempted | Success Rate | Status Breakdown |
| :--- | :---: | :---: | :---: | :--- |
| `smoke_test` (adjacent `00000 → 00001`) | 2 | **1 / 1** | **100.0%** | `SUCCESS`: 2 |
| `intra_observation_01` (150-frame gap) | 10 | **0 / 9** | **0.0%** | `SUCCESS`: 1 (ref), `SINGULAR_HOMOGRAPHY`: 4, `INSUFFICIENT_MATCHES`: 3, `HOMOGRAPHY_ESTIMATION_FAILED`: 1, `INSUFFICIENT_INLIERS`: 1 |
| `cross_observation_first_frames` (`obs_01` vs `obs_02..10`) | 10 | **2 / 9** | **22.2%** | `SUCCESS`: 3 (incl. ref), `INSUFFICIENT_INLIERS`: 5, `SINGULAR_HOMOGRAPHY`: 2 |
| `cross_observation_pose_matched` | 10 | **0 / 9** | **0.0%** | `SUCCESS`: 1 (ref), `INSUFFICIENT_INLIERS`: 4, `INSUFFICIENT_MATCHES`: 4, `SINGULAR_HOMOGRAPHY`: 1 |

---

## 4. Dashboard & API Integration

### 4.1 Backend Endpoints (`src/objectmemory/api/eth_registration.py`)
Mounted in `src/objectmemory/api/app.py` via `app.include_router(eth_router)`:
- `GET /api/eth-lounge/sessions` — Discovers available `observation_XX` sessions, frame counts, mesh availability, and returns the explicit `SUPPLEMENTARY_REGISTRATION_ONLY` scope notice.
- `GET /api/eth-lounge/sessions/{session_id}/frames` — Paginated, stride-configurable RGB frame listing (`offset`, `limit`, `stride`) with strict session/filename validation.
- `GET /api/eth-lounge/sessions/{session_id}/frames/{filename}` — Serves PNG frames safely without path traversal.
- `POST /api/eth-lounge/register` — Executes the frozen `PlanarRegistrar` (`AlignmentConfig` defaults) on any intra- or cross-session frame pair, returning `success`, `status`, `method_used`, inlier count, inlier ratio, reprojection error stats (only when computed), overlap fraction (only when computed), latency, `unreliable_warning`, and `aligned_image_data_url` (when warped image is produced).
- `GET /api/eth-lounge/historical-summary` — Serves the frozen baseline summary from `eth_asl_lounge_evaluation_results.json`.

### 4.2 Frontend View (`frontend/src/views/EthRegistrationView.tsx`)
Accessible via the **ETH Registration** tab in `Navbar.tsx`:
- Displays a prominent **Supplementary Registration Only** badge and **Scientific Scope Boundary** banner.
- Allows independent Reference and Query session selection, stride-based frame browsing, and quick presets (`Adjacent 0 → 1`, `150-Frame Gap 0 → 150`, `Cross-Obs 01 → 02`).
- Renders Reference, Query, and Aligned Output (when available) side by side, along with measured registration metrics and clear failure warnings.

---

## 5. Verification & Test Results

1. **Automated API Tests (`tests/test_eth_registration_api.py`):** **4 / 4 passed** (covering mocked session/frame discovery, path-traversal/invalid-input rejection, synthetic success/failure handling, and live `data/eth_asl_lounge` adjacent vs. 150-frame gap verification).
2. **Existing Physical-Pilot / Workspace Backend Tests (`tests/test_api_backend.py`):** **9 / 9 passed** (confirming zero regression to workspace/physical-pilot workflows).
3. **Frontend TypeScript & Production Build (`npm run build`):** Passed cleanly (`tsc -b && vite build`, `dist/assets/index-glQ0vwqz.js`, 408.33 kB).
4. **Live Browser End-to-End Verification (`http://127.0.0.1:8000/`):**
   - Loaded all 10 sessions (`observation_01`–`observation_10`).
   - **Adjacent (`observation_01/frame_00000.png → frame_00001.png`):** `SUCCESS` (`SIFT`, 418.2 ms), `355` inliers, `86.6%` inlier ratio, `0.61 px` mean reprojection error, `99.5%` overlap, aligned output rendered.
   - **150-Frame Gap (`observation_01/frame_00000.png → frame_00150.png`):** `INSUFFICIENT_MATCHES` (`SIFT`, 385.4 ms), `8` tentative matches (`< 10`), `0` inliers, `N/A` reprojection/overlap, clear failure warning displayed, no aligned warp rendered.
   - Zero console errors and zero failed network requests.

---

## 6. Limitations & Reproducibility

- **3D Parallax Boundary:** ETH ASL Lounge captures unconstrained handheld 3D room trajectories, whereas `PlanarRegistrar` models a 2D homography. Large camera translations/rotations cause expected registration failures (`INSUFFICIENT_MATCHES`, `INSUFFICIENT_INLIERS`, `SINGULAR_HOMOGRAPHY`).
- **No Object-Lifecycle Evaluation:** Do not cite ETH ASL Lounge for object detection/state accuracy or combine it with the 30-image Physical Pilot v2 benchmark (89.07% YOLOv8m B3).
- **Reproducing:** Ensure extracted frames exist under `data/eth_asl_lounge/extracted/observation_01..10/rgb/`, start FastAPI (`$env:PYTHONPATH="src"; python -m uvicorn objectmemory.api.app:app --host 127.0.0.1 --port 8000`), open `http://127.0.0.1:8000/`, select any workspace, and click the **ETH Registration** tab.
