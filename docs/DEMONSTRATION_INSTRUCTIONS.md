# ObjectMemory — Demonstration Instructions

**Document ID:** `OM-DOC-DEMO-01`  
**Date:** November 21, 2026  
**Audience:** Demonstrators, Evaluators, Technical Testers  
**Subject:** Setup, Verification, and Walkthrough Instructions  

---

## 1. Quick Start Demonstration

ObjectMemory includes an automated end-to-end demonstration script that executes the complete physical-cognitive reasoning pipeline without manual clicking.

> [!NOTE]
> **Scope of Automated Demonstration vs. Browser UI Verification:**
> The automated demonstration script (`scripts/run_final_demonstration.py`) programmatically exercises the backend algorithms, spatial-temporal data structures, and SQLite state transitions (`WorkspaceManager`). The web application was verified by launching the FastAPI server, validating HTTP health endpoints (`/api/health`, `/api/workspaces`), and compiling the React 18 production bundle (`tsc -b && vite build`). No programmatic browser DOM automation (e.g. Playwright or Selenium) was executed; full interactive UI verification requires manual browser walkthrough following Section 3 below.

### Prerequisites
- Python 3.10 or 3.11 with dependencies installed.
- Repository root as working directory:
  ```bash
  cd C:\Users\Rishh\.gemini\antigravity\scratch\objectmemory
  ```

### Run Automated Demonstration
```bash
python scripts/run_final_demonstration.py
```

### Expected Output
The script will sequentially demonstrate:
1. **Initial Scene Ingestion**: Loads baseline visit, registers coordinate frame, detects 5 objects, instantiates 5 persistent entity tracks.
2. **Stationary Observation**: Confirms 5 objects persist with 0 false removals and 0 spurious movements.
3. **Object Relocation**: Detects $\approx 198\text{px}$ displacement of `cup_01`, records an auditable `CONFIRMED_MOVED` event.
4. **Object Removal and Confirmation**: Removes `bottle_01`, transitions through `PENDING_REMOVAL` (streak 1) to `CONFIRMED_REMOVED` (streak 2).
5. **Occlusion Reasoning**: Occludes `scissors_01` with a folder, freezing state until recovery upon unmasking.
6. **Historical Object Retrieval**: Queries memory store and dumps the chronological change ledger.
7. **Registration Failure Handling**: Feeds a featureless image, resulting in `INSUFFICIENT_INLIERS` and triggering absence freezing without memory corruption.

Execution log is saved to: `research_validation/results/demo_execution_log.json`.

---

## 2. Interactive Web Application (FastAPI + React)

The interactive application allows visual exploration of the spatial-temporal memory graph, bounding box projections, change-event ledgers, and alignment overlays.

### 2.1 Start the FastAPI Backend
```bash
# From repository root
uvicorn objectmemory.api.app:app --host 0.0.0.0 --port 8000 --reload
```
- API Health Endpoint: [http://localhost:8000/api/health](http://localhost:8000/api/health)
- Swagger API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Mounted Frontend: [http://localhost:8000/](http://localhost:8000/) (served from `frontend/dist/`)

### 2.2 Optional: Run Frontend in Vite Development Mode
If actively modifying frontend UI components:
```bash
cd frontend
npm run dev
```
Open [http://localhost:5173/](http://localhost:5173/) in your web browser.

---

## 3. Interactive Walkthrough Guide

### Step 1: Create a Workspace
1. In the Web UI, click **Create Workspace**.
2. Enter Workspace ID: `electronics_bench_01`, Name: `Main Electronics Bench`.
3. Click **Create**.

### Step 2: Upload Initial Baseline Visit
1. In the workspace, click **Upload Observation**.
2. Choose file: `data/synthetic_workbench_v2/images/S01_stationary_visit_01.png`.
3. Set confidence threshold to `0.25`.
4. Click **Submit**.
5. Observe:
   - Detections appear on canvas with bounding boxes.
   - 5 entities created in `CONFIRMED_PRESENT` state.
   - This visit becomes the canonical reference frame.

### Step 3: Observe Object Movement
1. Click **Upload Observation**.
2. Select file: `data/synthetic_workbench_v2/images/S02_relocation_return_visit_02.png`.
3. Click **Submit**.
4. Observe:
   - Registration succeeds with high inlier ratio.
   - Cup is detected at a new coordinate.
   - Cup state updates to `CONFIRMED_MOVED`.
   - The **Evidence Ledger** logs a movement event with exact pixel displacement.

### Step 4: Verify Object Removal
1. Upload `data/synthetic_workbench_v2/images/S03_removal_confirmation_visit_02.png` (Visit 4).
   - Bottle is unobserved; enters `PENDING_REMOVAL` (yellow status badge).
2. Upload `data/synthetic_workbench_v2/images/S03_removal_confirmation_visit_03.png` (Visit 5).
   - Streak threshold reached; bottle transitions to `CONFIRMED_REMOVED` (red status badge).
   - Change event recorded in the audit trail.

### Step 5: Examine Entity History
1. Navigate to **Entity History** view.
2. Select the `bottle` or `cup` entity.
3. Review the complete lifecycle: timestamps, observation crops, spatial coordinates, and transition rationales.
