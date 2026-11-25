# ObjectMemory — Reproducible Demonstration Guide

This guide provides step-by-step instructions to set up, launch, and demonstrate **ObjectMemory** end-to-end using the FastAPI backend, React 18 web dashboard, and the reproducible physical pilot scenarios.

---

## 1. Environment and Dependency Setup

### 1.1 System Requirements
- **OS:** Windows 10/11 (or Linux/macOS with equivalent shell commands)
- **Python:** 3.11+ (verified on Python 3.11.9)
- **Node.js:** 18+ and npm (only required if rebuilding the frontend; pre-built assets already exist in `frontend/dist/`)
- **Hardware:** Standard x86_64 CPU (no GPU required)

### 1.2 Python Dependencies
From the repository root (`objectmemory/`):

```powershell
# Optional: create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install core runtime dependencies
pip install -r requirements.txt

# Install test runner (if running pytest)
pip install pytest httpx
```

Verify that required model weights are present in `models/`:
- `models/yolov8m.onnx` (~99.0 MB, **production default**)
- `models/yolov8n.onnx` (~12.2 MB, lightweight alternative)
- `models/yolov8s-worldv2.onnx`, `models/clip_text_quantized.onnx`, `models/clip_tokenizer.json` (open-vocabulary alternative)

### 1.3 Frontend Build (Optional)
The React SPA is already compiled in `frontend/dist/` and mounted automatically by FastAPI. Only rebuild if you modify frontend source files:

```powershell
cd frontend
npm install
npm run build
cd ..
```

---

## 2. Starting the Backend and Frontend

### 2.1 Single-Command Production Mode (Recommended)
Start the FastAPI server from the repository root with `PYTHONPATH=src`. FastAPI serves both the REST API and the pre-built React frontend at `http://127.0.0.1:8000/`:

```powershell
$env:PYTHONPATH = "src"
python -m uvicorn objectmemory.api.app:app --host 127.0.0.1 --port 8000
```

Verify the server is running:
- **Web Dashboard:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Swagger API Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check:** [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)
- **Registered Detectors:** [http://127.0.0.1:8000/api/detectors](http://127.0.0.1:8000/api/detectors) (should report `"default_detector": "yolov8m"`)

### 2.2 Optional Vite Dev Mode (Hot Reload)
If developing the frontend separately:
```powershell
cd frontend
npm run dev
```
Open [http://localhost:5173/](http://localhost:5173/) (proxies `/api` requests to port `8000`).

---

## 3. Creating a Workspace

1. Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) in your browser.
2. In the **Workspaces** sidebar or landing panel, click **Create Workspace**.
3. Enter a **Workspace ID** (e.g., `demo_bench_01`) and **Name** (e.g., `Physical Pilot Demo`).
4. Click **Create**. Each workspace receives an isolated directory under `data/workspaces/<workspace_id>/` containing its own `memory.db` SQLite database and `images/` folder.

---

## 4. Uploading and Processing Chronological Visits

ObjectMemory supports three ingestion modes in the **Overview & Ingest** view:
1. **Single Image Upload**: Upload one visit at a time with optional session name, ISO-8601 timestamp, notes, confidence threshold (default `0.25`), and detector selection (`yolov8m` default, `yolov8n`, or `yolo-world`).
2. **Batch Sequence Upload**: Select multiple images (`V01` through `V05`) at once; the UI processes them sequentially in chronological order.
3. **Webcam Live Monitor**: Capture frames directly from a connected camera manually or on an interval (browser camera permission required).

> **Important:** Visits must be submitted in monotonically non-decreasing timestamp order. Leave the timestamp blank to auto-assign sequential UTC timestamps, or ensure custom timestamps increase across visits.

---

## 5. Comparing Visits and Inspecting Persistent Memory

After uploading two or more visits:
1. **Before / After Scene Comparison**: Navigate to the **Scene Comparison** tab. Select a reference visit (typically `V01`) and a query visit (`V02`–`V05`) to view side-by-side / overlay comparisons, registration status (`REGISTERED` / `UNREGISTERED`), homography inlier ratio, and displacement vectors.
2. **Object Identities & Persistent States**: Navigate to **Entities & Beliefs** (or inspect the entity table on Overview) to view each tracked entity's persistent ID (e.g., `ent_...`), COCO class label, canonical bounding box, and current `EpistemicState`:
   - `CONFIRMED_PRESENT` (green)
   - `CONFIRMED_MOVED` (blue/amber)
   - `PENDING_REMOVAL` (yellow)
   - `CONFIRMED_REMOVED` (red)
   - `INSUFFICIENT_EVIDENCE` / `AMBIGUOUS_IDENTITY`
3. **Change Reports & Audit Trail**: Navigate to the **Change Timeline** tab to inspect chronological `ChangeEvent` records, showing prior state, new state, pixel displacement, and natural-language explanations.
4. **Diagnostics**: Check the **Diagnostics** tab to verify active detector weights (`models/yolov8m.onnx`), backend availability, and registration telemetry.

---

## 6. Demonstrating All Core Scenarios (Physical Pilot v2)

Use the 6 genuine physical tabletop scenarios in `data/physical_pilot_v2/` (`V01.jpeg`–`V05.jpeg` in each folder) by creating a fresh workspace per scenario:

### 6.1 Stationary Baseline Stability
- **Images:** `data/physical_pilot_v2/stationary/V01.jpeg` $\to$ `V05.jpeg`
- **Expected Behavior:** Visit 1 establishes the canonical reference plane and instantiates detected objects (`book`, `cup`, `potted plant`, `cell phone`, `scissors`) as `CONFIRMED_PRESENT`. Visits 2–5 preserve `CONFIRMED_PRESENT` with 0 false removals and 0 spurious moves despite camera viewpoint jitter.
- **Known Limitation:** The clear glass tumbler (`bottle`) is often missed by COCO `yolov8m` on the reflective desk surface and may not be instantiated.

### 6.2 Object Movement
- **Images:** `data/physical_pilot_v2/movement/V01.jpeg` $\to$ `V05.jpeg`
- **Expected Behavior:** Relocating an object beyond the displacement threshold triggers `CONFIRMED_MOVED` while preserving the original entity ID (`id_switches = 0`) and logging the pixel displacement in the Change Timeline.

### 6.3 Object Addition
- **Images:** `data/physical_pilot_v2/addition/V01.jpeg` $\to$ `V05.jpeg`
- **Expected Behavior:** When `banana` is introduced in `V03.jpeg`, `CrossVisitEntityAssociator` leaves it unmatched against existing tracks, and `TemporalMemoryEngine` instantiates a new persistent entity in `CONFIRMED_PRESENT` without disturbing existing tracks.

### 6.4 Object Removal & Multi-Visit Confirmation
- **Images:** `data/physical_pilot_v2/removal/V01.jpeg` $\to$ `V05.jpeg` (or `data/synthetic_workbench_v2/images/S03_removal_confirmation_visit_01.png` through `visit_04.png` for deterministic clear-bottle removal)
- **Expected Behavior:** On the first visit an instantiated object disappears, it transitions to `PENDING_REMOVAL` (unless blocked by patch similarity). Only after $\ge 2$ consecutive reliably registered vacant visits does it commit `CONFIRMED_REMOVED`.
- **Known Limitation on Physical `removal/`:** Because `yolov8m` misses the transparent glass bottle in `V01.jpeg` of `data/physical_pilot_v2/removal/`, `bottle_01` is never instantiated in that sequence. Use `data/synthetic_workbench_v2/images/S03_removal_confirmation_visit_01.png`–`04.png` to reliably demonstrate the `CONFIRMED_PRESENT` $\to$ `PENDING_REMOVAL` $\to$ `CONFIRMED_REMOVED` lifecycle visually.

### 6.5 Temporary Occlusion
- **Images:** `data/physical_pilot_v2/occlusion/V01.jpeg` $\to$ `V05.jpeg`
- **Expected Behavior:** When an object is covered by an occluder, vacancy verification / multi-visit streak gating prevents immediate false removal, allowing the entity to recover `CONFIRMED_PRESENT` once uncovered.

### 6.6 Return to Baseline
- **Images:** `data/physical_pilot_v2/return_to_baseline/V01.jpeg` $\to$ `V05.jpeg`
- **Expected Behavior:** An object moved away from its baseline in intermediate visits and returned to its original coordinates in later visits retains its single continuous entity ID (no duplicate track fragmentation) and logs a return-to-baseline event (`CONFIRMED_MOVED` $\to$ `CONFIRMED_PRESENT`).

---

## 7. Supplementary ETH ASL Lounge Registration Evaluation

If the ETH ASL Lounge frames are extracted locally under `data/eth_asl_lounge/extracted/observation_01..10/rgb/`, you can also test 2D planar registration under 3D room viewpoint shifts:

1. Select any workspace in the top bar and open the **ETH Registration** tab.
2. Select Reference and Query sessions/frames (or click **Preset: Adjacent (0 → 1)** and **Preset: 150-Frame Gap (0 → 150)**).
3. Click **Run Planar Registration** to inspect `PlanarRegistrar` status, inliers, reprojection error, overlap fraction, aligned warp output, and failure warnings.
4. *Note:* This view is strictly a supplementary registration evaluation; ETH ASL Lounge has no 2D object-lifecycle ground truth, and Physical Pilot v2 remains the primary benchmark.

---

## 8. Expected Behavior & Known Limitations

- **Inference Speed:** `yolov8m` runs on CPU at ~619 ms per 1600×1200 image. Batch uploads of 5 images typically complete in 3–5 seconds.
- **Planar Surface Requirement:** Planar homography co-registration assumes objects rest on a dominant flat surface (tabletop/workbench). Large 3D camera translations across a room will cause registration failure (`UNREGISTERED`), which safely freezes absence counters rather than corrupting memory.
- **Transparent/Reflective Objects:** Pretrained COCO `yolov8m` struggles with clear glass containers (`bottle` recall 46.15% on the physical pilot).
- **Automated Backend Demo Alternative:** You can also run the non-interactive backend verification script:
  ```powershell
  python scripts/run_final_demonstration.py
  ```

---

## 9. Troubleshooting

| Issue | Likely Cause | Resolution |
| :--- | :--- | :--- |
| `ModuleNotFoundError: No module named 'objectmemory'` | `src/` is not on `PYTHONPATH`. | Run `$env:PYTHONPATH = "src"` (PowerShell) or `export PYTHONPATH=src` (bash) before launching `uvicorn`. |
| UI shows `yolov8n` or stale assets | Browser cache or old server instance running. | Hard-refresh (`Ctrl+Shift+R`) and ensure no older `uvicorn` process is bound to port `8000`. |
| `400 Out-of-order visit rejected` | Uploaded visit timestamp is earlier than the latest saved session in that workspace. | Leave timestamp blank for auto-incrementing UTC timestamps, or use a newer ISO-8601 timestamp. |
| `UNREGISTERED` / `INSUFFICIENT_INLIERS` | Featureless image, extreme blur, or severe 3D perspective change. | Capture from a comparable overhead/angled tabletop viewpoint with textured background features. |
| Webcam feed unavailable | Running in headless browser or missing OS camera permissions. | Grant camera access in browser settings or use Single/Batch file upload instead. |
| Port `8000` already in use | Previous server process still active. | Stop the existing process or pass `--port 8001` to `uvicorn`. |
