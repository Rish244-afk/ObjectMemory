# ObjectMemory — Final Project Delivery Report

**Document ID:** `OM-FINAL-DELIVERY-01`  
**Date:** November 21, 2026  
**Status:** Completed, Verified, and Delivered  
**Evaluator:** Antigravity Research & Evaluation Harness  
**Applicability:** Full Repository, Core Engine, Evaluation Harness, Full-Stack App  

---

## 1. Executive Summary

This document represents the formal final project delivery report for **ObjectMemory: Long-Term Visual Memory for Detecting Persistent Environmental Changes**.

The project addresses the fundamental limitation of memoryless computer vision systems operating over repeated physical observations. By coupling planar coordinate homography, lightweight CPU object detection, multi-modal appearance extraction, optimal bipartite spatial association with ambiguity gating, and a multi-visit removal streak policy backed by visual patch vacancy verification, ObjectMemory successfully prevents false removal alarms and tracks persistent object identities across displacements.

The implementation, experimental validation, interactive application, and academic documentation are complete, verified against automated regression suites, and delivered without ungrounded claims.

---

## 2. What is Implemented

The delivered repository contains an end-to-end, fully functional system comprising:

### 2.1 Core Algorithms and Domain Logic (`src/objectmemory/`)
1. **Epistemic State Taxonomy (`domain/models.py`)**:
   - Decouples observation visibility from persistent environmental state.
   - States: `CONFIRMED_PRESENT`, `CONFIRMED_MOVED`, `PENDING_REMOVAL`, `CONFIRMED_REMOVED`, and `INSUFFICIENT_EVIDENCE`.
2. **ACID-Compliant Persistence (`persistence/`)**:
   - `SQLiteMemoryStore` with enforced foreign-key constraints (`PRAGMA foreign_keys = ON;`), indexing, and JSON serialization.
   - Atomic state transitions (`record_state_transition`) coupling entity belief updates and auditable change events in single transactions.
   - `ON DELETE RESTRICT` protection on audit ledgers.
3. **Planar Co-Registration Engine (`registration/`)**:
   - SIFT feature extraction with automatic fallback to ORB.
   - RANSAC geometric consensus estimating planar homography matrix $H \in \mathbb{R}^{3 \times 3}$.
   - Multi-criteria `AlignmentQualityEvaluator` checking condition numbers, corner convexity, scale distortion ($[0.15, 6.0]$), inlier ratios ($\ge 20\%$), and reprojection errors ($\le 4.0\text{px}$).
4. **Lightweight CPU Detection & Visual Signatures (`detection/`)**:
   - `OpenCVDNNDetector` executing Ultralytics YOLOv8 nano ONNX (`models/yolov8n.onnx`, 12.2 MB) strictly on CPU via OpenCV DNN.
   - `VisualSignatureExtractor` computing 148-dimensional unit descriptors (128-bin Hue-Saturation color histogram + 20-bin Sobel gradient orientations).
5. **Spatial-Temporal Associator (`temporal/associator.py`)**:
   - Kuhn-Munkres optimal bipartite matching combining bounding box IoU, normalized centroid distance, and appearance similarity.
   - Spatial displacement gating ($\tau_{\text{disp}} = 35.0\text{px}$) and ambiguity margin enforcement ($\epsilon_{\text{amb}} = 0.12$).
6. **Temporal Memory Engine (`temporal/engine.py`)**:
   - Chronological visit ingestion with monotonic timestamp verification.
   - Multi-visit unobserved streak counters ($\text{streak} \ge 2$) coupled with positive visual patch vacancy verification.
   - Transient occlusion handling and absence freezing during registration failure.

### 2.2 Standardized Benchmark Evaluation Harness (v2.0) (`benchmarks/`)
- `ManifestLoader`: Versioned v2.0 dataset manifest parser with path security, non-empty image path enforcement, and monotonic ordering checks.
- `BenchmarkConfigFactory`: Standardized configuration factory constructing B0 (memoryless baseline), B3 (complete engine), A5 (ambiguity ablation), and partial ablations A2\* and A3\*.
- `BenchmarkRunner`: Isolated `:memory:` execution engine capturing immutable per-visit prediction snapshots to eliminate historical state leakage, reporting 43 mathematically defined metrics.

### 2.3 Full-Stack Application (`src/objectmemory/api/` & `frontend/`)
- **FastAPI Backend (`api/app.py`)**: REST endpoints for workspace management, single-image observation upload, batch chronological ingestion, entity inspection, change event ledgers, and alignment comparison.
- **React 18 SPA (`frontend/`)**: Production-built TypeScript frontend featuring bounding box canvas overlays, interactive before/after split sliders, and evidence ledger inspection.

---

## 3. What Was Experimentally Evaluated

The evaluation matrix was executed using `BenchmarkRunner` across three distinct dataset tiers:

### 3.1 Evaluated Datasets
1. **Synthetic Workbench v2 (`data/synthetic_workbench_v2/`)**:
   - **Nature**: 25 observations across 5 sequences ($1080 \times 720$, PNG) generated using segmented real photographic cutouts (`cup`, `bottle`, `scissors`, `book`, `potted plant`), analytical contact shadows, directional cast shadows, procedural high-frequency wood grain, and corner ArUco fiducials.
   - **Scenarios**: `S01` stationary baseline, `S02` relocation ($\approx 198\text{px}$) and return, `S03` removal and confirmation, `S04` folder occlusion and recovery, `S05` lighting ($0.65\times$ to $1.25\times$) and viewpoint shifts.
2. **Exploratory Physical Desk Pilot (`data/real_workbench/`)**:
   - **Nature**: 7 sequential physical photographs taken of an office desk using a handheld smartphone camera.
3. **Public Trajectory: OpenLORIS-Scene Office1-1 (`research_validation/data/raw/`)**:
   - **Nature**: 10 keyframes from an Intel RealSense D435i camera on an automated guided vehicle (AGV) traversing an indoor office with OptiTrack 6-DOF ground-truth poses.
   - **Registration Boundary Study**: Used strictly to examine planar-registration failure modes under unconstrained 3D motion. Planar homography collapsed on 8 of 9 transitions due to non-planar depth parallax (>1.5m translations across the office), with only 1 small-motion transition (visit 1 -> visit 2, <=0.3m) achieving valid homography.
   - **Ground-Truth Limitations**: OpenLORIS-Scene provides 6-DOF trajectory poses for visual SLAM, but provides **zero native object bounding boxes or temporal lifecycle ground-truth annotations** for this experiment (all detection and temporal metrics are unscorable / `N/A`). It is explicitly not presented as successful object-memory validation.

### 3.2 Summary of Verified Benchmark Metrics

| Dataset | Config | Description | Det $F_1$ | False Removal Rate | False Discovery Prop. | Removal Recall (Visits) | Removal Recall (Events) | Confirmation Rate | Identity Consist. | Frag. | State Acc. |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Synthetic v2** (25 obs) | **B0** | Raw Detector Baseline | **0.987** | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | 0 | 0.904 |
| | **B3** | Complete ObjectMemory | *N/A* | **0.00%** (0/121) | **0.00%** (0/4) | **100%** (4/4) | **100%** (2/2) | **100%** (4/4) | **97.46%** (115/118) | **0** | **0.944** |
| | **A5** | No Ambiguity Margin | *N/A* | 0.00% (0/121) | 0.00% (0/4) | 100% (4/4) | 100% (2/2) | 100% (4/4) | 97.46% (115/118) | 0 | 0.944 |
| | **A2\*** | No Vacancy Verification | *N/A* | 0.00% (0/121) | 0.00% (0/4) | 100% (4/4) | 100% (2/2) | 100% (4/4) | 97.46% (115/118) | 0 | **0.928** |
| | **A3\*** | No FOV Safeguards | *N/A* | 0.00% (0/121) | 0.00% (0/4) | 100% (4/4) | 100% (2/2) | 100% (4/4) | 97.46% (115/118) | 0 | 0.944 |
| **Desk Pilot** (7 obs) | **B0** | Raw Detector Baseline | 0.698 (60/86) | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | 0 | 0.714 (30/42) |
| | **B3** | Complete ObjectMemory | *N/A* | **0.00%** (0/38) | *N/A* | **0.00%** (0/4) | **0.00%** (0/3) | **20.0%** (1/5) | **88.24%** (30/34) | **0** | **0.738** (31/42) |

*\* Marked `PARTIALLY_SUPPORTED` due to engine fallback logic.*

#### Metric Denominator Reconciliation & Exact JSON Values:
- **Removal Recall (Visits vs. Independent Events)**:
  - *Synthetic Workbench v2*: 4/4 visits (100.0%) across 3 visits for `bottle_01` (v3, v4, v5) and 1 visit for `scissors_01` (v5). Independent event recall is 2/2 events (100.0%) for bottle and scissors removal events.
  - *Desk Pilot*: 0/4 visits (0.0%) across `cup_02` (v3), `bottle_01` (v6), `notebook_01` (v6), `notebook_01` (v7). Independent event recall is 0/3 events (0.0%) because none fulfilled the multi-visit vacancy streak (>= 2 unobserved visits) before the pilot concluded.
- **Confirmation Rate**:
  - *Synthetic Workbench v2*: 4/4 eligible change events confirmed (100.0%) [cup move, bottle move, bottle removal, scissors removal].
  - *Desk Pilot*: 1/5 eligible change events confirmed (20.0%) [cup move confirmed; scissors micro-move, cup_02 removal, bottle_01 removal, notebook_01 removal unconfirmed].
- **False Removal Rate (FRR)**:
  - *Synthetic Workbench v2*: 0/121 non-removed evaluations (0.00%).
  - *Desk Pilot*: 0/38 non-removed evaluations (0.00%).
- **Identity Consistency (IC)**:
  - *Synthetic Workbench v2*: 115/118 observable present visits (97.46%, exact: 0.974576).
  - *Desk Pilot*: 30/34 observable present visits (88.24%, exact: 0.882353).
- **Why Temporal Metrics are N/A for B0**:
  - B0 is stateless and memoryless, evaluating single-frame raw YOLO detections (Synthetic: P=1.00, R=0.975, F1=0.987; Desk Pilot: P=0.577 [30/52], R=0.882 [30/34], F1=0.698 [60/86]). Because B0 does not form temporal tracks or streak counters, temporal change metrics (FRR, FDP, Removal Recall, Confirmation Rate, Identity Consistency) are undefined and reported as N/A rather than misleading zero scores.

---

## 4. What Works in the Demonstration

### 4.1 Automated Backend Demonstration (`scripts/run_final_demonstration.py`)
The automated demonstration script directly tests Python backend pipeline methods, spatial-temporal data structures, and SQLite state transitions (`WorkspaceManager`), producing the verified execution log in `research_validation/results/demo_execution_log.json`:

> [!NOTE]
> **Scope of Automated Demonstration vs. Browser UI Verification:**
> The automated demonstration script exercises the backend algorithms and data structures programmatically. The web application was verified by running the FastAPI server, validating HTTP health endpoints (`/api/health`, `/api/workspaces`), and successfully compiling the React 18 production bundle (`tsc -b && vite build`). No programmatic browser DOM automation (e.g. Playwright or Selenium) was executed; full interactive UI verification requires manual browser walkthrough following `docs/DEMONSTRATION_INSTRUCTIONS.md`.

1. **Initial Scene Ingestion**:
   - Ingested visit 1, estimated reference coordinate frame ($H = I$), detected 5 objects with YOLOv8n, and instantiated 5 persistent entity tracks.
2. **Stationary Stability**:
   - Ingested visit 2, achieved SIFT/ORB registration with 720 inliers (1.00 inlier ratio), confirmed all 5 objects with 0 false removals and 0 spurious movements.
3. **Object Relocation**:
   - Ingested visit 3 where the cup was displaced by $198.9\text{px}$. The associator linked the entity across displacement and recorded an auditable `CONFIRMED_MOVED` event.
4. **Object Removal and Confirmation**:
   - Ingested visits 4 and 5 where the bottle was removed. The bottle transitioned to `PENDING_REMOVAL` on streak 1 with verified vacancy, then transitioned to `CONFIRMED_REMOVED` on streak 2.
5. **Occlusion Reasoning and Recovery**:
   - Ingested visit 6 where the scissors were occluded by a folder, freezing removal progression. Ingested visit 7 where the folder was removed, recovering the entity as `CONFIRMED_PRESENT` without identity fragmentation.
6. **Historical Object Retrieval & Auditable Ledger**:
   - Queried SQLite store for entity lifecycles and printed the 12-event chronological audit trail with natural-language rationales and confidence scores.
7. **Registration Failure Handling**:
   - Ingested a featureless blank image, producing `INSUFFICIENT_INLIERS`. The engine safely froze entity states without corrupting memory or hallucinating removals.

---

## 5. Remaining Technical Limitations

To maintain academic and scientific integrity, the following technical boundaries are explicitly recorded:

1. **Dominant Plane Requirement**: The planar homography assumption requires observed objects to lie approximately on a dominant planar surface (desks, workbenches). As observed on OpenLORIS-Scene, unconstrained 3D multi-room AGV navigation with large translation ($>1.5\text{m}$) violates 2D homography due to depth parallax.
2. **Detector Sensitivity Ceiling**: When an object is missed by the detector for more consecutive visits than the unobserved streak threshold ($\ge 3$ visits), temporal tracking cannot maintain the entity indefinitely.
3. **Descriptor Invariance Boundaries**: Handcrafted 148-d HSV-Sobel descriptors have limited rotation invariance beyond $45^\circ$ in-plane rotation and are sensitive to non-uniform colored lighting (e.g. monochromatic laboratory LEDs).
4. **Partial Ablation Constraints**: Configurations A2\* and A3\* are partially supported because the core production engine contains internal safety fallbacks that cannot be completely deactivated via external flags alone.

---

## 6. Exact Commands to Run the Application

### 6.1 Start FastAPI Backend (with Built React SPA)
```bash
# From repository root:
uvicorn objectmemory.api.app:app --host 0.0.0.0 --port 8000 --reload
```
- Web Application: [http://localhost:8000/](http://localhost:8000/)
- API Documentation (Swagger): [http://localhost:8000/docs](http://localhost:8000/docs)
- Health Check: [http://localhost:8000/api/health](http://localhost:8000/api/health)

### 6.2 Optional: Run Frontend in Vite Development Mode
```bash
cd frontend
npm run dev
```
- Vite Dev Server: [http://localhost:5173/](http://localhost:5173/)

---

## 7. Exact Commands to Reproduce Experiments

### 7.1 Run Full Automated Regression Test Suite
```bash
python -m unittest discover -s tests -p "test_*.py"
```
*Expected: Ran 401 tests; 400 passed, 1 skipped, 0 failures, 0 errors.*

### 7.2 Run Automated End-to-End Demonstration
```bash
python scripts/run_final_demonstration.py
```
*Expected: Executes 7 demonstration steps; writes execution log to `research_validation/results/demo_execution_log.json`.*

### 7.3 Reproduce Synthetic Workbench v2 Benchmark Evaluations
```bash
python scripts/evaluate_synthetic_workbench_v2.py
```
*Expected: Executes B0, B3, A5, A2\*, A3\* on 25 observations; serializes JSON results in `research_validation/results/benchmarks/`.*

### 7.4 Re-Generate Synthetic Workbench v2 Dataset
```bash
python scripts/generate_synthetic_workbench_v2.py
```
*Expected: Generates 25 images and writes `data/synthetic_workbench_v2/manifest_v2.json`.*

---

## 8. Academic and Research Deliverables Summary

All required academic documentation has been produced and saved in the repository:

1. **Technical README**: [README.md](README.md)
2. **System Architecture**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
3. **Experimental Methodology & Tabulated Results**: [docs/EXPERIMENTAL_METHODOLOGY_AND_RESULTS.md](docs/EXPERIMENTAL_METHODOLOGY_AND_RESULTS.md)
4. **Limitations & Future Work**: [docs/LIMITATIONS_AND_FUTURE_WORK.md](docs/LIMITATIONS_AND_FUTURE_WORK.md)
5. **Demonstration Instructions**: [docs/DEMONSTRATION_INSTRUCTIONS.md](docs/DEMONSTRATION_INSTRUCTIONS.md)
6. **Final Year Project Report / Academic Thesis**: [research_validation/final_year_project_report.md](research_validation/final_year_project_report.md)
7. **Research Paper Draft (IEEE/CVPR format)**: [research_validation/research_paper_draft.md](research_validation/research_paper_draft.md)
8. **Final Oral Defense Presentation Structure**: [research_validation/final_presentation_structure.md](research_validation/final_presentation_structure.md)
9. **Corrected Evaluation Report**: [research_validation/synthetic_public_dataset_evaluation_report.md](research_validation/synthetic_public_dataset_evaluation_report.md)

---

## 9. Conclusion

The ObjectMemory project is complete. All code, datasets, evaluation harnesses, tests, demonstrations, and academic artifacts are verified, mathematically auditable, and reproducible from the repository source code.
