# ObjectMemory Studio: Comprehensive System Audit

**Audit Date:** 2026-10-20  
**Environment:** Windows 11, Python 3.11.9 (CPU-only execution), Node.js v24.14.1, npm 11.11.0  
**Repository Directory:** `C:\Users\Rishh\.gemini\antigravity\scratch\objectmemory`

---

## 1. Executive Summary

This system audit provides an exhaustive inspection of the ObjectMemory codebase, including its computer vision pipeline, temporal reasoning engine, persistent SQLite storage, FastAPI backend, and React + TypeScript frontend (**ObjectMemory Studio**).

### Execution Verification Summary
- **Python Unit & Regression Tests:** **184 tests passed** in **64.662s** (`OK`, 0 failures, 0 errors).
- **Frontend Production Build:** `tsc -b && vite build` built cleanly in **467ms** with **0 TypeScript or linting errors**.
- **Live Backend Integration Test:** All 10 live API endpoints verified (`tools/test_live_api.py`) with zero errors.

---

## 2. Architecture & Component Inventory

### 2.1 Core Computer Vision Pipeline
Located in [`src/objectmemory/pipeline/real_pipeline.py`](src/objectmemory/pipeline/real_pipeline.py), coordinated by `RealImagePipelineRunner`:
1. **Planar Image Co-Registration:** SIFT feature extraction, brute-force k-NN ratio test (Lowe's 0.75 threshold), RANSAC homography estimation ($p_{ref} \sim H \cdot p_{query}$).
2. **Real Object Detection:** Synchronous CPU inference via OpenCV DNN loading pretrained YOLOv8n ONNX (`models/yolov8n.onnx`, SHA1: `68f864475d06e2ec4037181052739f268eeac38d`).
3. **Canonical Projection:** Coordinate transformation of detected bounding boxes into the baseline coordinate frame using the verified homography.
4. **Appearance Signature Extraction:** 148-dimensional normalized feature vectors extracted from pixel crops (96-d 3D HSV color histogram + 48-d spatial Sobel gradient magnitude histograms + 4-d aspect/fill ratio).
5. **Cross-Visit Entity Association:** Kuhn-Munkres optimal assignment combining signature cosine similarity and canonical spatial overlap, with conservative cross-class fallback for detector label noise.
6. **Temporal Memory Engine:** Chronological multi-visit belief state machine enforcing strict monotonic transitions and auditable change events.

### 2.2 Image Registration Implementation
Located in [`src/objectmemory/registration/`](src/objectmemory/registration/):
- **Safeguards:** Explicit coordinate conventions $(w, h)$ and finite reprojection error validation.
- **Quality Gating:** Registration is accepted only if:
  - Homography matrix is non-singular and finite.
  - Inlier ratio $\ge 20\%$ (`min_registration_inlier_ratio = 0.20`).
  - Reprojection error mean $\le 4.0$ px (`max_reprojection_error = 4.0`).
  - Area scale ratio is bounded between 0.15 and 6.0 (prevents collapsed or blown-up quadrilaterals).
  - Overlap fraction $\ge 15\%$ of the reference area.
- When registration fails, canonical projection is safely suppressed to prevent spurious displacement calculations.

### 2.3 Object Detection Implementation
Located in [`src/objectmemory/detection/`](src/objectmemory/detection/):
- `OpenCVDNNDetector` runs ONNX models via `cv2.dnn.readNetFromONNX`.
- Operates strictly on CPU (latency ~115–150 ms per frame).
- Output is standardized as `Detection` dataclasses with `raw_bbox`, confidence score, label, and visual signature.
- Background exclusion safeguard: Classes such as `dining table` are excluded from foreground movement state transitions.

### 2.4 Signature Extraction & Cross-Visit Association
Located in [`src/objectmemory/temporal/associator.py`](src/objectmemory/temporal/associator.py):
- Decouples visual appearance from detector label.
- Computes cosine similarity between 148-d normalized appearance descriptors.
- **Conservative Cross-Class Matching:** When the detector flips an object's predicted class label (e.g. `book` detected as `laptop`), matches are accepted if:
  1. Spatial overlap is high (IoU $\ge 0.35$ with signature similarity $\ge 0.70$, or spatial IoU $\ge 0.75$).
  2. Association margin $\ge 0.15$ (prevents ambiguous assignment when multiple candidate objects compete).

### 2.5 Temporal Reasoning & State Transitions
Located in [`src/objectmemory/temporal/engine.py`](src/objectmemory/temporal/engine.py):
- **Epistemic States:** `CONFIRMED_PRESENT`, `CONFIRMED_MOVED`, `CONFIRMED_REMOVED`, `CONFIRMED_ADDED`, `PENDING_REMOVAL`, `INSUFFICIENT_EVIDENCE`, `AMBIGUOUS_IDENTITY`.
- **Displacement Baselines:** Tracks displacement relative to both `initial_bbox` and `last_reliable_bbox`.
- **Coordinate Drift Safeguard:** Displacements below `max_drift_step_displacement` (20 px) between consecutive visits that maintain $\text{IoU} \ge 0.35$ with initial position are classified as registration drift, preventing false `CONFIRMED_MOVED` events.
- **Boundary Clipping Safeguard:** Boxes touching image boundaries within 8 px margin require $\ge 100$ px displacement (`clipped_movement_displacement_threshold`) to confirm movement, preventing boundary truncation noise from causing false motion reports.
- **Removal Belief Progression:** Missing object requires consecutive verified vacant patches (`vacancy_similarity_max = 0.40`) across 2 reliable visits before confirming removal (`CONFIRMED_REMOVED`).

### 2.6 Database Schema & Persistence
Located in [`src/objectmemory/persistence/`](src/objectmemory/persistence/):
- SQLite schema version 2 enforcing `PRAGMA foreign_keys = ON;`.
- Tables: `sessions`, `entities`, `observations`, `change_events`, `pairwise_baseline_results`.
- Referential integrity: `ON DELETE RESTRICT` on historical tables guarantees audit trails cannot be silently deleted.
- Atomic transitions: `record_state_transition` executes entity update and event insertion within a single atomic database transaction.

### 2.7 FastAPI Backend
Located in [`src/objectmemory/api/`](src/objectmemory/api/):
- Endpoints:
  - System: `GET /api/health`
  - Workspaces: `POST /api/workspaces`, `GET /api/workspaces`, `GET /api/workspaces/{id}`, `DELETE /api/workspaces/{id}`
  - Ingest: `POST /api/workspaces/{id}/observations` (supports multipart PNG/JPEG, 25MB limit)
  - Observations: `GET /api/workspaces/{id}/observations`, `GET /api/workspaces/{id}/observations/{sid}`
  - Entities: `GET /api/workspaces/{id}/entities`, `GET /api/workspaces/{id}/entities/{eid}`
  - Events: `GET /api/workspaces/{id}/events`
  - Comparison: `GET /api/workspaces/{id}/compare`
  - Diagnostics: `GET /api/workspaces/{id}/diagnostics`
  - Static Files & Images: `GET /api/workspaces/{id}/images/{filename}`, `GET /` (SPA mount)
- Thread confinement resolved via per-workspace `threading.RLock()` and `check_same_thread=False`.
- Persistence durability: automatically restores reference image from session 1 on startup.

### 2.8 Frontend Architecture
Located in [`frontend/`](frontend/):
- Vite 8 + React 19 + TypeScript + Tailwind CSS v4.
- Zero mock data; communicates exclusively with `/api`.
- Views: Dashboard, Workspace Overview, Scene Comparison, Entity History, Timeline, Diagnostics.
- Coordinates canvas: preserves true image aspect ratios and scales bounding boxes accurately.

---

## 3. Automated Test Verification Record

### Test Execution Command
```powershell
$env:PYTHONPATH="src"
python -m unittest discover -s tests -p "test_*.py"
```

### Result
```
Ran 184 tests in 64.662s
OK
```

### Breakdown of Passing Tests
- `tests/test_domain_models.py`: 12 tests (timestamp normalization, coordinate bounds, unit intervals).
- `tests/test_persistence.py`: 18 tests (schema v2 migrations, foreign key constraints, atomic state transitions).
- `tests/test_registration.py`: 14 tests (planar homography, RANSAC gating, error bounds).
- `tests/test_detection.py`: 12 tests (OpenCV DNN inference, YOLOv8n validation, crop extraction).
- `tests/test_pairwise_baseline.py`: 15 tests (two-frame baseline comparisons).
- `tests/test_temporal_engine.py`: 32 tests (multi-visit state transitions, vacancy verification, removal progression).
- `tests/test_temporal_boundary_reappearance.py`: 14 tests (boundary clipping, reappearance logic, table exclusion).
- `tests/test_temporal_cross_class_association.py`: 12 tests (conservative cross-class matching, ambiguity margins).
- `tests/test_temporal_robustness.py`: 16 tests (lighting variation, viewpoint shifts).
- `tests/test_real_detector_pipeline.py`: 10 tests (end-to-end integration with YOLOv8n ONNX).
- `tests/test_illumination_sensitivity.py`: 8 tests (illumination sensitivity benchmarks).
- `tests/test_evaluator_adversarial.py`: 14 tests (adversarial dataset evaluation, out-of-order rejection).
- `tests/test_api_backend.py`: 7 comprehensive test suites (workspace isolation, upload processing, durability across restarts, error codes 400/404/413).

---

## 4. Known Limitations & Gaps

1. **Synthetic Pilot Data:** The 7 images in `data/real_workbench/visits` are synthetic collage fixtures, not physical camera captures. They provide a test fixture for the pipeline but do not establish real-world generalization.
2. **COCO 80 Class Taxonomy:** Specialized laboratory tools (multimeter, PCB, breadboard, soldering iron) fall outside the 80 COCO classes and are mapped to closest general classes (e.g. `laptop`, `book`, `scissors`) or missed.
3. **Monocular Planar Constraint:** Spatial comparisons rely on planar homography. Non-planar workbenches with significant 3D parallax require full 3D camera pose or depth sensors, which are outside the current CPU-only scope.
4. **Interactive 2.5D Visualization:** The application currently lacks a visual Spatial Memory Explorer for inspecting canonical scene layout and spatial memory beliefs.
5. **Evaluation Lab UI:** While CLI benchmark scripts exist, the web application lacks an interactive Evaluation Lab for configuring, executing, and exporting quantitative evaluation runs.

---

## 5. Audit Conclusion

The codebase is technically sound, modular, and fully tested. All 184 tests pass with zero regressions. The system is ready for the planned high-value enhancements:
- Phase 2: Browser & API end-to-end verification.
- Phase 3: Polished interface components (toasts, error boundaries, accessible tooltips).
- Phase 4: Advanced scene comparison with before/after slider.
- Phase 5: Spatial Memory Explorer (2.5D virtual workbench visualizer).
- Phase 6: Explainable temporal memory with an Evidence Ledger.
- Phase 7: Evaluation Lab with JSON/CSV export and dataset inspection.
