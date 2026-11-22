# Phase 14C: Physical Workbench Benchmarking & End-to-End Evaluation Report

**Document Version**: 1.0.0  
**Date**: October 29, 2026  
**Engineering Team**: ObjectMemory Computer Vision & Reliability Engineering  
**Workspace**: `C:\Users\Rishh\.gemini\antigravity\scratch\objectmemory`  
**Capture Status**: `AWAITING_PHYSICAL_DATA` (Harness & Ingestion Fully Implemented & Validated)

---

## 1. Executive Summary & Epistemic Boundaries

Phase 14C implements a reproducible physical workbench benchmark dataset infrastructure, validation suite, and three-track evaluation harness for ObjectMemory Studio.

### Scientific Fact & Status Classification Convention
- **[IMPLEMENTED]**: Core functionality written, integrated, and verified in code.
- **[TESTED]**: Unit and regression tests passing with logged exit codes.
- **[PENDING]**: Real-world evaluation awaiting user capture of physical photographs.
- **[LIMITATIONS]**: Documented physical, optical, and geometric bounds.

> [!IMPORTANT]
> **Scientific Integrity Safeguard**:  
> In strict accordance with Phase 14C rules, **no fabricated bounding boxes, fake detections, synthetic metrics, or imagined physical measurements are presented as real-world results**.  
> The ingestion harness, validation scripts, annotation schemas, and decoupled three-track evaluation pipeline are completely implemented and unit-tested. Because the physical photographs have not yet been captured by the user, the benchmark status is recorded honestly as `AWAITING_PHYSICAL_DATA`. Real-world performance claims are deferred until genuine photographic evidence is provided.

---

## 2. Architecture and Data Flow

The physical benchmarking harness implements the complete visual memory pipeline:

```
[Physical Photographic Frame (1920x1080)]
                   │
                   ▼
       [Dataset Manifest Loader]
       (PhysicalWorkbenchDataset)
                   │
       ┌───────────┴───────────────────────────────┐
       ▼                                           ▼
[Track A: Natural Detection]            [Track B & C: Pipeline]
  - Letterbox (aspect-preserved)          │
  - YOLOv8n & YOLO-World                  ├─► [Track B: Oracle Inputs]
  - Greedy 1-to-1 Bipartite Match         │     - Ground-Truth Boxes fed directly to Engine
  - TP / FP / FN / Precision / Recall     │     - Isolates temporal belief transitions
  - Mean Inference Latency                │
                                          └─► [Track C: Real Detector Inputs]
                                                │
                                                ▼
                                    [Planar Homography Registration]
                                      - SIFT/ORB feature matching
                                      - ArUco corner verification (if present)
                                      - Inlier ratio gating (>= 0.15)
                                                │
                                                ▼
                                    [Canonical Coordinate Projection]
                                      - Project query box via H matrix
                                                │
                                                ▼
                                    [Visual Signature Extractor]
                                      - 148-d HSV/LBP/texture descriptor
                                                │
                                                ▼
                                    [Entity Association & Temporal Engine]
                                      - Multi-visit track maintenance
                                      - Epistemic state machine updates
                                      - Vacancy confirmation (2-visit rule)
                                                │
                                                ▼
                                    [Atomic SQLite Persistence]
                                      - Entities, Observations, ChangeEvents
```

---

## 3. Dataset Specification: `data/physical_workbench_v1/`

### 3.1 Directory Layout `[IMPLEMENTED]`
- `data/physical_workbench_v1/manifest.json`: Master protocol manifest with 7 scenarios, 35 visits, 175 annotations.
- `data/physical_workbench_v1/README.md`: Laboratory capture manual and setup guide.
- `data/physical_workbench_v1/images/`: Target folder for photographic captures.
- `data/physical_workbench_v1/annotations/`: Per-scenario ground-truth annotations.
- `data/physical_workbench_v1/metadata/`: Camera configuration and ArUco board calibration spec.

### 3.2 Five Canonical Objects
1. `cup_01` (`cup`): White ceramic mug with handle ($\approx 8\text{ cm}$ diameter, $10\text{ cm}$ height).
2. `book_01` (`book`): Navy paperback book ($\approx 13\text{ cm} \times 20\text{ cm}$).
3. `scissors_01` (`scissors`): Metal craft scissors with red handles ($\approx 18\text{ cm}$ length).
4. `bottle_01` (`bottle`): Clear/metallic reusable water bottle ($\approx 7\text{ cm}$ diameter, $22\text{ cm}$ height).
5. `phone_01` (`cell phone`): Dark smartphone in matte protective case ($\approx 7.5\text{ cm} \times 15\text{ cm}$).

### 3.3 Seven Required Scenarios (5 Visits Each = 35 Visits Total)
- **S01 — Stationary Baseline**: All 5 objects stationary over 5 visits. (Tests false movement rate, false removal rate, identity stability).
- **S02 — Temporary Movement and Return**: Mug relocated $>35\text{ px}$ in V3; returned to exact baseline in V4. (Tests relocation recognition and return-to-baseline reasoning).
- **S03 — Persistent Relocation**: Smartphone moved $>50\text{ px}$ in V2; remains in new position for V3-V5. (Tests persistent change tracking without track bifurcation).
- **S04 — Permanent Removal**: Water bottle removed at V3 and remains absent. (Tests two-visit vacancy confirmation policy and confirmation delay).
- **S05 — Partial Occlusion**: Paperback book partially occluded (~30%) in V2, heavily occluded (~70%) in V3, revealed in V4. (Tests distinguishing temporary invisibility from removal).
- **S06 — Illumination Stress**: Scene lighting systematically altered across 5 visits (nominal $\to$ dimmed $\to$ directional side lamp $\to$ spotlight $\to$ nominal). (Tests false movement invariance under illumination shifts).
- **S07 — Viewpoint Perturbation**: Camera pan, tilt, and height jitter while workbench objects remain stationary. (Tests planar registration limits).

---

## 4. Evaluation Tracks

### Track A — Natural-Image Object Detection `[IMPLEMENTED & TESTED]`
- Evaluates `yolov8n` and `yolo-world` (both using aspect-preserving letterbox preprocessing).
- Uses class-aware greedy 1-to-1 bipartite matching at $\text{IoU} \ge 0.50$.
- Computes TP, FP, FN, Precision, Recall, F1 score per class and overall.
- Measures inference latency (pre-loop warm-up pass + timed passes).
- **Status**: Ready. Recorded as `PENDING_PHYSICAL_DATA` until images are placed in `images/`.

### Track B — Oracle Temporal Reasoning `[IMPLEMENTED & TESTED]`
- Passes manual ground-truth bounding boxes directly into `TemporalMemoryEngine`.
- Isolates multi-visit state transitions from upstream detector localization noise.
- Measures movement detection, return-to-baseline recognition, persistent relocation, occlusion handling, and vacancy confirmation delay.
- **Status**: Ready. Validated via automated unit test fixture sequence with 100% event accuracy ($4/4$ events passed, 0 identity switches, 0 duplicate tracks).

### Track C — End-to-End Physical Evaluation `[IMPLEMENTED & TESTED]`
- Feeds actual detector predictions through the complete pipeline:
  `RealImagePipelineRunner` $\to$ `PlanarRegistrar` $\to$ canonical projection $\to$ 148-d visual signature $\to$ `TemporalMemoryEngine` $\to$ SQLite.
- Measures registration success rate, identity switches, duplicate tracks, false movements, false removals, and confirmation delay.
- **Status**: Ready. Recorded as `PENDING_PHYSICAL_DATA` until images are placed in `images/`.

---

## 5. Automated Validation & Test Suite Results

### 5.1 Verification Commands and Exit Codes

| Command | Working Directory | Exit Status | Result |
| :--- | :--- | :---: | :--- |
| `python scripts/validate_physical_workbench_dataset.py` | Workspace root | `0` | All 7 sequences, 35 visits, 175 annotations validated. Status: `AWAITING_PHYSICAL_DATA`. |
| `python scripts/evaluate_physical_workbench.py` | Workspace root | `0` | Ingestion verified. Wrote `scripts/phase14c_physical_workbench_results.json` (`AWAITING_PHYSICAL_DATA`). |
| `python -m unittest tests.test_physical_workbench_dataset tests.test_physical_workbench_evaluation` | Workspace root | `0` | **14 tests passed** in 0.477s. |
| `python -m unittest discover tests` | Workspace root | `0` | **315 tests passed** in full suite (0 regressions). |

### 5.2 Test Coverage Highlights
- **Manifest Integrity**: Validates canonical scenarios, 5 visits each, sequential indices, non-chronological order detection.
- **Geometric Bounds**: Detects out-of-bounds boxes, inverted/malformed coordinates ($x_{\min} \ge x_{\max}$), and coordinate bounds.
- **Absence Semantics**: Strictly enforces distinct handling of occlusion (`PARTIALLY_OCCLUDED`), unobserved evidence (`INSUFFICIENT_EVIDENCE`), and confirmed removal (`ABSENT`). Catches contradictions (e.g. `ABSENT` paired with non-null bounding box).
- **ArUco Marker Calibration**: Tests `detect_aruco_corner_markers` with graceful degradation when markers are absent.
- **Zero-Observation Math**: Verifies that empty datasets or missing classes return `"N/A"` rather than raising `ZeroDivisionError` or reporting misleading $0.0\%$.
- **Historical Immutability**: Proves that all historical result files (`phase14b_*.json`, `phase10_*.json`, `phase9b_*.json`) were preserved unmodified.

---

## 6. Physical Data Capture & Import Instructions

To execute the physical benchmark on real images:

### Step 1: Camera & Workspace Setup
1. Position the 5 objects on a textured workbench surface.
2. (Optional but recommended) Tape ArUco markers with IDs 0, 1, 2, 3 at the 4 corners of the workspace.
3. Mount the camera on a rigid tripod looking at a $45^\circ\text{--}60^\circ$ oblique angle.
4. Lock camera manual exposure, manual focus, and daylight white balance.

### Step 2: Capture Photographs
Capture 5 photographs for each of the 7 scenarios following the protocol in [`data/physical_workbench_v1/README.md`](data/physical_workbench_v1/README.md).  
Save images to `data/physical_workbench_v1/images/` using the standard filenames:
- `S01_stationary_baseline_visit_01.jpg` ... `_visit_05.jpg`
- `S02_temporary_movement_return_visit_01.jpg` ... `_visit_05.jpg`
- `S03_persistent_relocation_visit_01.jpg` ... `_visit_05.jpg`
- `S04_permanent_removal_visit_01.jpg` ... `_visit_05.jpg`
- `S05_partial_occlusion_visit_01.jpg` ... `_visit_05.jpg`
- `S06_illumination_stress_visit_01.jpg` ... `_visit_05.jpg`
- `S07_viewpoint_perturbation_visit_01.jpg` ... `_visit_05.jpg`

### Step 3: Run Validation & Visual Overlay Inspection
```powershell
python scripts/validate_physical_workbench_dataset.py --dataset-dir data/physical_workbench_v1 --require-images --overlay-dir data/physical_workbench_v1/overlays
```

### Step 4: Execute Decoupled Three-Track Evaluation
```powershell
python scripts/evaluate_physical_workbench.py --dataset-dir data/physical_workbench_v1 --output-file scripts/phase14c_physical_workbench_results.json
```

---

## 7. Known Limitations & Technical Boundaries

1. **Planar Homography Assumption**: The current registration engine estimates a $3 \times 3$ projective homography matrix $H$. While valid for objects resting on a planar table observed from moderate angles, it does not model 3D parallax for tall objects or large camera baseline movements.
2. **Fixed Movement Threshold**: The $35\text{ px}$ movement threshold is calibrated for nominal $1920 \times 1080$ images. If images of different resolutions are used, pixel thresholds should be scaled accordingly.
3. **Vacancy Confirmation Delay**: By design, `TemporalMemoryEngine` enforces a 2-visit confirmation delay before transitioning an unobserved entity from `PENDING_REMOVAL` to `CONFIRMED_REMOVED`. In S04, an immediate removal at V3 is expected to confirm at V4.
4. **Hardware Execution Profile**: Designed strictly for CPU-only laptop environments (tested on Windows AMD Ryzen 7 / Intel i7 class hardware, 0 GPU dependencies).
