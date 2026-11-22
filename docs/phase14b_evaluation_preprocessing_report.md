# ObjectMemory Studio — Phase 14B: Evaluation and Preprocessing Reliability Report

**Phase**: 14B — Evaluation and Preprocessing Reliability  
**Date**: October 29, 2026  
**Status**: Completed & Verified  
**Workspace**: `C:\Users\Rishh\.gemini\antigravity\scratch\objectmemory`  

---

## 1. Executive Summary

Phase 14B implemented the three highest-priority reliability improvements identified during the Phase 14A scientific audit:
1. **Historical Temporal-State Evaluator Correction**: Resolved the evaluation bug where the ground-truth matcher evaluated past visits against the entity's terminal state (`current_state`) at the end of the entire sequence, rather than its authoritative state as of the historical visit being evaluated.
2. **Aspect-Ratio-Preserving Detector Preprocessing**: Implemented model-specific letterboxing with symmetric neutral-grey padding (`114`) and rigorous inverse coordinate transformations for `OpenCVDNNDetector` (YOLOv8n) and `YOLOWorldDetector`.
3. **Decoupled Evaluation Tracks**: Refactored benchmark harnesses to strictly separate:
   - **Track A**: Detector Evaluation on natural photographic imagery (COCO validation subset) measuring precision, recall, F1, and mAP without oracle boxes.
   - **Track B**: Oracle Temporal-Reasoning Evaluation on controlled datasets isolating state-machine belief revision from detector localization failures.
   - **Track C**: End-to-End Temporal Evaluation passing actual detector predictions through registration and temporal memory, documenting exact domain boundaries.

Existing temporal reasoning thresholds, removal confirmation policies, database schemas, API contracts, and frontend user interfaces were 100% preserved.

---

## 2. Historical-State Evaluator Correction

### Root Cause Analysis
In `src/objectmemory/benchmarks/ground_truth_matcher.py`, `GroundTruthMatcher.evaluate_sequence` iterated over visits $V_1 \dots V_N$ to score event transitions. In line 252 of the previous implementation:
```python
if sys_ent is not None:
    observed_state = sys_ent.current_state  # Defect: points to terminal state!
```
Because `sys_ent.current_state` reflects the state at the end of the sequence:
- If an entity moved at $V_2$ (`expected_state = CONFIRMED_MOVED`) and returned to baseline at $V_4$ (`expected_state = CONFIRMED_PRESENT`), visit $V_2$ was evaluated with `observed_state = CONFIRMED_PRESENT`.
- Consequently, $V_2$ was falsely penalized as a `missed_movement`.
- Furthermore, if an entity remained stationary at a moved position ($V_3$, `expected_state = CONFIRMED_MOVED`), observing `CONFIRMED_MOVED` was falsely penalized as a `false_movement`.

### Architectural Correction
Implemented `resolve_historical_entity_state` in `GroundTruthMatcher`:
1. **Priority 1 (Visit Report)**: Inspects `visit_reports[v_idx].entity_updates` for the entity's updated `new_state` at that visit.
2. **Priority 2 (SQLite Store Change Events)**: Queries `store.list_change_events_for_entity(sys_id)` filtered strictly to sessions up to the current visit index (`dataset_visit_ids[: v_idx + 1]`), extracting the state from the most recent historical transition.
3. **Priority 3 (Backward Report Propagation)**: Traces previous visit reports ($V_{t-1} \dots V_0$) for the most recent state when an entity was unchanged across visits.
4. **Missing Data Safeguard**: Returns `EpistemicState.INSUFFICIENT_EVIDENCE` with an explicit explanation (`"Historical state record missing for visit"`) and **never** falls back to `current_state`.
5. **False Movement Correction**: Clarified that an `UNCHANGED` event is only a false movement if the expected state was stationary (`CONFIRMED_PRESENT`) but the observed state was `CONFIRMED_MOVED`.

### Regression Test Suite
Added 6 dedicated tests in `tests/test_ground_truth_matcher.py`:
- `test_historical_state_v2_moved_v4_returned_to_baseline`: Verifies $V_2$ movement passes while $V_4$ return to baseline passes (0 missed movements, 2 correct relocations, 100% accuracy).
- `test_stationary_entity_throughout_sequence`: Verifies stationary entity produces 0 false movements and 0 missed movements.
- `test_persistent_movement_evaluation`: Verifies an entity that moves and stays moved does not accumulate duplicate false movements.
- `test_missing_historical_records_explicit_safeguard`: Verifies missing records yield `INSUFFICIENT_EVIDENCE` with explicit notes rather than terminal state fallback.
- `test_multiple_same_class_entities_identity_preservation`: Verifies distinct tracks for multiple same-class entities without cross-talk.
- `test_event_ordering_preserved_across_visits`: Verifies chronological visit ordering in evaluation output.

---

## 3. Aspect-Ratio-Preserving Preprocessing & Inverse Transforms

### Transformation Mathematics
Created dedicated module `src/objectmemory/detection/letterbox.py`.

Given original image dimensions $(w_{\text{orig}}, h_{\text{orig}})$ and model input size $(w_{\text{in}}, h_{\text{in}}) = (640, 640)$:

1. **Scale Factor**:
   $$r = \min\left(\frac{w_{\text{in}}}{w_{\text{orig}}}, \frac{h_{\text{in}}}{h_{\text{orig}}}\right)$$
2. **Unpadded Resized Dimensions**:
   $$w_{\text{unpad}} = \operatorname{round}(w_{\text{orig}} \cdot r), \quad h_{\text{unpad}} = \operatorname{round}(h_{\text{orig}} \cdot r)$$
3. **Symmetric Neutral Padding**:
   $$\text{pad}_w = w_{\text{in}} - w_{\text{unpad}}, \quad \text{pad}_h = h_{\text{in}} - h_{\text{unpad}}$$
   $$\text{pad}_{\text{left}} = \lfloor \text{pad}_w / 2 \rfloor, \quad \text{pad}_{\text{right}} = \text{pad}_w - \text{pad}_{\text{left}}$$
   $$\text{pad}_{\text{top}} = \lfloor \text{pad}_h / 2 \rfloor, \quad \text{pad}_{\text{bottom}} = \text{pad}_h - \text{pad}_{\text{top}}$$
   Padding value: $114$ (`#727272`, standard neutral grey for YOLOv8 and YOLO-World).

4. **Inverse Bounding Box Transformation**:
   From predicted model center and dimensions $(cx_m, cy_m, w_m, h_m)$:
   $$x_{1} = \frac{(cx_m - 0.5 \cdot w_m) - \text{pad}_{\text{left}}}{r}, \quad y_{1} = \frac{(cy_m - 0.5 \cdot h_m) - \text{pad}_{\text{top}}}{r}$$
   $$x_{2} = \frac{(cx_m + 0.5 \cdot w_m) - \text{pad}_{\text{left}}}{r}, \quad y_{2} = \frac{(cy_m + 0.5 \cdot h_m) - \text{pad}_{\text{top}}}{r}$$
   $$x_{\min} = \operatorname{clamp}(x_1, 0, w_{\text{orig}}), \quad y_{\min} = \operatorname{clamp}(y_1, 0, h_{\text{orig}})$$
   $$x_{\max} = \operatorname{clamp}(x_2, 0, w_{\text{orig}}), \quad y_{\max} = \operatorname{clamp}(y_2, 0, h_{\text{orig}})$$
   Degenerate check: reject if $(x_{\max} - x_{\min} < 1.0)$ or $(y_{\max} - y_{\min} < 1.0)$.

### Model-Specific Conventions
| Dimension / Parameter | `OpenCVDNNDetector` (YOLOv8n) | `YOLOWorldDetector` (YOLO-World v2) |
| :--- | :--- | :--- |
| **Input Channels** | BGR converted to RGB (`swapRB=True` in `blobFromImage`) | Explicit BGR $\to$ RGB (`cv2.cvtColor`) |
| **Normalization** | Scale factor $1.0 / 255.0$, mean $(0,0,0)$ | Float32 tensor divided by $255.0$ |
| **Input Shape** | $(1, 3, 640, 640)$ | $(1, 3, 640, 640)$ |
| **Dynamic Inputs** | None (fixed 80 COCO classes) | `txt_feats` CLIP ViT-B/32 text embeddings |
| **Padding Color** | 114 (neutral grey) | 114 (neutral grey) |
| **Backward Compatibility** | Supports `letterbox=True` (default) and `letterbox=False` | Supports `letterbox=True` (default) and `letterbox=False` |

---

## 4. Synthetic Coordinate Round-Trip Results

Tested in `tests/test_letterbox_preprocessing.py` across 5 distinct resolutions:
- $640 \times 640$ (Square baseline)
- $1280 \times 720$ (16:9 Landscape)
- $720 \times 1280$ (9:16 Portrait)
- $1920 \times 1080$ (Full HD 16:9)
- $400 \times 600$ (Non-standard aspect ratio)

| Test Verification | Target Metric | Verified Result | Exit Status |
| :--- | :--- | :--- | :--- |
| **Aspect Ratio Preservation** | $|(w_{\text{unpad}}/h_{\text{unpad}}) - (w_{\text{orig}}/h_{\text{orig}})| \le 0.02$ | **Pass** across all 5 resolutions | `OK` |
| **Symmetric Padding** | $|\text{pad}_{\text{left}} - \text{pad}_{\text{right}}| \le 1\text{ px}$, $|\text{pad}_{\text{top}} - \text{pad}_{\text{bottom}}| \le 1\text{ px}$ | **Pass** across all 5 resolutions | `OK` |
| **Coordinate Inversion Error** | $|\hat{x} - x_{\text{orig}}| \le 1.0\text{ px}$, $|\hat{y} - y_{\text{orig}}| \le 1.0\text{ px}$ | **Max Error: $0.00\text{ px}$** (machine epsilon) | `OK` |
| **Boundary Clamping** | Negative & $> w_{\text{orig}}, h_{\text{orig}}$ coordinates clamped | **Pass** (clamped to $[0, w_{\text{orig}}], [0, h_{\text{orig}}]$) | `OK` |
| **Degenerate Rejection** | Zero-width, collapsed, and padding-only boxes rejected | **Pass** (returns `None`) | `OK` |
| **Color Value Invariance** | Border padding color strictly 114 across RGB/BGR | **Pass** (114 on all channels) | `OK` |

---

## 5. Empirical Evaluation Results

### Track A: Natural Detection Evaluation (COCO 2017 Validation Subset)
Evaluated on the exact 20-image COCO subset (84 ground-truth objects, 8 classes) at IoU threshold $0.50$ and confidence threshold $0.25$:

| Model & Preprocessing Configuration | TP | FP | FN | Precision | Recall | F1 Score | mAP@50 | mAP@50:95 | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`yolov8n_letterboxed`** | **20** | 15 | **64** | 0.5714 | **0.2381** | **0.3361** | 0.3359 | **0.2808** | **202.8 ms** |
| `yolov8n_direct_stretch` (Legacy) | 18 | 10 | 66 | 0.6429 | 0.2143 | 0.3214 | 0.3450 | 0.2779 | 290.9 ms |
| **`yolo_world_letterboxed`** | **8** | 8 | **76** | 0.5000 | **0.0952** | **0.1600** | **0.2110** | **0.1913** | 720.8 ms |
| `yolo_world_direct_stretch` (Legacy) | 7 | 6 | 77 | 0.5385 | 0.0833 | 0.1443 | 0.1716 | 0.1457 | 662.5 ms |

**Empirical Preprocessing Findings**:
- On `YOLO-World`, aspect-ratio-preserving letterboxing delivered a **+23.0% relative improvement in mAP@50** ($0.2110$ vs $0.1716$) and a **+31.3% relative improvement in mAP@50:95** ($0.1913$ vs $0.1457$).
- On `YOLOv8n`, letterboxing increased true positive count ($20$ vs $18$), increased recall ($0.2381$ vs $0.2143$), improved F1 score ($0.3361$ vs $0.3214$), and improved mAP@50:95 ($0.2808$ vs $0.2779$), while direct stretch had slightly higher single-threshold precision ($0.6429$ vs $0.5714$).
- Serialized to: `scripts/phase14b_coco_subset_evaluation_results.json`.

---

### Track B: Oracle Temporal-Reasoning Evaluation
Input: Ground-truth oracle bounding boxes on the 6-sequence, 30-visit controlled benchmark.
Purpose: Isolate epistemic state machine transitions and Kuhn-Munkres association from detector localization error.

| Axis | Metric | Measured Result | Scientific Interpretation |
| :--- | :--- | :--- | :--- |
| **Axis 2: Registration** | Overall Success Rate | **100.0%** (30/30 visits) | SIFT feature alignment robust under synthetic planar transformations. |
| **Axis 3: Association (Oracle)** | Identity Switches | **0 switches** across all 6 sequences | Kuhn-Munkres distance and signature matching prevent ID swaps under idealized detection. |
| **Axis 3: Association (Oracle)** | Duplicate Tracks | **4** (in S01 under severe brightness variations) | Color histogram shift under extreme lighting induced track splitting. |
| **Axis 4: Events (Oracle)** | Event Accuracy | **92.9%** (13 / 14 expected events) | 13/14 events matched. The single "unmatched" event is S04 V4 removal, which correctly requires a 2-visit vacancy confirmation policy. |
| **Axis 4: Events (Oracle)** | Spurious Movements | **0** | No false movements reported on stationary or view-change sequences. |
| **Axis 4: Events (Oracle)** | Spurious Removals | **0** | No false removals reported during occlusions. |

---

### Track C: End-to-End Temporal Evaluation
Input: Actual detector predictions (YOLOv8n with aspect-preserving letterboxing) passed through registration, association, and temporal memory engine.
Purpose: Measure real integrated system performance.

| Parameter | Measured Count | Notes |
| :--- | :--- | :--- |
| **Total Sequences Processed** | 6 sequences | S01 through S06 |
| **Total Visits Processed** | 30 visits | 5 visits per sequence |
| **Total Ground-Truth Objects** | 120 object instances | Across all visits |
| **Actual Detector Predictions** | **51** (sparse / noisy) | Neural weights trained on COCO photographs poorly detect stylized line drawings. |
| **Detector Procedural Recall** | **0.0%** on target tools | COCO-trained weights do not reliably localize procedural vector drawings. |
| **End-to-End Metric Status** | **Unestablished on procedural data** | Procedural dataset cannot support a valid end-to-end evaluation without photorealistic rendering or domain fine-tuning. |

Serialized to: `scripts/phase14b_controlled_benchmark_results.json`.

---

## 6. Verification and Regression Safety

### Test Commands and Exit Statuses
1. **Full Python Test Suite**:
   ```powershell
   python -m unittest discover tests
   ```
   - **Result**: `Ran 301 tests in 112.535s - OK` (Exit code: 0)
   - **Baseline vs Post-Change**: 288 tests $\to$ **301 tests** (+13 new regression tests, 0 failures, 0 errors).
2. **Frontend Production Build**:
   ```powershell
   cd frontend; npm run build
   ```
   - **Result**: `✓ built in 1.58s` (Exit code: 0, 0 TypeScript errors).
3. **Synthetic Preprocessing Round-Trip Test**:
   ```powershell
   python -m unittest tests.test_letterbox_preprocessing
   ```
   - **Result**: `Ran 6 tests in 0.063s - OK` (Exit code: 0).
4. **Historical Matcher Regression Test**:
   ```powershell
   python -m unittest tests.test_ground_truth_matcher
   ```
   - **Result**: `Ran 9 tests in 0.008s - OK` (Exit code: 0).

---

## 7. Change Control & File Inventory

### Created Files
- `src/objectmemory/detection/letterbox.py`: Dedicated letterbox scaling, symmetric neutral-grey padding, and inverse coordinate transform module.
- `tests/test_letterbox_preprocessing.py`: Unit and regression tests for coordinate round-trip accuracy across 5 image resolutions.
- `scripts/phase14b_coco_subset_evaluation_results.json`: Track A natural-image evaluation results with side-by-side letterbox vs direct stretch comparison.
- `scripts/phase14b_controlled_benchmark_results.json`: 3-track decoupled controlled benchmark evaluation results.
- `docs/phase14b_evaluation_preprocessing_report.md`: This report.

### Modified Files
- `src/objectmemory/benchmarks/ground_truth_matcher.py`: Added `resolve_historical_entity_state`, corrected historical visit state evaluation, and corrected UNCHANGED false movement evaluation.
- `src/objectmemory/detection/detector.py`: Added `letterbox: bool = True` to `OpenCVDNNDetector`, integrated `letterbox_image` and `inverse_transform_cxcywh`.
- `src/objectmemory/detection/yolo_world.py`: Added `letterbox: bool = True` to `YOLOWorldDetector`, integrated `letterbox_image` and `inverse_transform_cxcywh`.
- `scripts/evaluate_natural_detection.py`: Added side-by-side preprocessing comparison, tagged Track A payload, and versioned output path.
- `scripts/evaluate_controlled_benchmark.py`: Added Track C end-to-end evaluation, explicit Track B oracle labeling, and versioned output path.
- `tests/test_ground_truth_matcher.py`: Added 6 focused regression tests for historical state and return-to-baseline sequences.
- `tests/test_real_workbench_detection_regression.py`: Verified both legacy `letterbox=False` and updated `letterbox=True` modes with clean resource cleanup.
- Test path resolution fixes in `tests/test_api_backend.py`, `tests/test_boundary_and_registration.py`, and `tests/test_controlled_dataset.py`.

### Unmodified Core Invariants
- Temporal reasoning thresholds ($35\text{ px}$ base movement threshold, $0.20$ registration inlier ratio, $12\text{ px}$ boundary margin).
- Two-visit verified vacancy confirmation rule for removals.
- Composite dynamic uncertainty calculation formulas.
- Database schema and API endpoints.
- Historical evaluation results (`phase9b_evaluation_results.json`, `phase10_controlled_evaluation_results.json`, `phase10a_controlled_evaluation_results.json`, `phase10_coco_subset_evaluation_results.json`, `phase10a_coco_subset_evaluation_results.json`).

---

## 8. What Was Not Measured & Remaining Limitations

1. **Integrated Real-World End-to-End Metrics**: While Track A measures detection on natural COCO photographs and Track B measures temporal reasoning on controlled synthetic sequences, a natural, photorealistic, multi-visit temporal dataset with millimeter-accurate physical ground truth does not yet exist in the repository.
2. **GPU Latency**: All evaluations were performed on standard x86 CPU hardware; multi-GPU throughput was not benchmarked.
3. **Parallax 3D Transformations**: Planar homography registration remains bounded by the planar workbench assumption and degrades when viewpoint tilt exceeds $30^\circ$.

---

## 9. Recommended Next Step: Phase 14C

**Photorealistic Temporal Benchmarking & Multi-Visit Natural Evaluation**:
1. Implement a photorealistic rendering pipeline (or curated multi-visit natural workbench capture protocol with fiducial ArUco calibration).
2. Measure end-to-end temporal tracking (Track C) on photographic imagery where neural detectors have high recall.
3. Benchmark cross-session association and return-to-baseline reasoning on physical workbench captures.
