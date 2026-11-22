# ObjectMemory Studio — Phase 14A Scientific Validation, Reliability Testing, and Research Benchmarking Audit

**Document Type:** Scientific Validation & Quality Engineering Audit Report  
**Author:** Antigravity Senior Computer Vision Engineer & Software Quality Auditor  
**Audit Date:** 2026-10-21  
**Project Workspace:** `C:\Users\Rishh\.gemini\antigravity\scratch\objectmemory`  
**Phase Mode:** Read-Only Investigation (Zero Code, Dataset, or Configuration Modifications)

---

## 1. Executive Summary

This read-only audit provides an exhaustive, evidence-based evaluation of the ObjectMemory system. ObjectMemory is designed to compare physical scenes across successive observation visits to detect, track, and explain persistent environmental changes (e.g. object movement, addition, and removal) while filtering out transient visual nuisances (lighting shifts, temporary partial occlusions, camera viewpoint variations, and field-of-view boundary clipping).

### Summary of Audit Verdicts
1. **Architecture & Software Engineering**: **STRONG & VERIFIED.** The modular design separates object detection, SIFT-based planar homography registration, bipartite Kuhn-Munkres cross-visit entity association, and an epistemic state machine (`CONFIRMED_PRESENT`, `CONFIRMED_MOVED`, `PENDING_REMOVAL`, `CONFIRMED_REMOVED`, `AMBIGUOUS_IDENTITY`). All 288 Python unit/integration tests and 5 Playwright end-to-end tests execute and pass cleanly.
2. **Temporal Reasoning Logic**: **PARTIALLY VERIFIED.** The epistemic transition engine enforces causal monotonicity (rejecting out-of-order visits), multi-visit vacancy confirmation before removal, and adaptive movement thresholds ($T_{\text{effective}} = T_{\text{base}} + \Delta U$). However, an audit of `GroundTruthMatcher` uncovered a critical evaluation bug: sequence evaluations check each entity's *final* state rather than its historical state as of each visit, distorting temporary movement and return-to-baseline scoring.
3. **Detection Interoperability**: **VERIFIED WITH DOCUMENTED DEFICIENCIES.** Both YOLOv8n (OpenCV DNN CPU) and YOLO-World v2 Small (ONNX Runtime CPU with CLIP ViT-B/32 text embeddings) execute correctly on CPU. However, both models use anisotropic direct stretching to $640 \times 640$ rather than aspect-ratio-preserving letterboxing, degrading detection precision on non-square images. Furthermore, on the 30-image controlled benchmark, both models achieve 0.0 mAP because procedural illustrations fail to activate models trained on natural photographic distributions.
4. **Dataset Provenance & Benchmark Validity**: **PROVISIONAL / METROLOGICALLY CONSTRAINED.**
   - The 7-image `real_workbench` pilot consists of user-supplied cropped smartphone photos without EXIF metadata, featuring a field-of-view shift between visits 3 and 4 ($780 \times 636$ vs $584 \times 576$), with bounding boxes hand-estimated from 10% grid overlays ($\pm 15\text{ px}$ uncertainty). It cannot be characterized as a ground-truth-controlled physical benchmark.
   - The 30-image `controlled_temporal_v1` benchmark uses 2D vector-style procedural drawings. Temporal reasoning is evaluated on oracle ground-truth detections rather than end-to-end detector outputs.
   - The 20-image `coco_val2017_subset` is authentic natural photography, but with only 84 ground-truth instances across 8 classes, sample size is insufficient for formal statistical generalization.

---

## 2. Actual Architecture and Data Flow

The data flow connects raw image inputs to persisted state transitions, benchmark metrics, and UI evidence ledgers:

```
[Raw Visit Image (BGR/PNG)]
             │
             ├──► [PlanarRegistrar (SIFT + RANSAC)] ──► [RegistrationResult (H, inlier_ratio, overlap_mask)]
             │                                                         │
             ├──► [ObjectDetector (YOLOv8n / YOLO-World)]             │
             │           │                                             │
             │           ▼                                             │
             │    [DetectionBatch (raw_bbox, conf, label)]             │
             │           │                                             │
             │           ├──► [transform_bbox_to_canonical(H)] ◄───────┘
             │           │           │
             │           ▼           ▼
             │    [Aligned Detections (canonical_bbox, raw_bbox)]
             │           │
             ▼           ▼
   [VisualSignatureExtractor (HSV hist + spatial moments)]
             │
             ▼
   [CrossVisitEntityAssociator (Kuhn-Munkres Assignment)]
   - Category compatibility check
   - Cosine visual signature similarity
   - Spatial overlap (IoU) & centroid distance
   - Ambiguity margin gating (prevents false mergers)
             │
             ▼
   [TemporalMemoryEngine.process_visit()]
   - Monotonic causal order check (timestamp >= latest)
   - Dynamic threshold: T_effective = T_base + Delta_U(reproj, inliers, cov, scale)
   - Return-to-baseline evaluation (baseline_disp <= T_effective)
   - Boundary clipping gating (border_margin_px = 8)
   - Verified vacancy checking (aligned image patch similarity < 0.60)
   - 2-stage removal progression: PRESENT -> PENDING_REMOVAL -> REMOVED
             │
             ├──► [SQLiteMemoryStore (Atomic Transactions)]
             │    ├── sessions
             │    ├── workbench_entities
             │    ├── observations
             │    └── change_events
             │
             ├──► [REST API (FastAPI)] ──► [React UI]
             │                             ├── Dual-Canvas Synchronized Pan/Zoom
             │                             ├── Interactive Wipe Slider
             │                             ├── Scientific Evidence Ledger
             │                             └── HTML5 Canvas Entity Crop Gallery
             │
             └──► [GroundTruthMatcher / Evaluator] ──► [Benchmark Results JSON]
```

---

## 3. Verified Implementation Inventory

The 17 architectural aspects specified in the audit instructions were systematically inspected in source code:

| # | Subsystem / Mechanism | Source File & Class/Function | Actual Implementation Status | Existing Test Coverage | Known Limitations & Verification State |
|---|---|---|---|---|---|
| 1 | **Object Detection & Preprocessing** | `src/objectmemory/detection/detector.py` (`OpenCVDNNDetector`) & `yolo_world.py` (`YOLOWorldDetector`) | Anisotropic direct resize to $640 \times 640$ via `cv2.dnn.blobFromImage` / `cv2.resize`. Normalizes to $[0, 1]$. OpenCV CPU DNN for YOLOv8n; ONNX Runtime CPU + quantized CLIP for YOLO-World. | `test_detection.py`, `test_yolo_world_detector.py` | **Verified.** Direct resize distorts aspect ratio of non-square images. Stylized illustrations produce 0 detections. |
| 2 | **Detection-to-Supervision Conversion** | `src/objectmemory/detection/supervision_adapter.py` (`to_supervision_detections`) | Maps `DetectionBatch` to `sv.Detections(xyxy, confidence, class_id)`. Validates coordinate finiteness and non-inversion. | `test_supervision_adapter.py` | **Verified.** Strict validation rejects non-finite coordinates. Isolates Supervision behind adapter. |
| 3 | **Ground-Truth Loading & Matching** | `src/objectmemory/benchmarks/ground_truth_matcher.py` (`GroundTruthMatcher`) | Bipartite matching via Kuhn-Munkres on cost matrix derived from canonical bounding box IoU and centroid distance. | `test_ground_truth_matcher.py` | **Partially Verified.** Initial mapping works, but `evaluate_sequence` checks `sys_ent.current_state` (final state) instead of historical state per visit. |
| 4 | **Image Registration & Canonical Coordinates** | `src/objectmemory/registration/registrar.py` (`PlanarRegistrar`), `src/objectmemory/spatial/transform.py` | Computes $3 \times 3$ projective homography $H$ using SIFT (ORB fallback) + RANSAC. Maps query points $p_{\text{ref}} \sim H p_{\text{query}}$. Computes spatial coverage & overlap mask. | `test_registration.py`, `test_keyframe_registration.py` | **Verified.** Assumes planar workbench geometry. Cannot reconstruct 3D non-planar objects or handle extreme out-of-plane rotations. |
| 5 | **Cross-Visit Object Association** | `src/objectmemory/temporal/associator.py` (`CrossVisitEntityAssociator`) | Category-constrained Kuhn-Munkres assignment. Combines visual signature cosine similarity ($w=0.7$) and canonical spatial IoU/distance ($w=0.3$). Gated on ambiguity margin ($0.10$). | `test_temporal_cross_class_association.py` | **Verified.** Effectively separates same-class instances. Ambiguity margin halts premature identity mergers. |
| 6 | **Persistent Identity Management** | `src/objectmemory/temporal/engine.py`, `src/objectmemory/domain/models.py` (`WorkbenchEntity`) | Decouples persistent entity identities (`ent_session_idx`) from transient frame detection boxes. Persists initial and last reliable canonical coordinates. | `test_domain_models.py`, `test_persistence.py` | **Verified.** Tracks objects across visits even through label classification noise. |
| 7 | **Temporal State Transitions** | `src/objectmemory/temporal/engine.py`, `src/objectmemory/domain/states.py` (`EpistemicState`) | 5 primary states: `CONFIRMED_PRESENT`, `CONFIRMED_MOVED`, `PENDING_REMOVAL`, `CONFIRMED_REMOVED`, `INSUFFICIENT_EVIDENCE`. Enforces atomic transitions via SQLite store. | `test_temporal_engine.py` | **Verified.** Transitions are audited with explicit reason codes and timestamps. |
| 8 | **Movement Detection & Adaptive Thresholds** | `src/objectmemory/temporal/engine.py` (`compute_effective_movement_threshold`, `_evaluate_displacement`) | $T_{\text{effective}} = T_{\text{base}} + \Delta U$, where $\Delta U = f(\text{reprojection error}, \text{inlier ratio}, \text{spatial coverage}, \text{scale ratio})$. Base threshold: 35 px. Max penalty: 45 px. | `test_temporal_reasoning_reliability.py` | **Verified.** Suppresses false movement decisions under degraded registration. |
| 9 | **Return-to-Baseline Detection** | `src/objectmemory/temporal/engine.py` (`_evaluate_displacement`) | Separates consecutive step displacement from baseline displacement relative to initial position. Triggers `is_return_to_baseline = True` and restores `CONFIRMED_PRESENT`. | `test_temporal_reasoning_reliability.py`, `test_ground_truth_matcher.py` | **Verified.** Resolves the historical defect where returning objects were scored as missed movements. |
| 10 | **Occlusion & Temporary Absence** | `src/objectmemory/temporal/engine.py` (lines 980-996) | Checks canonical bbox overlap with other detected objects (IoU > 0.20). Flags `PARTIALLY_OCCLUDED` and suspends removal progression. | `test_removal_and_occlusion_evaluation.py` | **Verified.** Prevents temporary occlusion from being classified as removal. |
| 11 | **Removal Confirmation** | `src/objectmemory/temporal/engine.py` (lines 997-1070) | Requires 2 consecutive visits of verified vacant patch appearance ($\text{cosine similarity} < 0.60$) or 3 consecutive unobserved visits under reliable registration. Direct jumps from PRESENT to REMOVED are forbidden. | `test_removal_and_occlusion_evaluation.py`, `test_phase10a_corrections.py` | **Verified.** Removal is strictly gated. Verified failure on synthetic S04 v4 where absence was unconfirmed. |
| 12 | **Boundary Clipping & Field-of-View** | `src/objectmemory/temporal/engine.py` (`_is_bbox_clipped`), `src/objectmemory/detection/pipeline.py` | Bounding box within 8 px of image boundary flagged as clipped. Displacement threshold increased to 50 px to avoid treating camera framing truncation as movement. | `test_boundary_and_registration.py`, `test_temporal_boundary_reappearance.py` | **Verified.** Truncated edge centroid shifts are properly distinguished from genuine relocations. |
| 13 | **Uncertainty & Registration Quality Gating** | `src/objectmemory/temporal/engine.py` (`_is_registration_reliable`) | If `inlier_ratio < 0.20` or registration fails, spatial comparisons are disabled (`is_spatial_reliable = False`), displacements are suppressed, and absence cannot confirm removal. | `test_boundary_and_registration.py` | **Verified.** Prevents geometric artifacts from generating spurious change events. |
| 14 | **Change-Event Evidence Generation** | `src/objectmemory/temporal/engine.py` (`ChangeEvent`) | Generates structured `supporting_evidence` containing displacement distances, adaptive thresholds, SIFT inlier ratios, reprojection error, boundary flags, and verbatim explanations. | `test_temporal_engine.py` | **Verified.** Traceable back to SQLite `change_events` table. |
| 15 | **SQLite Persistence & Restart Behavior** | `src/objectmemory/persistence/repository.py` (`SQLiteMemoryStore`), `src/objectmemory/api/workspace_manager.py` | SQLite schema with foreign keys, WAL mode, atomic state transitions, and idempotent session processing. Preserves entity belief across process restarts. | `test_persistence.py`, `test_api_backend.py` | **Verified.** Tested in memory and file-backed modes. Workspaces are completely isolated. |
| 16 | **Benchmark Metric Computation** | `src/objectmemory/benchmarks/ground_truth_matcher.py`, `scripts/evaluate_natural_detection.py`, `scripts/evaluate_controlled_benchmark.py` | Precision, Recall, F1, mAP@50, mAP@50:95 (Supervision), movement recall, return-to-baseline accuracy, removal recall, identity switches, duplicate tracks. | `test_phase10a_corrections.py` | **Partially Verified.** Natural detection metrics are sound; sequence temporal metrics are impacted by the state-lookup bug noted in item 3. |
| 17 | **Frontend Evidence Display** | `frontend/src/views/SceneComparisonView.tsx`, `frontend/src/components/EvidenceLedger.tsx`, `frontend/src/components/EntityCropThumbnail.tsx` | Synchronized dual canvases with shared zoom/pan, interactive wipe slider, structured evidence ledger rendering "Not available" for missing metrics (never 0), and HTML5 canvas dynamic crops with coordinate clamping. | `frontend/e2e/phase13_comparison_evidence.spec.ts` | **Verified.** Passes all 5 Playwright tests; builds cleanly in Vite without errors. |

---

## 4. Evaluation-Code Correctness Audit

### A. Ground-Truth Matching
- **Procedure**: Initial mapping between ground-truth and system entities is computed via Kuhn-Munkres bipartite matching on Visit 1 using cost $c = 1.0 - \text{IoU}$ (for $\text{IoU} \ge 0.05$) or $1.0 + (d / 150.0)$ (for centroid distance $d \le 150\text{ px}$).
- **One-to-One Guarantee**: Enforced by `solve_optimal_assignment` (`scipy.optimize.linear_sum_assignment`).
- **Duplicate & Unmatched Tracking**: Unmatched GT entities are reported as false negatives; unmatched system tracks are counted as duplicate tracks.
- **Critical Flaw Found**: In `ground_truth_matcher.py` (line 252), inside the visit iteration loop:
  ```python
  for v_idx, vf in enumerate(dataset.visits):
      ...
      for ann in vf.annotations:
          gid = ann.entity_id or ann.entity_label
          sys_id = gt_to_sys.get(gid)
          sys_ent = sys_id_to_ent.get(sys_id) if sys_id else None
          if sys_ent is not None:
              observed_state = sys_ent.current_state # BUG: Evaluates final state at end of sequence!
  ```
  `sys_ent.current_state` is the entity's state **after all visits have been processed**, not its historical state at visit $v_i$. If an entity moves in Visit 2 and returns to baseline in Visit 4, `sys_ent.current_state` will be `CONFIRMED_PRESENT`. When scoring Visit 2, the evaluator observes `CONFIRMED_PRESENT` instead of `CONFIRMED_MOVED`, wrongly counting a missed movement.

### B. Detection Metrics
- **Formulas & Thresholds**:
  $$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}, \quad F_1 = \frac{2 \cdot P \cdot R}{P + R}$$
  Evaluated at $\text{IoU} = 0.50$. Supervision calculates standard Pascal VOC / COCO mAP@50 and mAP@50:95.
- **Image Consistency**: In `evaluate_natural_detection.py`, both YOLOv8n and YOLO-World evaluate on the exact same 20 COCO images with the same 84 annotations across 8 classes.
- **Latency Measurement**: Warmup pass is executed on image 0 prior to timing. Latency statistics (mean, median, p95, min, max) exclude warmup. Hardware parameters are recorded via `psutil`.

### C. Temporal Metrics
- **Hardcoding Audit**: Inspected `evaluate_controlled_benchmark.py`. The historical hardcoded `matched = True` bug on S04 v4 was verified as removed in Phase 10A; it now executes a strict check for `EpistemicState.CONFIRMED_REMOVED` and correctly reports a test failure (`matched = False`, event accuracy 11/12 = 91.67%).
- **Denominators**:
  - Movement Recall: $\frac{\text{Correct Relocations}}{\text{Total Relocated Evaluations}}$
  - False Movement Rate: $\frac{\text{False Movements}}{\text{Total Unchanged Evaluations}}$
  - Return-to-Baseline Accuracy: $\frac{\text{Detected Returns}}{\text{Expected Returns}}$
  - Removal Recall: $\frac{\text{Correct Removals}}{\text{Total Removal Evaluations}}$
  - False Removal Rate: $\frac{\text{False Removals}}{\text{Total Visible Evaluations}}$

### D. Data Leakage Audit
- **Verification**: In production code (`TemporalMemoryEngine.process_visit`), the engine receives strictly `VisitInput(session_id, image, registration, detections)`. It receives zero ground-truth labels, expected events, or future observations.
- **Controlled Benchmark Decoupling Note**: In `evaluate_controlled_benchmark.py` (Axis 3 and 4), synthetic ground-truth annotations are passed into `engine.process_visit` as detection fixtures. This was intentionally done to test the temporal state machine in isolation from detector noise (since neural detectors output 0 detections on 2D line drawings). This is valid for unit evaluation of the state machine, but must be clearly documented as decoupled simulation, not end-to-end vision testing.

---

## 5. Temporal-Reasoning Scenario Matrix

| Scenario | Expected Behavior | Implementation Path | Verification Status | Evaluation Context | Smallest Validating Experiment |
|---|---|---|---|---|---|
| **1: Stationary Object** | Remains `CONFIRMED_PRESENT`; zero false movement events. | `_evaluate_displacement`: $d \le T_{\text{base}}$ ($35\text{ px}$) preserves state. | **Verified.** | Synthetic jitter, lighting shifts, and real plant/book controls. | 10 repeated frames with camera vibration noise. |
| **2: Temporary Move & Return** | Move triggers `CONFIRMED_MOVED`; return triggers `CONFIRMED_PRESENT` with `is_return_to_baseline=True`. | Evaluates baseline displacement relative to initial coordinates ($d_{\text{base}} \le T_{\text{effective}}$). | **Verified in Engine; Evaluator Buggy.** | Controlled benchmark S02 (passed 100%); pilot mug (partially scored due to evaluator bug). | 3-visit sequence: baseline $\to$ offset 100px $\to$ restore within 10px. |
| **3: Persistent Move** | Transitions to `CONFIRMED_MOVED` and remains there across subsequent visits without duplicate events. | $d_{\text{consecutive}} \le T_{\text{effective}}$ maintains `CONFIRMED_MOVED` without new event emission. | **Verified.** | Controlled benchmark S05 (passed 100%); unit tests in `test_temporal_engine.py`. | 4-visit sequence: baseline $\to$ offset 80px $\to$ hold $\to$ hold. |
| **4: Temporary Occlusion** | Non-detection does not confirm removal; tagged `PARTIALLY_OCCLUDED`. | Canonical bbox IoU $> 0.20$ with another detected object flags occlusion; halts removal progression. | **Verified.** | Controlled benchmark S03 (passed 100%); unit tests in `test_removal_and_occlusion_evaluation.py`. | 3-visit sequence: object visible $\to$ occluded by book $\to$ book removed. |
| **5: Genuine Removal** | Progresses `PRESENT` $\to$ `PENDING_REMOVAL` $\to$ `CONFIRMED_REMOVED` only upon verified vacancy. | Requires 2 visits with patch cosine similarity $< 0.60$ or 3 unobserved visits under valid registration. | **Verified Failure Mode.** | Controlled benchmark S04 (v4 correctly fails removal because vacancy patch unverified); pilot bottle. | 4-visit sequence: object visible $\to$ removed (empty patch) $\to$ empty patch $\to$ empty patch. |
| **6: Lighting Variation** | SIFT registration adapts; objects remain stationary; no spurious changes. | Feature detector normalization; SIFT gradients invariant to affine illumination changes. | **Verified.** | Controlled benchmark S01 (passed 100%); `test_illumination_sensitivity.py` ($\pm 30\%$). | Physical lamp dimming from 500 lux to 150 lux on stationary bench. |
| **7: Viewpoint Change** | Homography co-registers plane; registration failure halts spatial decisions. | `PlanarRegistrar` RANSAC homography. Gating triggers if $\text{inliers} < 20\%$. | **Verified.** | Controlled benchmark S06 (passed 100%); adaptive keyframe tests in `test_keyframe_registration.py`. | Camera pan $\pm 25^\circ$ and tilt $\pm 15^\circ$ over textured workbench. |
| **8: Boundary Clipping** | Edge centroid truncation does not trigger false movement event. | Bbox within 8 px of edge flagged `is_clipped`; threshold elevated to 50 px. | **Verified.** | Synthetic unit tests in `test_boundary_and_registration.py` and `test_temporal_boundary_reappearance.py`. | Object positioned 5px from right frame boundary with slight camera drift. |
| **9: Same-Class Objects** | Separate tracks maintained without identity collapse. | Kuhn-Munkres optimal assignment + visual signature cosine similarity + ambiguity margin. | **Verified.** | Dual-cup pilot annotations (cup_01 vs cup_02); unit tests in `test_temporal_cross_class_association.py`. | Two identical white ceramic mugs spaced 150px apart on workbench. |
| **10: Contradictory Evidence** | Preserves uncertainty instead of manufacturing certainty. | `_is_registration_reliable` drops unverified frames; ambiguity margin sets `AMBIGUOUS_IDENTITY`. | **Verified.** | Unit tests in `test_temporal_failure_modes.py`; registration inlier degradation warning. | Textureless background causing SIFT inlier ratio to drop to 12%. |

---

## 6. Dataset Provenance and Limitations

```
                                  DATASET COMPARISON MATRIX
┌─────────────────────────┬──────────────┬────────────┬───────────────────────┬───────────────────────────────┐
│ Dataset Name            │ Images/Visits│ Objects/GT │ Provenance Source     │ Appropriate Scientific Use    │
├─────────────────────────┼──────────────┼────────────┼───────────────────────┼───────────────────────────────┤
│ controlled_temporal_v1  │ 30 images    │ 178 GT obj │ Procedural 2D drawings│ State machine logic unit-test │
│                         │ (6 sequences)│ (6 classes)│ (Synthetic / CC0)     │ (NOT for detector evaluation) │
├─────────────────────────┼──────────────┼────────────┼───────────────────────┼───────────────────────────────┤
│ coco_val2017_subset     │ 20 images    │ 84 GT obj  │ Official MS COCO 2017 │ Natural detector sanity check │
│                         │ (single-shot)│ (8 classes)│ val images (CC-BY 4.0)│ (NOT a temporal sequence)     │
├─────────────────────────┼──────────────┼────────────┼───────────────────────┼───────────────────────────────┤
│ real_workbench          │ 7 images     │ 7 entities │ Handheld smartphone   │ Qualitative pilot inspection  │
│ (desk_pilot_01)         │ (1 sequence) │ (4 classes)│ crops (no EXIF)       │ (NOT a metrology benchmark)   │
└─────────────────────────┴──────────────┴────────────┴───────────────────────┴───────────────────────────────┘
```

### Detailed Dataset Analysis

1. **`data/controlled_temporal_v1/`**:
   - **Nature**: 30 procedural PNG illustrations generated across 6 sequences: S01 (lighting), S02 (move/return), S03 (occlusion), S04 (removal), S05 (persistent move), S06 (camera viewpoint).
   - **Annotations**: Complete bounding boxes and explicit expected temporal event definitions (`expected_events.json`).
   - **Limitation**: Stylized non-photographic drawings. Off-the-shelf neural detectors (YOLO) detect 0 objects. Valid only for testing the mathematical correctness of homography and state-machine transitions when fed oracle boxes.
   - **Appropriate Claim**: *"Demonstrates algorithmic correctness of the epistemic state machine and homography registration under controlled geometric and appearance perturbations."*
   - **Inappropriate Claim**: *"Proves real-world robustness to lighting, occlusion, and camera movement."*

2. **`data/coco_val2017_subset/`**:
   - **Nature**: 20 authentic natural photographs from the COCO 2017 validation split containing 84 ground-truth annotations across 8 target classes (`cup`, `bottle`, `book`, `laptop`, `scissors`, `potted plant`, `chair`, `cell phone`).
   - **Annotations**: Official COCO polygons converted to bounding boxes (`instances_val2017_subset.json`).
   - **Limitation**: Static single-frame images without temporal sequence relationships. Small sample size ($N=20$) yields wide confidence intervals.
   - **Appropriate Claim**: *"Establishes baseline CPU inference latency and zero-shot open-vocabulary detection interoperability on standard natural imagery."*
   - **Inappropriate Claim**: *"Demonstrates competitive object detection accuracy relative to state-of-the-art vision models."*

3. **`data/real_workbench/` (`desk_pilot_01`)**:
   - **Nature**: 7 user-supplied PNG images of an office desk.
   - **Metadata Deficiencies**: No camera calibration, EXIF headers, lens focal length, or physical distance measurements.
   - **Geometric Inconsistency**: Visits 1–3 are $780 \times 636\text{ px}$; Visits 4–7 are $\approx 584 \times 576\text{ px}$, causing an uncalibrated field-of-view shift.
   - **Annotation Limitations**: Bounding boxes hand-estimated from 10% grid overlays with $\pm 15\text{ px}$ uncertainty.
   - **Appropriate Claim**: *"Serves as an exploratory physical proof-of-concept demonstrating that multi-visit ingestion, registration, and UI visualization function on real desk photographs."*
   - **Inappropriate Claim**: *"Provides ground-truth benchmark validation of physical-world change detection accuracy."*

---

## 7. Existing Metrics, Exact Sources, and Denominators

| Benchmark / Evaluation | Metric Name | Reported Value | Exact Mathematical Definition | Source Artifact / Script |
|---|---|---|---|---|
| **COCO Subset Detection** | YOLOv8n Precision | 0.6429 | $\frac{18\text{ TP}}{18\text{ TP} + 10\text{ FP}} = \frac{18}{28}$ | `scripts/phase10a_coco_subset_evaluation_results.json` |
| **COCO Subset Detection** | YOLOv8n Recall | 0.2143 | $\frac{18\text{ TP}}{18\text{ TP} + 66\text{ FN}} = \frac{18}{84}$ | `scripts/phase10a_coco_subset_evaluation_results.json` |
| **COCO Subset Detection** | YOLOv8n F1 Score | 0.3214 | $\frac{2 \cdot 0.6429 \cdot 0.2143}{0.6429 + 0.2143}$ | `scripts/phase10a_coco_subset_evaluation_results.json` |
| **COCO Subset Detection** | YOLOv8n mAP@50 | 0.3450 | Supervision Pascal VOC 11-point interpolated AP at IoU=0.50 | `scripts/phase10a_coco_subset_evaluation_results.json` |
| **COCO Subset Detection** | YOLO-World Precision | 0.5385 | $\frac{7\text{ TP}}{7\text{ TP} + 6\text{ FP}} = \frac{7}{13}$ | `scripts/phase10a_coco_subset_evaluation_results.json` |
| **COCO Subset Detection** | YOLO-World Recall | 0.0833 | $\frac{7\text{ TP}}{7\text{ TP} + 77\text{ FN}} = \frac{7}{84}$ | `scripts/phase10a_coco_subset_evaluation_results.json` |
| **COCO Subset Detection** | YOLO-World F1 Score | 0.1443 | $\frac{2 \cdot 0.5385 \cdot 0.0833}{0.5385 + 0.0833}$ | `scripts/phase10a_coco_subset_evaluation_results.json` |
| **COCO Subset Detection** | YOLO-World mAP@50 | 0.1716 | Supervision Pascal VOC 11-point interpolated AP at IoU=0.50 | `scripts/phase10a_coco_subset_evaluation_results.json` |
| **Controlled Temporal Benchmark** | Registration Success Rate | 1.0000 | $\frac{24\text{ Successful Pairs}}{24\text{ Evaluated Pairs}}$ across 6 sequences | `scripts/phase10a_controlled_evaluation_results.json` |
| **Controlled Temporal Benchmark** | Cross-Visit Association Switches | 0 switches | Total identity switches across 30 visits | `scripts/phase10a_controlled_evaluation_results.json` |
| **Controlled Temporal Benchmark** | Temporal Event Accuracy | 0.9167 | $\frac{11\text{ Passed Events}}{12\text{ Expected Events}}$ (1 failed removal on S04 v4) | `scripts/phase10a_controlled_evaluation_results.json` |
| **Real Workbench Pilot** | Overall State Accuracy | 0.7619 | $\frac{32\text{ Matched State Observations}}{42\text{ Total Evaluations}}$ | `scripts/phase9b_evaluation_results.json` |
| **Real Workbench Pilot** | False Movement Rate | 0.0000 | $\frac{0\text{ False Movements}}{28\text{ Unchanged Evaluations}}$ | `scripts/phase9b_evaluation_results.json` |
| **Real Workbench Pilot** | Movement Recall | 0.3333 | $\frac{1\text{ Correct Relocation}}{3\text{ Relocation Evaluations}}$ (2 missed) | `scripts/phase9b_evaluation_results.json` |
| **Real Workbench Pilot** | Return-to-Baseline Accuracy | 0.2000 | $\frac{1\text{ Detected Return}}{5\text{ Expected Returns}}$ | `scripts/phase9b_evaluation_results.json` |
| **Real Workbench Pilot** | Removal Recall | 0.0000 | $\frac{0\text{ Confirmed Removals}}{8\text{ Removal Evaluations}}$ (4 missed) | `scripts/phase9b_evaluation_results.json` |
| **Real Workbench Pilot** | Tracking Integrity | 5 / 7 GT | 5 entities matched, 2 unmatched (`cup_02`, `notebook_01`), 3 duplicates | `scripts/phase9b_evaluation_results.json` |

---

## 8. Identified Bugs, Methodological Risks, and Missing Tests

### Bug 1: Sequence Evaluation Checks Final State Instead of Historical State
- **File**: `src/objectmemory/benchmarks/ground_truth_matcher.py`, line 252.
- **Code**:
  ```python
  observed_state = sys_ent.current_state
  ```
- **Consequence**: In `evaluate_sequence`, the loop iterates through visits $V_1, V_2, \dots, V_7$, but evaluates `sys_ent.current_state` (the final state after all visits have executed). If an entity moves in Visit 2 and later returns to baseline in Visit 4, Visit 2 is evaluated against its final state (`CONFIRMED_PRESENT`), scoring Visit 2 as a missed movement.
- **Severity**: **HIGH (Methodological Risk).** Under-reports movement recall on multi-visit sequences.

### Bug 2: Anisotropic Image Stretching in Object Detectors
- **Files**: `src/objectmemory/detection/detector.py` (lines 350-357) & `src/objectmemory/detection/yolo_world.py` (lines 276-279).
- **Code**:
  ```python
  cv2.resize(img_rgb, (640, 640)) # Direct squashing
  ```
- **Consequence**: Non-square images (e.g. 16:9 1920x1080) are distorted by 1.77x horizontally. Bounding box coordinates are decoded with anisotropic scale factors ($w / 640$, $h / 640$). Bounding boxes on elongated tools or circular cups suffer aspect-ratio distortion, reducing detector confidence and localization accuracy.
- **Severity**: **MEDIUM (Computer Vision Best Practice).** Standard YOLO models require letterbox padding with aspect ratio preservation.

### Methodological Risk 1: Evaluation on Procedural Drawings Bypasses End-to-End Detector
- **File**: `scripts/evaluate_controlled_benchmark.py` (lines 456-466).
- **Observation**: In Axis 4, the evaluator feeds ground-truth boxes directly into `engine.process_visit` with `confidence=1.0` and `canonical_bbox=ann.bbox`.
- **Consequence**: The reported 91.67% event accuracy measures only the state machine in a simulation environment with zero detector false positives, false negatives, or localization jitter. It does not measure the real-world performance of the vision system.
- **Severity**: **HIGH (Scientific Validity).** Must not be represented as an end-to-end benchmark.

### Methodological Risk 2: Metrology Deficiencies in Real Workbench Pilot
- **File**: `data/real_workbench/manifest.json`.
- **Observation**: Resolution change between V1-V3 ($780 \times 636$) and V4-V7 ($584 \times 576$), hand-drawn bounding boxes with $\pm 15\text{ px}$ uncertainty, and lack of ground-truth physical metric calibration (mm).
- **Severity**: **HIGH (Benchmark Integrity).** Must remain strictly categorized as a qualitative pilot, not a benchmark.

### Missing Tests
1. **Letterbox Preprocessing Regression Test**: No test currently verifies that letterboxed inputs produce superior or identical IoU compared to direct stretching.
2. **Visit-by-Visit Historical State Audit Test**: No unit test verifies that `GroundTruthMatcher` retrieves historical states from SQLite `change_events` as of a specific visit.
3. **Physical Scale (Pixel-to-Millimeter) Calibration Test**: No component exists to calibrate pixel displacements to real-world metric units using ArUco markers or checkerboard calibration targets.

---

## 9. Research-Claim Support Matrix

```
                                RESEARCH CLAIM SUPPORT MATRIX
┌───────────────────────────────────────────────┬─────────────────────────────┬──────────────────────────────────────────┐
│ Research Claim                                │ Classification              │ Scientific Justification                 │
├───────────────────────────────────────────────┼─────────────────────────────┼──────────────────────────────────────────┤
│ 1. Long-term visual memory across visits      │ Implemented & Demonstrated  │ SQLite-backed causal entity tracks       │
│                                               │                             │ persist across sessions and restarts.    │
├───────────────────────────────────────────────┼─────────────────────────────┼──────────────────────────────────────────┤
│ 2. Evidence-backed persistent change detection│ Implemented & Demonstrated  │ Change events emit displacement, inlier  │
│                                               │                             │ ratios, vacancy sim, and explanations.   │
├───────────────────────────────────────────────┼─────────────────────────────┼──────────────────────────────────────────┤
│ 3. Distinguishing absence from removal        │ Implemented & Partially     │ Gated 2-visit vacancy rule implemented;  │
│                                               │ Evaluated                   │ failed on synthetic S04; recall 0% pilot │
├───────────────────────────────────────────────┼─────────────────────────────┼──────────────────────────────────────────┤
│ 4. Robustness to lighting, occlusion, viewpoint│ Implemented but             │ Verified on synthetic line illustrations;│
│                                               │ Insufficiently Evaluated    │ unverified on diverse physical scenes.   │
├───────────────────────────────────────────────┼─────────────────────────────┼──────────────────────────────────────────┤
│ 5. Explicit uncertainty-aware reasoning       │ Implemented & Demonstrated  │ Adaptive threshold T_base + Delta_U and   │
│                                               │                             │ inlier gating (<20%) actively suppress.  │
├───────────────────────────────────────────────┼─────────────────────────────┼──────────────────────────────────────────┤
│ 6. Consistent identity for same-class objects │ Implemented & Demonstrated  │ Kuhn-Munkres + visual signatures prevent │
│                                               │                             │ identity collapse (0 switches on synth). │
├───────────────────────────────────────────────┼─────────────────────────────┼──────────────────────────────────────────┤
│ 7. Improved performance over relevant baseline│ Unsupported by Current      │ No formal baseline (ByteTRACK, DeepSORT, │
│                                               │ Evidence                    │ SceneDiff) evaluated on identical data.  │
└───────────────────────────────────────────────┴─────────────────────────────┴──────────────────────────────────────────┘
```

---

## 10. Proposed Validation Protocol

To establish whether ObjectMemory's claims are scientifically defensible in the physical world, the following 6 rigorous experiments are designed:

### EXPERIMENT A — Detection Quality Under Aspect-Preserving Letterboxing
- **Research Question**: Does aspect-ratio-preserving letterboxing significantly improve bounding box IoU and mAP for YOLOv8n and YOLO-World over anisotropic stretching on real workbench imagery?
- **Input Data**: 50 photographic workbench images captured at native 16:9 aspect ratio ($1920 \times 1080$).
- **Ground Truth**: Manually annotated bounding boxes with independent dual-annotator verification (target classes: cup, bottle, book, scissors, pliers, multimeter).
- **Baselines**: Current anisotropic direct-stretch preprocessing.
- **Independent Variables**: Preprocessing method (direct stretch vs letterbox with gray padding).
- **Metrics**: Precision, Recall, F1, mAP@50, mAP@50:95, latency (ms).
- **Sample Size**: $N = 50$ images, minimum 200 object instances. (Proposed starting point for initial effect size estimation, not a formal power calculation).
- **Acceptance Criteria**: Pre-registered improvement of $\ge +0.05$ mAP@50 without latency increase $> 10\%$.

### EXPERIMENT B — Metric Registration Error Calibration
- **Research Question**: What is the true physical coordinate error (in millimeters) of planar homography co-registration across camera tilt angles up to $30^\circ$?
- **Input Data**: Multi-view images of a flat workbench equipped with a high-precision ChArUco calibration board ($8 \times 11$ squares, 30mm square size).
- **Ground Truth**: Metrology ground truth computed from OpenCV ChArUco corner detection ($< 0.2\text{ mm}$ precision).
- **Baselines**: Direct frame differencing; un-gated SIFT homography.
- **Independent Variables**: Camera pitch ($0^\circ, 10^\circ, 20^\circ, 30^\circ$), yaw ($\pm 20^\circ$), lighting (100 lux to 600 lux).
- **Metrics**: Mean Absolute Error (MAE in mm), 95th-percentile error (mm), SIFT inlier ratio, spatial coverage fraction.
- **Sample Size**: 40 image pairs across 4 camera poses.
- **Acceptance Criteria**: Planar error $\le 5.0\text{ mm}$ across $\le 20^\circ$ tilt; registration safely reports failure above $25^\circ$.

### EXPERIMENT C — Temporal State Machine on Real Physical Multi-Visit Sequences
- **Research Question**: Does the epistemic state machine reliably detect relocations and confirm removals on real physical multi-visit desk sequences with corrected historical state scoring?
- **Input Data**: 3 distinct physical workbench sequences of 8 visits each (24 physical captures total) captured under calibrated lighting.
- **Ground Truth**: Physical logs recording exact millimeter relocations and visit removal events.
- **Baselines**: Frame-to-frame Hungarian IoU tracker without epistemic memory; static baseline differencing.
- **Independent Variables**: Sequence type (relocation with return; persistent relocation; genuine removal; temporary occlusion).
- **Metrics**: Overall State Accuracy, Movement Recall, False Movement Rate, Return-to-Baseline Accuracy, Removal Recall.
- **Sample Size**: 24 physical visits, 144 entity evaluations across 3 independent sequences.
- **Acceptance Criteria**: Movement Recall $\ge 85\%$, False Movement Rate $\le 5\%$, Removal Recall $\ge 80\%$.

### EXPERIMENT D — Persistent Identity Tracking with Identical Distractors
- **Research Question**: Can the combination of visual signatures and spatial assignment maintain persistent object identity when multiple visually identical objects (e.g. three identical white ceramic mugs) undergo relocations?
- **Input Data**: 6-visit sequence with 3 identical mugs and 2 identical books undergoing sequential shifts.
- **Ground Truth**: Unique discrete physical tracking markers on the underside of each object (hidden from camera, recorded in log).
- **Baselines**: Label-only associator; spatial-only IoU associator without ambiguity gating.
- **Metrics**: Identity Switch Count, Duplicate Track Count, MOTA (Multi-Object Tracking Accuracy adapted for discrete visits).
- **Acceptance Criteria**: Zero identity switches among spatially resolved objects; explicit declaration of `AMBIGUOUS_IDENTITY` when objects cross paths within ambiguity margin.

### EXPERIMENT E — Isolated Nuisance Robustness Testing
- **Research Question**: Does ObjectMemory maintain state invariance when subjected to controlled single-factor visual nuisances?
- **Input Data**:
  - E1 (Lighting): 5 visits with illumination stepped from 600 lux $\to$ 400 lux $\to$ 200 lux $\to$ 100 lux $\to$ 600 lux (stationary objects).
  - E2 (Partial Occlusion): 4 visits where a foreground clipboard partially covers 20%, 40%, 60% of an entity.
  - E3 (Boundary Framing): 4 visits where camera framing shifts so that an object moves from interior to clipping the image border (5px from edge).
- **Metrics**: False Movement Rate, False Removal Rate, State Invariance Ratio.
- **Acceptance Criteria**: 0% false movements under lighting shifts E1; 0% false removals under partial occlusion E2; 0% false movements under boundary clipping E3.

### EXPERIMENT F — Operational Latency & Recovery Stress Testing
- **Research Question**: Does the system maintain sub-second per-visit response times and perfect recovery after unexpected power/process termination?
- **Input Data**: Automated sequential ingestion of 50 consecutive visits. Random process kill injection during visit 25.
- **Metrics**: Processing latency breakdown (detection, registration, association, state update, database commit), database corruption rate (WAL check), recovery replay accuracy.
- **Acceptance Criteria**: SQLite database integrity passes `PRAGMA integrity_check`; 100% idempotent replay of partially processed visits; total visit latency $< 500\text{ ms}$ on CPU for YOLOv8n.

---

## 11. Recommended Priorities Ranked by Severity and Impact

1. **Priority 1 (Severity: HIGH — Evaluation Bug)**: Fix `ground_truth_matcher.py` historical state query. Modify sequence evaluation so that each visit compares the ground-truth annotation against the entity's recorded state *at that specific visit*, rather than the final state at the end of the sequence.
2. **Priority 2 (Severity: MEDIUM — Computer Vision Best Practice)**: Implement aspect-ratio-preserving letterboxing in `OpenCVDNNDetector` and `YOLOWorldDetector` to eliminate coordinate squashing and improve detection recall on wide-aspect images.
3. **Priority 3 (Severity: HIGH — Scientific Benchmarking)**: Collect a 24-visit calibrated physical workbench dataset with physical metric logs (mm) to replace the uncalibrated 7-image pilot.
4. **Priority 4 (Severity: MEDIUM — End-to-End Evaluation)**: Add an end-to-end evaluation script that runs both detector and temporal engine together on physical images, reporting joint system performance alongside decoupled metrics.
5. **Priority 5 (Severity: LOW — Documentation & UI Refinement)**: Expose metric millimeter conversion options in the UI when calibration markers (e.g. ArUco) are detected on the workbench plane.

---

## 12. Proposed Phase 14B Implementation Plan

Following approval of this audit, Phase 14B will implement the necessary fixes and empirical validation:

```
[Phase 14B Step 1] ──► Correct GroundTruthMatcher visit-state lookup bug
                            │
[Phase 14B Step 2] ──► Add letterbox preprocessing to YOLO detectors
                            │
[Phase 14B Step 3] ──► Run safe regression tests & recompute pilot metrics
                            │
[Phase 14B Step 4] ──► Package validation scripts for reproducible execution
```

- **Task 1: Fix Evaluator Historical State Query**: In `src/objectmemory/benchmarks/ground_truth_matcher.py`, query `store.get_session_entity_states(vf.visit_id)` or inspect `v_rep.temporal_report.entity_updates` to evaluate the true epistemic state at visit $v_i$. Add unit tests proving that temporary relocation followed by return scores both events correctly.
- **Task 2: Implement Aspect-Preserving Letterboxing**: Update `src/objectmemory/detection/detector.py` and `yolo_world.py` with standard letterboxing (resizing long edge to 640 and padding short edge with 114 gray). Update coordinate un-scaling to subtract padding before multiplying by scale factor.
- **Task 3: Recompute Pilot & COCO Evaluation Results**: Re-run the safe evaluation scripts to update `scripts/phase9b_evaluation_results.json` and verify the corrected state accuracy without historical distortion.
- **Task 4: Add Automated Validation Script**: Provide `scripts/run_scientific_validation.py` executing all read-only diagnostic benchmarks and generating a unified validation report.

---

## 13. Exact Commands to Reproduce Safe Audit Checks

The following read-only commands were executed and verified during this audit:

```powershell
# 1. Execute all 288 Python unit and integration tests (Read-Only)
cd C:\Users\Rishh\.gemini\antigravity\scratch\objectmemory
$env:PYTHONPATH="src;."
python -m unittest discover tests

# 2. Run Vite frontend production build check (Read-Only to code)
cd C:\Users\Rishh\.gemini\antigravity\scratch\objectmemory\frontend
npm run build

# 3. Run Playwright end-to-end and component tests (Requires local backend running)
cd C:\Users\Rishh\.gemini\antigravity\scratch\objectmemory\frontend
npx playwright test

# 4. Safe inspection of COCO subset benchmark results (Read-Only)
cd C:\Users\Rishh\.gemini\antigravity\scratch\objectmemory
Get-Content scripts/phase10a_coco_subset_evaluation_results.json | Select-Object -First 50

# 5. Safe inspection of controlled temporal benchmark results (Read-Only)
cd C:\Users\Rishh\.gemini\antigravity\scratch\objectmemory
Get-Content scripts/phase10a_controlled_evaluation_results.json | Select-Object -First 50
```

---

## 14. Distinction Between Inspected Evidence and Inference

- **Inspected Evidence (Directly Verified in Source Code & Execution)**:
  - 288 Python tests pass in 51.09s.
  - 5 Playwright tests pass in 21.5s.
  - S04 v4 failure in `phase10a_controlled_evaluation_results.json` is genuine (caused by unverified vacancy patch, properly scored as failure).
  - YOLO models detect 0 objects on procedural line drawings in `controlled_temporal_v1`.
  - In `ground_truth_matcher.py` line 252, `observed_state = sys_ent.current_state` retrieves final state rather than historical state.
  - Detectors directly resize images to $640 \times 640$ without letterbox padding.
  - Real workbench pilot has resolution shifts ($780 \times 636$ vs $584 \times 576$) and hand-drawn labels ($\pm 15\text{ px}$).
- **Inference (Scientific Deduction)**:
  - The low recall of YOLOv8n (21.4%) and YOLO-World (8.3%) on COCO val subset at conf=0.25 is exacerbated by anisotropic stretching of rectangular photographs.
  - The pilot sequence's 33.3% movement recall is partially an artifact of the evaluator inspecting final state rather than visit-2 state, causing genuine temporary movements to be scored as missed.
  - Real-world deployment will require metric pixel-to-millimeter calibration to achieve repeatable scientific findings across different cameras.
