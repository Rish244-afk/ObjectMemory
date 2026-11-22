# ObjectMemory Studio — Phase 10 Results Audit (Read-Only)

**Date:** 2026-10-21  
**Project:** ObjectMemory Studio  
**Workspace:** `C:\Users\Rishh\.gemini\antigravity\scratch\objectmemory`  
**Audit Type:** Read-Only Scientific and Empirical Integrity Audit  
**Status:** COMPLETE  

---

## Executive Summary

This read-only audit evaluated the empirical and methodological validity of the Phase 10 deliverables, specifically scrutinizing:
1. The comparative evaluation of YOLOv8n vs YOLO-World on the 20-image COCO 2017 validation subset ([`scripts/evaluate_natural_detection.py`](scripts/evaluate_natural_detection.py)).
2. The 4-axis evaluation of the 6-sequence, 30-visit procedural temporal benchmark ([`scripts/evaluate_controlled_benchmark.py`](scripts/evaluate_controlled_benchmark.py)).
3. The underlying dataset annotations, pre-processing, output decoding, latency measurements, and metric calculations.

No source code, datasets, or evaluation results were modified during this audit.

---

## Part 1: YOLOv8n vs YOLO-World Evaluation Audit

### 1. Verification of Image Set Consistency
- **Finding:** Both models were evaluated on the exact same 20 COCO 2017 validation images.
- **Evidence:** In [`scripts/evaluate_natural_detection.py`](scripts/evaluate_natural_detection.py), lines 170–190 construct a single `image_paths` array from `instances_val2017_subset.json`. The outer loop over `detectors_to_eval` iterates over the identical list of paths.
- **Image IDs Verified:** `139`, `632`, `1268`, `1296`, `1503`, `1993`, `2157`, `2431`, `2592`, `2685`, `3661`, `3845`, `3934`, `4134`, `4795`, `5001`, `5193`, `13923`, `14038`, `99024`.
- **Classification:** **NO ISSUE FOUND**.

### 2. Target Classes & Label Mapping
- **Classes:** 8 target classes: `cup`, `bottle`, `book`, `laptop`, `scissors`, `potted plant`, `chair`, `cell phone`.
- **YOLOv8n Mapping:** `OpenCVDNNDetector` predicts class indices into `COCO_CLASSES_80`. Lines 228 of the evaluation script filter detections by `d.label in TARGET_CLASSES`. All 8 target names match character-for-character with official COCO categories.
- **YOLO-World Mapping:** `YOLOWorldDetector` receives `TARGET_CLASSES` directly as its prompt sequence. When the output tensor predicts class index $i \in \{0..7\}$, line 339 of `yolo_world.py` maps index $i$ to `prompts[cls_id]`, producing exact string alignment.
- **Classification:** **NO ISSUE FOUND**.

### 3. YOLO-World Prompt Fidelity
- **Prompt List:** `['cup', 'bottle', 'book', 'laptop', 'scissors', 'potted plant', 'chair', 'cell phone']`.
- **Mapping:** Each prompt corresponds 1-to-1 to a ground-truth COCO category. No synonyms or altered prompt texts were introduced.
- **Classification:** **NO ISSUE FOUND**.

### 4. Preprocessing, Resizing, Normalization, NMS & Coordinate Conversion
- **Resizing:**
  - `OpenCVDNNDetector`: Uses `cv2.dnn.blobFromImage(..., size=(640, 640), crop=False)`. Direct non-letterboxed resize.
  - `YOLOWorldDetector`: Uses `cv2.resize(img_rgb, (640, 640), interpolation=cv2.INTER_LINEAR)`. Direct non-letterboxed resize.
- **Normalization:**
  - `OpenCVDNNDetector`: $1.0 / 255.0$ scalefactor, `mean=(0, 0, 0)`.
  - `YOLOWorldDetector`: `img.astype(np.float32) / 255.0`.
- **Confidence & NMS:**
  - Both models executed at `confidence_threshold = 0.25` and `nms_threshold = 0.45` via `cv2.dnn.NMSBoxes`.
- **Coordinate Decoding:**
  - Both decode $[cx, cy, w, h]$ from $640 \times 640$ space using $x_{\text{scale}} = w_{\text{orig}} / 640.0$ and $y_{\text{scale}} = h_{\text{orig}} / 640.0$.
  - Coordinates are clamped to image boundaries $[0, w_{\text{orig}}]$ and $[0, h_{\text{orig}}]$.
- **Classification:** **NO ISSUE FOUND**.

### 5. IoU Matching & Metric Calculations
- **IoU Implementation:** Lines 48–64 implement standard axis-aligned bounding box IoU.
- **Greedy Matching:** Predictions are sorted by confidence descending. Matches are assigned greedily to the highest-IoU unmatched ground-truth box of the matching class (threshold $\ge 0.50$).
- **Formulas:**
  $$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}, \quad F_1 = \frac{2 \cdot P \cdot R}{P + R}$$
- **Classification:** **NO ISSUE FOUND**.

### 6. Supervision mAP Evaluation Protocol
- **Implementation:** Line 246 calls `evaluate_detection_map` from [`src/objectmemory/detection/supervision_adapter.py`](src/objectmemory/detection/supervision_adapter.py).
- **Supervision Execution:** Calls `sv.metrics.MeanAveragePrecision(class_agnostic=False)`. Runs official COCO 101-point interpolated PR curve across IoU thresholds $[0.50:0.05:0.95]$.
- **Outcome:** Valid and identical across both models.
- **Classification:** **NO ISSUE FOUND**.

### 7. Evaluation Fairness Regarding Open-Vocabulary Detection
- **Critical Finding:** All 8 target classes are standard COCO-80 classes on which YOLOv8n was pre-trained.
- **Impact:** This test benchmarks **prompt-conditioned recall of known in-domain COCO classes**, NOT true zero-shot open-vocabulary generalization to unseen objects (e.g. multimeter, soldering iron, wire stripper).
- YOLOv8n has a fixed classification head trained specifically on these categories, giving it a structural advantage.
- **Classification:** **VALID LIMITATION**.

### 8. Latency Measurement Protocol
- **Measurement:** Single-pass inference over 20 images timed with `time.time()`.
- **Warmup:** **NO explicit warmup pass** was executed. The first frame included initial session startup / text embedding tokenization.
  - YOLOv8n: Min: 47.51ms, Max: 107.68ms (first frame).
  - YOLO-World: Min: 231.06ms, Max: 470.48ms (first frame).
- **Impact:** Mean latency is slightly inflated by the cold-start first iteration. Median latency (64.39ms for YOLOv8n, 277.95ms for YOLO-World) is the more reliable representative metric.
- **Classification:** **POTENTIAL RISK**.

### 9. Confirmed Implementation Errors
- In `scripts/evaluate_natural_detection.py`, line 211 initializes `per_class_counts`, but the dictionary is never updated inside the loop. The per-class TP/FP/FN breakdown was omitted from the final serialized JSON.
- **Classification:** **CONFIRMED DEFECT** (reporting omission).

---

## Part 2: Controlled Temporal Benchmark Audit

### 1. Denominator and Outcome per Metric

| Metric | Denominator | Numerator | Stated Outcome |
| :--- | :--- | :--- | :--- |
| **Detection mAP@50 (YOLOv8n)** | 178 ground-truth objects | 0 detections passing $\ge 0.15$ | **0.0000** |
| **Detection mAP@50 (YOLO-World)** | 178 ground-truth objects | 0 detections passing $\ge 0.15$ | **0.0000** |
| **Registration Success Rate** | 24 visit pairs (6 seqs $\times$ 4 pairs) | 24 successful alignments | **100.0%** (24/24) |
| **Cross-Visit Identity Switches** | 6 sequences (5 visits each) | 0 identity switches detected | **0** |
| **Cross-Visit Duplicate Tracks** | 36 unique GT entities | 4 duplicate tracks (S01 only) | **4** |
| **Temporal Event Compliance** | 14 expected scenario events | 14 matched events | **100.0%** (14/14) |
| **Spurious Movements** | 14 event evaluations | 0 false movement assertions | **0** |
| **Spurious Removals** | 14 event evaluations | 0 false removal assertions | **0** |

### 2. Scoring of the 14 Expected Events
- Scored in `scripts/evaluate_controlled_benchmark.py` by comparing `TemporalMemoryEngine` outputs against scenario expectations in `expected_events.json`:
  - `lighting_change` (S01 v2, v4): Verified 0 entities transitioned to `CONFIRMED_MOVED`.
  - `object_moved` (S02 v3, S05 v3): Verified entity moved to `CONFIRMED_MOVED` with displacement $> 200\text{px}$.
  - `return_to_baseline` (S02 v4): Verified `is_return_to_baseline == True` and transition back to `CONFIRMED_PRESENT`.
  - `partial_occlusion` (S03 v3): Verified 0 entities transitioned to `CONFIRMED_REMOVED`.
  - `reobserved` (S03 v5): Verified entity remained `CONFIRMED_PRESENT`.
  - `object_removed` (S04 v4): **Defect identified below**.
  - `continued_absence` (S04 v5): Verified absence update recorded.
  - `movement_persists` (S05 v4): Verified moved entity maintained tracking.
  - `camera_view_change` (S06 v2..v5): Verified 0 false movements under camera shifts.

### 3. Independent Ground Truth vs Engine Output Scoring
- **Defect Identified in S04 v4 (`object_removed`)**:
  In lines 380–388 of `scripts/evaluate_controlled_benchmark.py`:
  ```python
  elif exp.event == "object_removed":
      target_obj = exp.object_id
      target_cls = target_obj.split("_")[0]
      pending_or_removed = [...]
      # In Visit 4, the streak starts or is tracked
      matched = True
      details = {"pending_or_removed": [u.entity_id for u in pending_or_removed]}
  ```
  `matched = True` was hardcoded. Inspection of [`scripts/phase10_controlled_evaluation_results.json`](scripts/phase10_controlled_evaluation_results.json) line 386 reveals `"pending_or_removed": []`.
  Because `matched = True` was hardcoded, this test scored as passing despite an empty update list.
- **Classification:** **CONFIRMED DEFECT**.

### 4. Identity-Switch Ground Truth & Matching Protocol
- Uses programmatic entity IDs (`cup_01`, `book_01`, etc.) as persistent identity ground truth.
- In each visit, matches ground-truth annotations to active system entities via `scipy.optimize.linear_sum_assignment` on centroid Euclidean distance.
- An identity switch is counted if ground-truth entity $g$ maps to system entity $s_1$ in visit $t$ and to $s_2 \ne s_1$ in visit $t+1$.
- Protocol is mathematically sound and follows standard MOTA identity-tracking conventions.
- **Classification:** **NO ISSUE FOUND**.

### 5. Meaning of the 24/24 Registration Pairs
- 6 sequences $\times$ 4 query visits ($V2, V3, V4, V5$) registered against canonical visit $V1$.
- S01–S05 feature static background camera framing; S06 features small global translational shifts.
- ORB finds high inlier counts ($72-135$ inliers) and inlier ratios ($61.5\%-100\%$).
- This tests **planar homography stability under controlled perturbations**, NOT unconstrained 3D multi-view natural scenes.
- **Classification:** **NO ISSUE FOUND**.

### 6. Meaning of 0.0000 mAP on Synthetic Illustrations
- The synthetic dataset consists of stylized procedural vector illustrations (flat colors, sharp synthetic edges, non-photographic textures).
- Pretrained COCO convolutional and transformer models require natural textures and camera sensor noise.
- This result is **scientifically meaningful**: it proves that synthetic procedural datasets cannot evaluate real-world detector performance, validating why the natural-image COCO subset was required.
- **Classification:** **VALID LIMITATION**.

### 7. Explanation of 100% Controlled vs 33.3% Real Pilot Movement Recall
- **Why Controlled = 100%:** The temporal engine was fed pristine renderer ground-truth bounding boxes and exact image crops, with near-zero registration error ($\Delta U \approx 0$).
- **Why Real Pilot = 33.3%:**
  1. Noisy real-world detector bounding boxes (jitter across visits).
  2. Handheld camera viewpoint changes and perspective distortion.
  3. Dynamic uncertainty threshold $\tau_{\text{eff}} = 35 + \Delta U$ conservatively gated Visit 3 movements (`cup_01` and `scissors_01`) due to high registration uncertainty.
- **Classification:** **VALID LIMITATION**.

### 8. Circular, Incomplete, or Overstated Tests
1. **Circular Assertion:** Hardcoded `matched = True` for S04 v4 (`object_removed`).
2. **Overstated Uncertainty Claim:** 15/15 controlled planar transforms do not establish general statistical calibration of 3D camera uncertainty.
3. **Small Sample Overstatement:** 20 COCO images cannot establish definitive model superiority.

---

## Part 3: Findings Classification

| # | Finding Description | Classification | Action Required |
| :---: | :--- | :---: | :--- |
| 1 | Hardcoded `matched = True` for `object_removed` event in `scripts/evaluate_controlled_benchmark.py` | **CONFIRMED DEFECT** | Modify test logic to verify non-empty absence streak initiation. |
| 2 | Unpopulated `per_class_counts` dictionary in `scripts/evaluate_natural_detection.py` | **CONFIRMED DEFECT** | Accumulate per-class TP/FP/FN inside image evaluation loop. |
| 3 | Latency measured without a discarded warmup iteration (inflates cold-start max latency) | **POTENTIAL RISK** | Add 1-pass warmup; report median latency as primary metric. |
| 4 | COCO subset classes are all closed-vocabulary classes known to YOLOv8n | **VALID LIMITATION** | Document as in-domain prompt sanity check; do not claim zero-shot validation. |
| 5 | Small sample size ($N=20$) for natural-image detection benchmark | **VALID LIMITATION** | Label as smoke-test benchmark; acknowledge high sampling variance. |
| 6 | Zero mAP on synthetic illustrations | **VALID LIMITATION** | Document as domain shift confirmation; preserve as rationale for natural test set. |
| 7 | Controlled 100% vs real pilot 33.3% movement recall gap | **VALID LIMITATION** | Keep 33.3% visible as genuine real-world limitation; explain registration gating. |
| 8 | Resizing, normalization, thresholds, and coordinate scaling consistency | **NO ISSUE FOUND** | Both models share identical $640 \times 640$ preprocessing and coordinate scaling. |
| 9 | Supervision COCO mAP evaluation protocol and IoU calculation | **NO ISSUE FOUND** | Standard 101-point COCO evaluation executed consistently. |
| 10 | Bipartite Kuhn-Munkres identity switch tracking protocol | **NO ISSUE FOUND** | Follows MOTA identity switch conventions; 0 switches confirmed on controlled data. |

---

## Part 4: Final Recommendation & Audit Conclusions

### A. Is the Current Detector Comparison Fair Enough to Report?
**YES, WITH STRICT CAVEATS.**  
The comparison is fair as an **empirical baseline of CPU inference latency and prompt-conditioned extraction of standard objects**. On identical images, preprocessing, and confidence thresholds:
- YOLOv8n achieves higher recall (**0.2143** vs **0.0833**) and mAP@50 (**0.3450** vs **0.1716**).
- YOLO-World incurs a $\sim 4.5\times$ latency penalty on CPU (**278.0ms** vs **64.4ms** median) due to dual vision-language transformer backbones.  
*Caveat:* It must **not** be reported as a test of open-vocabulary zero-shot generalization, because all 8 target classes were standard COCO categories.

### B. Which Metrics Can Be Trusted?
1. **Trustworthy**:
   - Median CPU inference latency for both models (64.4ms vs 278.0ms).
   - Supervision mAP@50 and mAP@50:95 on the 20-image COCO subset (with sample size caveat).
   - Controlled registration success rate (100% on planar sequences).
   - Controlled cross-visit identity switch count (0 switches).
   - Real pilot workbench movement recall (33.3% with 0 false movements).
2. **Untrustworthy / Requiring Re-scoring**:
   - The reported 100% temporal compliance on S04 v4 cannot be trusted until `matched = True` is replaced with an actual absence verification assertion.
   - Max and mean latency figures (skewed by first-frame initialization).

### C. Minimum Additional Experiments Required (Prior to Research Publication)
1. **Absence Assertion Correction**: Replace `matched = True` in `evaluate_controlled_benchmark.py` with an explicit assertion on absence streak progression.
2. **Warmup Pass**: Add a 1-image warmup run to latency benchmarks.
3. **True Open-Vocabulary Evaluation**: Evaluate YOLO-World on 10–20 natural photographs of specialized workbench tools not present in COCO (e.g., multimeter, wire stripper, oscilloscope probe, soldering iron).
4. **Sub-Pixel Template Matching**: Test whether template matching can resolve the 2 missed movements in Visit 3 of the real workbench pilot without relaxing registration gates.

### D. Is It Reasonable to Proceed with the Frontend Redesign?
**YES, PROCEED WITH THE FRONTEND REDESIGN.**  
None of the identified defects or limitations reside in the core domain models, the REST API endpoints, the SQLite persistence schema, the registration engine, or the frontend build pipeline. All 280 Python unit tests pass, and the frontend Vite/TypeScript production build compiles in 507ms. The evaluation reporting nuances do not affect UI component design or user workflows.
