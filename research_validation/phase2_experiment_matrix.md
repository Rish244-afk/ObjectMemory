# ObjectMemory: Phase 2 Experiment & Baseline Matrix
**Project:** ObjectMemory: Long-Term Visual Memory for Detecting Persistent Environmental Changes  
**Workspace:** `C:\Users\Rishh\.gemini\antigravity\scratch\objectmemory`  
**Status:** Frozen Experimental Plan

---

## 1. Frozen Baseline Configurations

To prove the central hypothesis—that evidence-gated, uncertainty-aware reasoning reduces unsupported decisions—we define four frozen baselines evaluated under identical detection inputs, camera observations, and ground-truth annotations.

### Baseline Summary Specification

| Baseline ID | Name | Architectural Description | Purpose in Benchmark |
|---|---|---|---|
| **B0** | Memoryless Detector | Raw YOLOv8n detections on each frame independently with zero cross-frame memory. | Establishes the lower-bound performance of per-frame detection without temporal reasoning. |
| **B1** | Standard Hungarian Association | YOLOv8n + standard Hungarian assignment (IoU + appearance), but memoryless change logic (instant disappearance = removal). | Isolates the contribution of basic tracking/association without epistemic multi-visit state reasoning. |
| **B2** | Ungated Temporal Reasoning | Association + multi-visit state transitions, but with all reliability and vacancy safeguards disabled. | Directly isolates the risk of ungated temporal reasoning (shows failure modes of naive state progression). |
| **B3** | Complete ObjectMemory | Proposed full system: registration-conditioned, vacancy-verified, ambiguity-gated epistemic memory. | Demonstrates the full proposed research contribution. |

---

### Detailed Baseline Engineering Specifications

#### B0 — Memoryless YOLO Detection
- **Components Enabled:**
  - `ONNXObjectDetector` (`cv2.dnn`, YOLOv8n ONNX weights, `conf_threshold = 0.35`, `nms_threshold = 0.45`).
- **Components Disabled:**
  - Image registration, Hungarian associator, temporal engine, persistent SQLite memory.
- **Cost Formulation & Assignment:** None. Detections in visit $v_t$ are treated as independent observations.
- **Change Detection Policy:**
  - Pairwise comparison between visit $v_t$ and initial visit $v_0$ based purely on raw detector IoU.
  - If a label detected in $v_0$ is missing in $v_t$, predict `REMOVED`.
  - If detected at new pixel coordinates ($\Delta d > 35$ px), predict `MOVED`.
- **Expected Failure Mode:** Momentary detector false negatives (flickering) cause instant false removal declarations. Camera viewpoint translation causes instant false movement declarations.

#### B1 — YOLO + Standard Hungarian Association
- **Components Enabled:**
  - `ONNXObjectDetector`.
  - `CrossVisitEntityAssociator` with naive assignment:
    $$C_{ij} = 1.0 - \left(0.50 \cdot \text{IoU}_{ij} + 0.50 \cdot \text{sim}_{ij}\right)$$
- **Components Disabled:**
  - Image registration quality gating (always assumes frames are aligned).
  - Ambiguity margin (forces assignment to highest affinity regardless of competitor closeness).
  - Multi-visit vacancy verification and `PENDING_REMOVAL` state.
- **Change Detection Policy:**
  - If an entity track is not associated in visit $v_t$, it is immediately marked `REMOVED`.
  - If associated at new position, marked `MOVED`.
- **Expected Failure Mode:** Missed detections immediately drop tracks; camera perspective shifts produce erroneous IoU drops; near-identical objects cause silent identity swaps.

#### B2 — Association + Ungated Temporal Reasoning
- **Components Enabled:**
  - `ONNXObjectDetector`.
  - `CrossVisitEntityAssociator`.
  - Multi-visit state transitions (`CONFIRMED_PRESENT`, `PENDING_REMOVAL`, `CONFIRMED_REMOVED`, `CONFIRMED_MOVED`).
- **Components Disabled (Safeguards Bypassed in Evaluation Mode):**
  - *Registration Gating Disabled:* Allows state updates and absence reasoning even when camera registration fails or inlier ratio $< 0.20$.
  - *Vacancy Verification Disabled:* Any non-detection advances removal streak, skipping patch appearance check.
  - *Field-of-View Gating Disabled:* Objects outside camera overlap mask are assumed absent rather than `OUT_OF_FRAME`.
  - *Occlusion Gating Disabled:* Occluded canonical regions are assumed vacant.
- **Expected Failure Mode:** Under camera tilt or partial occlusion, objects are prematurely degraded to `PENDING_REMOVAL` and falsely confirmed removed. Under poor registration, noisy coordinates produce false movement events.

#### B3 — Complete ObjectMemory (Proposed System)
- **Components Enabled:**
  - `ONNXObjectDetector` (CPU inference).
  - `PlanarRegistrar` (SIFT, Lowe's ratio $0.75$, RANSAC reproj error $\le 4.0$ px, inlier ratio $\ge 0.20$, scale bounds $[0.15, 6.0]$).
  - `CrossVisitEntityAssociator` with dual-axis ambiguity margin ($0.12$).
  - `TemporalMemoryEngine` with:
    - Registration-conditioned absence gating (freezes on misalignment).
    - Overlap mask check (overlap $< 70\%$ flags `OUT_OF_FRAME`).
    - Foreground occlusion check (IoU $> 0.20$ with another object flags `PARTIALLY_OCCLUDED`).
    - Aligned patch appearance vacancy verification ($\text{patch\_sim} < 0.40$).
    - Multi-visit streak requirement ($k = 2$ consecutive verified vacant visits).
    - Dynamic uncertainty margin $T_{\text{eff}} = T_{\text{base}} + \Delta U$.
    - Coordinate drift mitigation and boundary truncation artifact handling.
  - `SQLiteMemoryStore` with transactional audit logging and `ON DELETE RESTRICT` immutability.

---

## 2. Independent Ablation Experiments

To identify the exact marginal value of each architectural mechanism, six independent ablations are derived by disabling one safeguard from B3 at a time.

```
       [B3: Complete ObjectMemory]
                    |
    +---------------+---------------+---------------+---------------+---------------+---------------+
    |               |               |               |               |               |               |
 [A1: No Reg-    [A2: No Vacancy [A3: No FOV /   [A4: Single-    [A5: No Ambiguity [A6: No Re-ID
  Freeze]         Check]          Occlusion]      Visit Rem]      Margin]          Recovery]
```

### Ablation Specifications

| Ablation ID | Mechanism Modified / Disabled | Exact Code Modification in Eval Mode | Targeted Outcome Metric | Expected Scientific Finding |
|---|---|---|---|---|
| **A1** | **No Registration-Aware Absence Freezing** | In `engine.py:983`, force bypass of `_is_registration_reliable(visit.registration)` check. Allow absence reasoning to proceed on unaligned frames. | **False Removal on Registration Failure Rate** | Demonstrates that without registration gating, camera misalignment causes catastrophic false removals of unobserved objects. |
| **A2** | **No Appearance-Based Vacancy Verification** | In `engine.py:1044`, bypass visual patch extraction and signature cosine similarity check. Treat all unobserved bounding boxes as `is_vacant = True`. | **False Removal on Detector Miss Rate** | Demonstrates that vacancy verification is essential to suppress false removals when the detector suffers momentary false negatives on physical objects. |
| **A3** | **No Visibility & Field-of-View Safeguards** | In `engine.py:1012` and `1026`, bypass `overlap_mask` check and inter-detection IoU occlusion check. | **False Removal on Out-of-Frame & Occlusion Rate** | Proves that checking camera frustum overlap and foreground occlusions prevents mistaking out-of-frame or hidden objects for removals. |
| **A4** | **Single-Visit Removal Confirmation** | Set `vacancy_visits_for_removal = 1` (down from 2). Transition immediately `CONFIRMED_PRESENT` $\to$ `CONFIRMED_REMOVED` on the first non-detection. | **Removal Confirmation Latency ($RCL$) vs. False Removal Rate ($FRR$)** | Quantifies the fundamental latency-precision trade-off: single-visit removes objects 1 visit faster but drastically inflates false removals. |
| **A5** | **No Ambiguity Handling in Association** | Set `ambiguity_margin = 0.0`. Disable dual-axis candidate competition rejection in `associator.py:147-162`. | **Identity Switches ($ID_{\text{sw}}$) & Duplicate Track Rate ($DTR$)** | Proves that ambiguity gating prevents silent false identity mergers when identical distractors reside in close proximity. |
| **A6** | **No Historical Identity Recovery** | In `engine.py:613` and `377`, disable matching re-appearing objects to previously confirmed removed entities. Treat all re-appearances as brand-new entity creations. | **Track Fragmentation & Re-ID Accuracy** | Proves the necessity of long-term memory for recognizing returned objects rather than polluting the database with duplicate entities. |

---

## 3. Comprehensive Experiment Grid

| Experiment Code | Dataset | System Variant | Challenge Evaluated | Primary Metric | Secondary Metric |
|---|---|---|---|---|---|
| **`EXP-01-B0`** | Physical Benchmark (`PHYS-01` to `05`) | Baseline B0 | Detection-only memoryless baseline | Detection F1, Epistemic Accuracy | False Removal Rate |
| **`EXP-02-B1`** | Physical Benchmark (`PHYS-01` to `05`) | Baseline B1 | Standard tracking baseline | Identity Consistency, FRR | Unsupported Movement Rate |
| **`EXP-03-B2`** | Physical Benchmark (`PHYS-01` to `05`) | Baseline B2 | Ungated temporal reasoning | Epistemic Accuracy | False Removal on Present |
| **`EXP-04-B3`** | Physical Benchmark (`PHYS-01` to `05`) | Proposed B3 | Full proposed system | Epistemic Accuracy, IDC | False Removal Rate, RCL |
| **`EXP-05-A1`** | `PHYS-04-VIEW` & OpenLORIS pairs | Ablation A1 | Camera alignment failure | False Removals on Misalignment | Registration Reliability |
| **`EXP-06-A2`** | `PHYS-01-STAT` & `PHYS-03-REM` | Ablation A2 | Detector false negatives | False Removals on Miss | Vacancy Precision |
| **`EXP-07-A3`** | `PHYS-03-REM` & `PHYS-04-VIEW` | Ablation A3 | Occlusion & Out-of-frame | False Removals on Occlusion | Out-of-Frame Accuracy |
| **`EXP-08-A4`** | `PHYS-03-REM` | Ablation A4 | Removal confirmation policy | Removal Confirmation Latency | False Removal Rate |
| **`EXP-09-A5`** | `PHYS-05-DIST` | Ablation A5 | Identical distractor ambiguity | Identity Switch Count | Ambiguity Abstention Rate |
| **`EXP-10-A6`** | `PHYS-03-REM` | Ablation A6 | Object return after removal | Duplicate Track Rate | Re-identification Accuracy |
| **`EXP-11-LORIS`**| OpenLORIS-Scene (`office1-1` vs `1-2`, `1-3`, `1-6`) | Proposed B3 + Evaluator | Real robotic camera motion & lighting | Registration Success Rate | Mean Reprojection Error |

---

## 4. Execution Isolation & Integrity Guarantee

To ensure scientific integrity:
1. **Isolated SQLite Database per Experiment:** Every baseline and ablation run instantiates its own isolated SQLite database in a scratch directory (e.g. `artifacts/experiments/{exp_id}.sqlite3`). The production application database is never touched.
2. **Deterministic Random Seed:** Any randomized sampling or RANSAC feature matching initializes `np.random.seed(42)` and `cv2.setRNGSeed(42)`.
3. **Identical Visual Input Vectors:** Pre-computed YOLOv8n detections and visual signatures are exported to disk once and shared across B0–B3 and A1–A6 to guarantee that differences in temporal performance stem strictly from temporal reasoning logic, not detector variability.
