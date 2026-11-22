# ObjectMemory Research — Experimental Protocol & Benchmark Design (Parts E & F)

**Document Version:** 1.0.0  
**Date:** November 2026  
**Auditor:** Antigravity Autonomous Research & Verification Suite  
**Project:** ObjectMemory (Long-Term Visual Memory for Persistent Environmental Changes)  
**Target File:** `research/experimental_protocol.md`

---

## 1. Executive Summary

This protocol defines a rigorous, publication-grade experimental benchmark design to empirically validate the scientific claims of `ObjectMemory`. It evaluates dataset suitability across standard public benchmarks and our custom physical workbench datasets, formalizes four standardized baselines, specifies seven fine-grained evaluation metrics, details five targeted ablation studies, and establishes strict statistical reporting requirements.

---

## 2. Part E: Dataset Suitability Audit

We evaluate three candidate evaluation benchmarks: **OpenLORIS-Scene** (ICRA 2020), **ChangeSim** (IROS 2021), and the **Physical Workbench Dataset** (`data/real_workbench/` and `data/physical_workbench_v1/`).

```
+---------------------------------------------------------------------------------------------------+
|                                  DATASET SUITABILITY SUMMARY                                      |
+--------------------------+-----------------------+-----------------------+------------------------+
| Dimension                | OpenLORIS-Scene (2020)| ChangeSim (IROS 2021) | Physical Workbench     |
+--------------------------+-----------------------+-----------------------+------------------------+
| Environment              | 5 Real Indoor Arenas  | UE4 Synthetic Factory | Real Desktop/Workbench |
| Trajectory / Viewpoint   | 6-DoF Wheeled Robot   | 6-DoF Simulated Drone | Monocular Multi-Visit  |
| Geometry Paradigm        | Non-planar 3D Metric  | Non-planar 3D Metric  | Dominant Planar Surface|
| Primary Annotation       | SLAM Trajectory / Poses| Pixel Change Masks    | Entity BBoxes & Events |
| Native Planar Homography | Low (3D clutter)      | Low (Warehouse aisles)| High (Planar Worktop)  |
| Immediate Plug-and-Play  | Needs Monocular Ortho | Needs Planar Slicing  | Fully Compatible       |
| Role in Research         | Generalization / Drift| Scaled Stress Testing | Primary Validation     |
+--------------------------+-----------------------+-----------------------+------------------------+
```

---

### 2.1 Dataset 1: OpenLORIS-Scene (ICRA 2020)
* **Citation:** Xuesong Shi et al., *IEEE ICRA 2020*.
* **Environment & Sensors:** Real indoor environments (office, home, cafe, corridor) recorded across multiple sessions by a wheeled service robot. Sensors include Intel RealSense D435i (RGB-D), T265 (fisheye stereo + IMU), and wheel encoders.
* **Dynamic Elements & Annotations:** Dynamic human traffic, furniture relocations, seasonal lighting variations. Ground truth provides high-precision sub-centimeter multi-camera trajectory poses from optical motion capture (OptiTrack).
* **Suitability for ObjectMemory:**
  - *High Value for Perceptual Drift:* Excellent for evaluating illumination robustness (morning vs. noon vs. night lighting changes).
  - *Challenging for Object-Level Changes:* Does not natively include fine-grained per-object 2D/3D bounding box trajectories or annotated event transitions (`MOVED`, `REMOVED`, `REAPPEARED`).
* **Limitations for ObjectMemory's Specific Scope:**
  - ObjectMemory is architecturally designed around dominant planar workspaces (homography $H \in \mathbb{R}^{3 \times 3}$). OpenLORIS consists of arbitrary 3D free-space robot navigation with severe 3D parallax. Direct planar homography estimation fails in open corridors.
* **Required Adaptations:**
  - Extract tabletop or desk subsets from the `office` or `home` sequences where the robot pauses in front of a flat work surface.
  - Manually annotate 2D object bounding boxes and temporal state transitions across visits.

---

### 2.2 Dataset 2: ChangeSim (IROS 2021)
* **Citation:** Jin-Man Park et al., *IEEE/RSJ IROS 2021*.
* **Environment & Sensors:** Photorealistic synthetic industrial warehouse built in Unreal Engine 4. Provides multi-modal sensor feeds: RGB, depth, semantic segmentation, normal maps, and noisy pose estimates.
* **Dynamic Elements & Annotations:** Repeated sweeps of industrial aisles containing movable crates, pallets, and equipment. Changes include object insertions, removals, and replacements under varying lighting, dust, and non-overlapping camera sweeps. Provides dense pixel-level ground-truth change masks.
* **Suitability for ObjectMemory:**
  - *Exceptional Scalability:* Thousands of sequential pairs/sweeps with perfectly known ground truth.
  - *Systematic Stress Testing:* Parameterizable lighting degradation, dust density, and camera viewpoint angles.
* **Limitations for ObjectMemory's Specific Scope:**
  - Evaluated on dense 2D pixel difference segmentation masks rather than persistent relational entity graphs with identity re-identification.
  - Multi-aisle industrial navigation has significant non-planar depth variation.
* **Required Adaptations:**
  - Select planar warehouse loading docks, packing tables, or flat pallet floors where a dominant ground plane exists.
  - Convert dense binary change masks into discrete bounding boxes and associate them into multi-visit entity histories.

---

### 2.3 Dataset 3: Physical Workbench Dataset (Our Benchmark)
* **Workspace Path:** `data/real_workbench/` (V01–V12) and `data/physical_workbench_v1/` (V01–V10).
* **Environment & Sensors:** Real desktop workspace monitored by a monocular RGB camera (smartphone and external webcam, $1920 \times 1080$, uncalibrated).
* **Dynamic Elements & Annotations:**
  - Everyday household and office entities: `bottle_01`, `notebook_01`, `cup_01`, `marker_01`.
  - Rich dynamic events across 10–12 discrete visits: static baseline, subtle translation, large translation, return-to-baseline, temporary visual occlusion (hand / foreign object occluding for 1–2 visits), confirmed permanent removal, reappearance after multi-visit absence, camera pan/tilt, and registration stress visits.
  - Ground-truth manifest (`manifest.json`) provides verified visit-level canonical coordinates $[x_{min}, y_{min}, x_{max}, y_{max}]$ and entity state annotations.
* **Suitability for ObjectMemory:**
  - Directly matches the target problem: long-term planar visual memory with fine-grained state transitions and forensic audit requirements.
* **Limitations:**
  - Relatively small sample size (10–12 visits per sequence, 4–6 objects per scene).
* **Enhancements for Paper Submission:**
  - Expand to 5 independent physical workbench scenes (varying textures, lighting conditions, and object densities) to demonstrate statistical significance across at least 60 total visits and 30 unique entities.

---

## 3. Part F: Experimental Protocol Design

### 3.1 Standardized Baselines

To prove that each component of `ObjectMemory` provides measurable scientific value, we define four standardized baselines:

```
+---------------------------------------------------------------------------------------------------+
|                                      EXPERIMENTAL BASELINES                                       |
+---------------------------------------------------------------------------------------------------+
| Baseline 1: Detection-Only (Zero Temporal Memory)                                                |
|   - Architecture: Raw YOLOv8n inference on each incoming frame independently.                    |
|   - Tracking/Memory: None. No cross-visit association.                                            |
|   - Change Logic: None. Only reports instantaneous detections.                                    |
+---------------------------------------------------------------------------------------------------+
| Baseline 2: Detection + Naive Association (Standard Tracking-by-Detection)                        |
|   - Architecture: YOLOv8n + Kuhn-Munkres matching based on raw pixel IoU / Euclidean distance.   |
|   - Tracking/Memory: Kalman filter in image pixel coordinates. Tracks deleted after N misses.    |
|   - Change Logic: Distance threshold in pixel space; no homography registration; no vacancy check.|
+---------------------------------------------------------------------------------------------------+
| Baseline 3: Temporal Reasoning WITHOUT Registration Safeguards (Ungated Memory)                   |
|   - Architecture: YOLOv8n + Kuhn-Munkres + 6-state FSM.                                           |
|   - Tracking/Memory: Warps to canonical space using raw homography without condition gating.     |
|   - Change Logic: Advances absence streak on any missed detection; no patch vacancy check.        |
+---------------------------------------------------------------------------------------------------+
| Baseline 4: Full ObjectMemory (Ours)                                                              |
|   - Architecture: Complete verified pipeline.                                                     |
|   - Tracking/Memory: Decoupled raw detection store + 6-gate homography verification.              |
|   - Change Logic: Dynamic threshold (35px + uncertainty), 2-gate vacancy verification,           |
|     return-to-baseline detection, ambiguity margin check, and forensic audit logging.            |
+---------------------------------------------------------------------------------------------------+
```

---

### 3.2 Evaluation Metrics

We establish seven rigorous evaluation axes:

#### 1. Detection Performance
- **$\text{mAP@50}$**: Mean Average Precision at $\text{IoU} \ge 0.50$ in canonical space.
- **Precision / Recall / F1-Score**: Evaluated against ground-truth object bounding boxes.

#### 2. Identity Consistency & Association
- **$\text{IDF1}$ (Identification F1-score)**: Harmonic mean of identification precision and recall across visits.
- **$\text{IDSW}$ (Identity Switches)**: Number of times an entity is incorrectly assigned a different entity ID.
- **$\text{HOTA}$ (Higher Order Tracking Accuracy)**: Balanced evaluation of detection, association, and localization accuracy.

#### 3. Movement Detection Accuracy
- Evaluated on visits where an object was translated ($d > T_{\text{effective}}$):
  $$\text{Movement Precision} = \frac{TP_{\text{move}}}{TP_{\text{move}} + FP_{\text{move}}}, \quad \text{Movement Recall} = \frac{TP_{\text{move}}}{TP_{\text{move}} + FN_{\text{move}}}$$
- *Crucial Test:* $FP_{\text{move}}$ under camera viewpoint jitter and unaligned visits.

#### 4. Removal Detection Accuracy & Delay
- **Removal F1-Score**: Detection of permanent object extraction.
- **Mean Confirmation Delay ($\bar{\tau}_{\text{removal}}$)**: Average number of visits between true physical removal and system state transitioning to `CONFIRMED_REMOVED` (expected $\bar{\tau} = 2.0$ visits).
- **False Removal Rate ($FPR_{\text{removal}}$)**: Spurious removals triggered during temporary occlusions or camera misalignments.

#### 5. Reappearance & Re-Identification Accuracy
- **Reappearance Recall**: Fraction of returned objects correctly linked to their historical entity record (instead of initializing an extraneous new entity).
- **Historical Signature Specificity**: Fraction of distinct objects sharing the same class that are NOT erroneously cross-matched upon reappearance.

#### 6. Temporal State Transition Accuracy
- **Entity-Visit State Accuracy**: Fraction of total evaluated entity-visits ($N = 42$ in Real Workbench) where predicted state matches ground truth:
  $$\text{Acc}_{\text{state}} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(\hat{S}_{i,v} == S_{i,v}^*)$$

#### 7. Evidence Robustness & Fault Tolerance
- **Sensor Evidence Retention Rate ($R_{\text{raw}}$)**:
  $$R_{\text{raw}} = \frac{N_{\text{persisted detections in unregistered frames}}}{N_{\text{ground truth objects in unregistered frames}}}$$
  (Target: $100\%$ for ObjectMemory; $0\%$ for standard SLAM/filtering pipelines).

---

### 3.3 Targeted Ablation Studies

To empirically justify the specific architectural design choices, five ablations must be conducted:

```
+---------------------------------------------------------------------------------------------------+
|                                     TARGETED ABLATION STUDIES                                     |
+---------------------------------------------------------------------------------------------------+
| Ablation 1: Registration Quality Gating                                                           |
|   - Variant A: Full 6-criterion algebraic gate (finite, cond < 10^7, inliers >= 12, etc.).        |
|   - Variant B: Naive RANSAC homography without quality gating (no condition number or quad check). |
|   - Variant C: No homography (evaluate purely in pixel frame coordinates).                        |
|   - Hypothesis: Variant B triggers catastrophic false movements/removals under camera tilt.       |
+---------------------------------------------------------------------------------------------------+
| Ablation 2: Vacancy Verification Module                                                           |
|   - Variant A: Full two-gate check (IoU/FoV occlusion guard + 2-visit patch similarity < 0.35).   |
|   - Variant B: Absence streak counter only (streak >= 2 confirms removal, no patch check).        |
|   - Variant C: Single-visit patch check (streak = 1 confirms removal).                            |
|   - Hypothesis: Variant B yields high FPR_removal under temporary occlusion; Variant C yields     |
|     premature false removals on detector flickering.                                              |
+---------------------------------------------------------------------------------------------------+
| Ablation 3: Identity Ambiguity Margin Check                                                        |
|   - Variant A: Ambiguity check active (Δaff < 0.12 transitions to AMBIGUOUS_IDENTITY).           |
|   - Variant B: Hard greedy Kuhn-Munkres assignment without margin checks.                         |
|   - Hypothesis: Variant B causes permanent identity swaps when two identical cups are collocated. |
+---------------------------------------------------------------------------------------------------+
| Ablation 4: Baseline Anchor & Return-to-Baseline Recovery                                         |
|   - Variant A: Preserves t_0 baseline anchor and verifies distance <= T_effective.                 |
|   - Variant B: Memoryless state machine (evaluates distance only relative to visit t-1).           |
|   - Hypothesis: Variant B cannot distinguish an object returning home from a new displacement,     |
|     accumulating unbounded cumulative drift over multi-hop movements.                             |
+---------------------------------------------------------------------------------------------------+
| Ablation 5: Dynamic Uncertainty-Inflated Movement Threshold                                       |
|   - Variant A: Dynamic threshold T_effective = 35.0px + Δu (scaled by condition number & error).  |
|   - Variant B: Fixed static threshold T_fixed = 35.0px.                                           |
|   - Variant C: Fixed conservative threshold T_fixed = 60.0px.                                     |
|   - Hypothesis: Variant B has high false movement alarms on oblique views; Variant C misses       |
|     subtle genuine 35–50px relocations.                                                           |
+---------------------------------------------------------------------------------------------------+
```

---

### 3.4 Statistical Rigor & Reporting Guidelines

For submission to leading computer vision and robotics conferences (e.g., CVPR, ECCV, IROS, ICRA, RA-L):

1. **Repetition & Variance:**
   - Every stochastic operation (feature extraction, RANSAC fitting, detector inference) must be evaluated across **5 fixed random seeds** ($\{42, 101, 777, 1337, 2026\}$).
   - Report all results as $\text{Mean} \pm \text{Standard Deviation}$ (e.g., $\text{IDF1} = 91.4 \pm 1.2\%$).

2. **Significance Testing:**
   - Paired two-tailed Student's $t$-test or Wilcoxon signed-rank test across benchmark sequences comparing Baseline 4 against Baseline 3 and Baseline 2.
   - Only claim statistical superiority when $p < 0.05$.

3. **Hyperparameter Transparency:**
   - Explicitly report all parameter values in an appendix table:
     - Feature detector: SIFT (1000 features, Lowe ratio 0.75, RANSAC threshold 3.0px)
     - Homography bounds: $\text{cond} < 10^7$, inliers $\ge 12$, ratio $\ge 0.20$, reprojection $\le 4.0$px
     - Association: Visual signature cosine $\ge 0.65$, IoU weight $+0.10$, ambiguity $\Delta \text{aff} < 0.12$
     - Temporal states: Movement baseline $35.0$px, vacancy cosine $< 0.35$, confirmation streak $= 2$.
