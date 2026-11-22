# ObjectMemory Research Validation — Evaluation Split Protocol (Task 5)

**Document Version:** 1.0.0  
**Date:** November 2026  
**Auditor:** Antigravity Autonomous Research Suite  
**Project:** ObjectMemory (Long-Term Visual Memory for Persistent Environmental Changes)  
**Target File:** `research_validation/evaluation_split_protocol.md`

---

## 1. Cardinal Evaluation Rules

To prevent data leakage, inflated performance metrics, and methodological invalidity, the following rules are strictly enforced:

1. **Strict Sequence-Level Disjoint Splits:**  
   Frames belonging to a single continuous video trajectory or visit timeline are **never randomly partitioned** across development, validation, and test splits. Entire continuous sequences (trajectories or multi-visit setups) are assigned to one split as a single atomic unit.
2. **Separation of Real-World and Synthetic Results:**  
   Simulation results (ChangeSim) and real-world physical benchmark results (OpenLORIS-Scene, Physical Workbench) are **never aggregated into a single blended metric**. They are reported in strictly separate tables.
3. **Immutability of Held-Out Test Sequences:**  
   Test sequences are never inspected during threshold tuning or prompt engineering. Evaluation on test splits occurs strictly once under fixed hyperparameters.

---

## 2. Sequence-Level Partitioning Strategy

```
+---------------------------------------------------------------------------------------------------+
|                                 SEQUENCE SPLIT ALLOCATION MATRIX                                  |
+----------------------+-----------------------+-----------------------+----------------------------+
| Dataset              | Development Split     | Validation Split      | Held-Out Test Split        |
+----------------------+-----------------------+-----------------------+----------------------------+
| **OpenLORIS-Scene**  | `office1-1` (Base)    | `office1-2` (Lighting)| `office1-3` (Moved Obj)    |
| (Real Robot 6-DoF)   | (SIFT/ORB calibration)| `office1-4` (Transit) | `office1-5`, `office1-6`   |
+----------------------+-----------------------+-----------------------+----------------------------+
| **ChangeSim**        | `Warehouse_0 / Seq_0` | `Warehouse_0 / Seq_1` | `Warehouse_0 / Seq_0_dust` |
| (Simulation)         | (Reference + Query)   | (Different Trajectory)| `Warehouse_1 / Seq_0`      |
+----------------------+-----------------------+-----------------------+----------------------------+
| **Physical**         | `data/real_workbench` | `S01_stationary`      | `S02_movement_return`      |
| **Workbench**        | (V01–V12 historical)  | `S06_illumination`    | `S03_persistent_reloc`     |
| (In-Situ Planar)     |                       |                       | `S04_removal`, `S05_occlus`|
+----------------------+-----------------------+-----------------------+----------------------------+
| **OpenLORIS-Object** | Factor: Normal Light  | Factor: Clutter       | Factor: Weak/Strong Light  |
| (Detector Stress)    | Occlusion 0%          | Normal/Complex        | Occlusion 25%, 50%         |
+----------------------+-----------------------+-----------------------+----------------------------+
```

---

### 2.1 OpenLORIS-Scene Splits
OpenLORIS-Scene provides multi-session trajectories across the exact same spatial environment (`office1`). We assign sequences chronologically:
* **Development Split (`office1-1`):**  
  Used to calibrate feature detector thresholds (max features, Lowe ratio, RANSAC reprojection error) on office carpet and desk textures.
* **Validation Split (`office1-2`, `office1-4`):**  
  Used to tune keyframe promotion policies and verify homography condition number gating under ambient lighting transitions.
* **Held-Out Test Split (`office1-3`, `office1-5`, `office1-6`, `office1-7`):**  
  Reserved exclusively for final benchmarking. Contains genuine object relocations and multi-visit absences.

---

### 2.2 ChangeSim Splits
ChangeSim provides distinct simulated warehouse environments and query trajectories:
* **Development Split (`Warehouse_0 / Seq_0` Reference & Query):**  
  Used to establish baseline planar ground extraction and verify coordinate alignment between paired passes.
* **Validation Split (`Warehouse_0 / Seq_1`):**  
  Evaluates alternative flight trajectories through the same warehouse to test generalization over different viewpoints.
* **Held-Out Test Split (`Warehouse_0 / Seq_0_dust` & `Warehouse_1 / Seq_0`):**  
  - `Warehouse_0 / Seq_0_dust`: Stress tests system robustness under severe particulate turbidity without modifying thresholds.
  - `Warehouse_1 / Seq_0`: Unseen warehouse layout testing spatial memory initialization from scratch.

---

### 2.3 Physical Workbench Splits
* **Development Split (`data/real_workbench/` V01–V12):**  
  Historical 12-visit desktop pilot used for bug fixing, regression tests, and initial metric validation.
* **Validation Split (`physical_workbench_v1` Scenarios S01 & S06):**  
  - `S01_stationary_baseline`: 5 visits verifying 0% false movement rate.
  - `S06_illumination_stress`: Fluorescent vs. incandescent lighting.
* **Held-Out Test Split (`physical_workbench_v1` Scenarios S02, S03, S04, S05, S07):**  
  Evaluates the core epistemic capabilities:
  - `S02_temporary_movement_return`: `MOVED_BACK_TO_BASELINE` detection.
  - `S03_persistent_relocation`: Displaced entity lifecycle.
  - `S04_permanent_removal`: Two-gate vacancy verification.
  - `S05_partial_occlusion`: Occlusion-gated removal suppression.
  - `S07_viewpoint_perturbation`: Homography quality gating under oblique angles.

---

## 3. Reporting Protocols & Statistical Metrics

### 3.1 Separate Reporting Tables
Evaluation papers will present results across three independent tables:
1. **Table 1: In-Situ Real-World Workbench Results (`Physical Workbench`):** Primary evaluation of fine-grained entity states, baseline recovery, and explainable audit trails.
2. **Table 2: Lifelong Robotic Vision Drift Results (`OpenLORIS-Scene`):** Evaluation of long-term feature registration, unaligned frame fault-tolerance, and absence freezing.
3. **Table 3: Scaled Controlled Simulation Results (`ChangeSim`):** Evaluation of change detection precision/recall and change localization against dense ground truth masks.

### 3.2 Evaluation Metrics per Sequence
* **State Transition Accuracy ($\text{Acc}_{\text{state}}$):** Fraction of entity-visits whose epistemic state matches ground truth.
* **False Removal Rate ($FPR_{\text{removal}}$):** Percentage of temporary occlusions or unaligned frames incorrectly classified as removed.
* **Mean Confirmation Delay ($\bar{\tau}_{\text{removal}}$):** Number of visits between physical removal and confirmation (expected: $2.0$ visits).
* **Return-to-Baseline Recall:** Percentage of returned objects correctly classified as `MOVED_BACK_TO_BASELINE`.
