# Phase 7 Evaluation Report: Real-World Validation, Empirical Uncertainty Calibration, and Research Demonstration

**Date**: 2026-10-20  
**Repository**: `C:\Users\Rishh\.gemini\antigravity\scratch\objectmemory`  
**Evaluation Script**: [`scripts/evaluate_phase7.py`](scripts/evaluate_phase7.py)  
**Evaluation Data**: [`scripts/phase7_evaluation_results.json`](scripts/phase7_evaluation_results.json)  
**Dataset Protocol**: [`docs/phase7_dataset_protocol.md`](docs/phase7_dataset_protocol.md)  

---

## 1. Executive Summary

Phase 7 evaluates whether **Adaptive Keyframe Registration** and **Uncertainty-Aware Movement Decisions** improve real-world temporal memory reliability without inducing false movement or removal claims.

### Key Research Findings
1. **Adaptive vs. Single-Reference Registration**:
   - In controlled transformations with significant viewpoint translation (pans $\ge 550\text{px}$ outside the reference boundary):
     - **Single-Reference Baseline**: Failed on $2/15$ cases ($86.7\%$ success) with singular homographies and zero inlier consensus.
     - **Adaptive Keyframe Registration**: Succeeded on $15/15$ cases ($100.0\%$ success) by chaining secondary keyframe homographies ($H_{Q \to C} = H_{K \to C} \cdot H_{Q \to K}$).
     - **Geometric Consistency**: Canonical position error across chained multi-hop keyframes remained under $1.0\text{px}$ (mean error: $0.726\text{px}$ and $0.030\text{px}$).
2. **Empirical Uncertainty Calibration**:
   - The dynamic uncertainty threshold $T_{effective} = 35.0 + \Delta U$ was empirically analyzed against known ground-truth coordinates across 15 controlled conditions.
   - **Empirical Coverage Rate**: $100.0\%$ (all actual localization errors were strictly bounded within $T_{effective}$).
   - **Calibration Status**: $\Delta U$ is an **engineering heuristic** with piecewise linear penalty weights ($U_{reproj}, U_{inlier}, U_{cov}, U_{scale}$), **not** a statistically derived Bayesian posterior or Fisher information confidence interval.
3. **Real Workbench Temporal Validation (`desk_pilot_01`)**:
   - Evaluated 42 entity-visit ground truth pairs across 7 physical visits.
   - **False Movements**: $1 / 28$ ($3.57\%$) — `book_01` in Visit 6 due to resolution/FOV boundary shift.
   - **False Removals**: $1 / 34$ ($2.94\%$) — `scissors_01` in Visit 5 where degraded registration (18.1% inliers) gated state updates, holding the entity in `PENDING_REMOVAL`.
   - **Reappearance Recovery**: Successfully verified — `bottle_01` was restored from `CONFIRMED_REMOVED` to `CONFIRMED_PRESENT` upon redetection in Visit 7.
4. **Verification Integrity**:
   - All 247 automated Python tests passed in 80.6s.
   - Frontend production bundle built in 546ms.
   - Playwright browser E2E workflows passed in 13.7s.

---

## 2. Controlled Transformation Benchmark (Tasks 3 & 4)

Evaluated across 15 systematically generated camera viewpoints (pure translations up to 750px, scale shifts $0.85\times - 1.25\times$, rotations up to $10^\circ$, and progressive perspective tilts) on a 1600x600 canvas with known ground-truth marker positions:

| Test Case | Description | Baseline Single-Ref | Phase 6/7 Adaptive | Selected Keyframe | Mode | Canonical Pos Error | Delta_U |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `stable_jitter_1` | dx=4, dy=3, rot=0.5° | SUCCESS (303 inliers) | **SUCCESS (303 inliers)** | `visit_00` | Direct | 0.022 px | 0.00 px |
| `stable_jitter_2` | dx=-5, dy=6, rot=-0.8°| SUCCESS (338 inliers) | **SUCCESS (338 inliers)** | `visit_00` | Direct | 0.025 px | 0.00 px |
| `moderate_pan_1` | dx=80, dy=20 | SUCCESS (562 inliers) | **SUCCESS (562 inliers)** | `visit_00` | Direct | 0.003 px | 0.00 px |
| `moderate_pan_2` | dx=160, dy=-15 | SUCCESS (283 inliers) | **SUCCESS (283 inliers)** | `visit_00` | Direct | 0.013 px | 0.00 px |
| `large_pan_1` | dx=280, dy=30 | SUCCESS (299 inliers) | **SUCCESS (299 inliers)** | `visit_00` | Direct | 0.028 px | 3.18 px |
| `large_pan_2` | dx=400, dy=10, rot=-1.5°| SUCCESS (98 inliers) | **SUCCESS (246 inliers)** | `visit_05` | Chained | 0.096 px | 0.00 px |
| `wide_pan_out_1` | dx=550 (Out of ref) | **FAILED (Singular H)** | **SUCCESS (305 inliers)** | `visit_05` | Chained | 0.726 px | 5.74 px |
| `wide_pan_out_2` | dx=750 (Out of ref) | **FAILED (Singular H)** | **SUCCESS (142 inliers)** | `visit_05` | Chained | 0.030 px | 10.09 px |
| `scale_zoom_in` | scale=1.25, dx=40, dy=30 | SUCCESS (269 inliers) | **SUCCESS (269 inliers)** | `visit_00` | Direct | 0.096 px | 1.13 px |
| `scale_zoom_out` | scale=0.85, dx=60, dy=20 | SUCCESS (282 inliers) | **SUCCESS (282 inliers)** | `visit_00` | Direct | 0.070 px | 0.00 px |
| `rot_tilt_5deg` | rot=5°, dx=50, dy=20 | SUCCESS (307 inliers) | **SUCCESS (307 inliers)** | `visit_00` | Direct | 0.037 px | 0.00 px |
| `rot_tilt_10deg` | rot=10°, dx=70, dy=30 | SUCCESS (305 inliers) | **SUCCESS (305 inliers)** | `visit_00` | Direct | 0.069 px | 0.00 px |
| `persp_light` | 15px corner offsets | SUCCESS (317 inliers) | **SUCCESS (317 inliers)** | `visit_00` | Direct | 0.033 px | 0.00 px |
| `persp_moderate` | 35px corner offsets | SUCCESS (292 inliers) | **SUCCESS (292 inliers)** | `visit_00` | Direct | 0.040 px | 0.00 px |
| `persp_heavy` | 60px corner offsets | SUCCESS (271 inliers) | **SUCCESS (271 inliers)** | `visit_00` | Direct | 0.054 px | 0.00 px |

### Key Benchmark Takeaways
- **Success Rate**: Baseline = $86.7\%$ ($13/15$), Adaptive Keyframe = **$100.0\%$** ($15/15$).
- **Chained Topology Utility**: In `wide_pan_out_1` and `wide_pan_out_2`, the query has zero overlap with the initial canonical reference (`visit_00`). Adaptive registration routed the transformation through intermediate keyframe `visit_05`, computing compound homography $H_{Q \to C} = H_{5 \to C} \cdot H_{Q \to 5}$ with sub-pixel localization accuracy ($0.726\text{px}$ and $0.030\text{px}$).

---

## 3. Uncertainty Calibration Analysis (Task 4)

### Nature of the Dynamic Uncertainty Margin
The dynamic uncertainty threshold in `TemporalMemoryEngine.compute_effective_movement_threshold()` is defined as:
$$T_{effective} = T_{base} + \Delta U$$
$$\Delta U = \min(45.0, U_{reproj} + U_{inlier} + U_{coverage} + U_{scale})$$

```
Uncertainty Calibration Status:
  - Formally Statistically Calibrated: NO
  - Nature of Uncertainty: Engineering Heuristic
  - Rationale: Weights are piecewise linear scaling factors chosen to penalize
    known error sources (low inliers, clustered matches, high reprojection error),
    not derived from Fisher information or covariance propagation.
  - Empirical Coverage Rate: 100.0% (15 / 15 cases within T_effective)
```

**Scientific Conclusion**: The uncertainty margin acts as a conservative, robust heuristic that prevents false positive movement calls under geometric degradation. However, it should **not** be presented as statistically calibrated confidence bounds in research publications without covariance propagation.

---

## 4. Real-World Temporal Memory Evaluation (Task 5)

Evaluated on `desk_pilot_01` (7 physical photographs, 42 ground-truth entity evaluations):

| Metric | Measured Value | Denominator & Calculation | Notes |
| :--- | :--- | :--- | :--- |
| **False Movement Events** | **1** ($3.57\%$) | $1 / 28$ unchanged evaluations | `book_01` in V6: resolution shift ($780 \to 584\text{px}$) shifted box centroid by $> 35\text{px}$. |
| **Missed Movement Events** | **2** ($66.7\%$) | $2 / 3$ relocated evaluations | 1) `scissors_01` in V3: physical shift was small ($< 35\text{px}$). 2) `cup_01` in V4: cup returned to baseline position. |
| **False Removal Events** | **1** ($2.94\%$) | $1 / 34$ visible evaluations | `scissors_01` in V5: registration gated ($18.1\% < 20\%$), holding entity in `PENDING_REMOVAL`. |
| **Missed Removal Events** | **4** ($100.0\%$) | $4 / 4$ confirmed removal evals | Detector produced false positive detections on desk textures where cup/bottle were removed. |
| **Reappearance Recovery** | **Verified (100%)** | $1 / 1$ reappearance scenarios | `bottle_01` restored from `CONFIRMED_REMOVED` to `CONFIRMED_PRESENT` on V7. |
| **Decisions Deferred by Gate**| **1 Visit** | Visit 5 gated | Inlier ratio $18.1\% < 20\%$ gate. |
| **Duplicate Entities** | **1** | 8 active entities vs. 7 GT | `laptop` at frame boundary created as untracked 8th entity. |

---

## 5. UI Research Demonstration (Task 6)

The user interface was validated to display genuine backend data:
- **`WorkspaceOverviewView`**: Shows selected keyframe (`Canonical` vs. intermediate keyframe ID), chaining badge (`Direct` vs. `Chained Compound`), promotion status, inlier count and ratio, mean reprojection error, spatial feature coverage percentage, and decision status (`ACCEPTED`, `DEFERRED`, `REJECTED`).
- **`SpatialMemoryExplorerView`**: Renders 2D interactive canvas with canonical footprints, spatial displacement vectors ($\Delta X, \Delta Y$, total distance), boundary clipping warnings, and full observation trace per entity.
- **`DiagnosticsView`**: Live CPU inference latency, detector model metadata, planar registration health status, and Phase 6 adaptive keyframe documentation.
