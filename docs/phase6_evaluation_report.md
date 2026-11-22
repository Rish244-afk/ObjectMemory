# Phase 6 Evaluation Report: Adaptive Keyframe Registration & Uncertainty-Aware Spatial Memory

**Date**: 2026-10-20  
**Repository**: `C:\Users\Rishh\.gemini\antigravity\scratch\objectmemory`  
**Evaluation Script**: [`scripts/evaluate_phase6.py`](scripts/evaluate_phase6.py)  
**Evaluation Data**: [`scripts/phase6_evaluation_results.json`](scripts/phase6_evaluation_results.json)  

---

## 1. Executive Summary

Phase 6 implements and evaluates **Adaptive Keyframe Registration** and **Uncertainty-Aware Movement Decisions** for ObjectMemory Studio.

### Headline Results
1. **Wide-Baseline Panning Sequences**:
   - Baseline Single-Reference Registration: **50.0%** success rate (frames 3 and 4 fail as field of view shifts beyond reference boundary).
   - Phase 6 Adaptive Keyframe Registration: **100.0%** success rate (chained secondary keyframes maintain verified compound mapping to canonical space).
2. **False Movement Mitigation**:
   - In 2 out of 3 edge-case jitter scenarios where camera noise and feature extrapolation pushed centroid displacement beyond the 35.0px baseline threshold, dynamic uncertainty margin $\Delta U$ held the belief in `CONFIRMED_PRESENT`, preventing false positive movement alerts.
3. **Physical Multi-Visit Workbench Validation**:
   - 7 physical visit frames evaluated: 6/7 successfully co-registered (85.7%).
   - Visit 5 correctly gated with `INSUFFICIENT_INLIERS`, preventing corrupted coordinate projections.
4. **Complete Automated Test Suite**:
   - **247 Python tests passing** (235 prior + 12 new keyframe regression tests).
   - Frontend production build: compiled cleanly in 546ms.
   - Playwright browser E2E workflows: **2 passed in 13.7s**.

---

## 2. Comparative Evaluation Metrics

### 2.1 Controlled Synthetic Panning Sequence (4 Visits)

| Metric | Phase 5 Baseline (Direct Reference) | Phase 6 (Adaptive Keyframe) | Delta / Improvement |
| :--- | :--- | :--- | :--- |
| **Registration Success Rate** | 50.0% (2 / 4) | **100.0%** (4 / 4) | **+50.0%** |
| **Mean Inlier Ratio** | 46.5% | **90.7%** | **+44.2%** |
| **Mean Reprojection Error** | 2.00 px | **1.00 px** | **-50.0% error** |
| **Keyframes Promoted** | 1 (Fixed) | **4** (Topology) | Topology expansion |
| **Compound Multi-Hop Warps**| 0 | **2** | Seamless chaining |
| **Mean Latency per Frame** | 148.9 ms | 215.4 ms | +66.5 ms (multi-candidate) |

### 2.2 Physical Multi-Visit Workbench Imagery (`data/real_workbench/visits/`)

| Visit ID | Baseline Status | Baseline Inliers | Phase 6 Status | Selected Keyframe | Mode | Decision Reason |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `visit_01` | SUCCESS | 100% (100) | **SUCCESS** | `visit_01` | Canonical Direct | Initial Canonical Reference Established |
| `visit_02` | SUCCESS | 55.3% (78) | **SUCCESS** | `visit_01` | Canonical Direct | Inlier ratio 55.3% $\ge 20\%$ gate |
| `visit_03` | SUCCESS | 34.1% (29) | **SUCCESS** | `visit_01` | Canonical Direct | Inlier ratio 34.1% $\ge 20\%$ gate |
| `visit_04` | SUCCESS | 22.2% (18) | **SUCCESS** | `visit_01` | Canonical Direct | Inlier ratio 22.2% $\ge 20\%$ gate |
| `visit_05` | INSUFFICIENT_INLIERS | 18.1% (17) | **REJECTED** | `visit_01` | Gated Safeguard | Inlier ratio 18.1% $< 20\%$ gate; spatial decisions gated |
| `visit_06` | SUCCESS | 25.4% (18) | **SUCCESS** | `visit_01` | Canonical Direct | Inlier ratio 25.4% $\ge 20\%$ gate |
| `visit_07` | SUCCESS | 27.5% (19) | **SUCCESS** | `visit_01` | Canonical Direct | Inlier ratio 27.5% $\ge 20\%$ gate |

*Note: In the physical sequence, camera motion stayed within direct line-of-sight of Visit 1. Visit 5 was degraded by severe lighting / blur; existing safeguards gated Visit 5 cleanly in both baseline and Phase 6.*

### 2.3 Uncertainty-Aware Movement Threshold Benchmark

Evaluated coordinate displacement under variable registration noise with a base threshold $T_{base} = 35.0\text{px}$:

| Scenario | $\epsilon_{reproj}$ | Inlier Ratio | Spatial Cov. | Dynamic Margin $\Delta U$ | Effective $T_{eff}$ | Tested Disp | Fixed Assertion | Adaptive Assertion | Result |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **High Precision Aligned** | 0.8 px | 75.0% | 65.0% | **0.0 px** | 35.0 px | 42.0 px | MOVED | MOVED | Genuine Movement Verified |
| **Moderate Camera Jitter** | 2.8 px | 40.0% | 35.0% | **15.1 px** | 50.1 px | 42.0 px | **MOVED (False)** | **PRESENT (Correct)** | **False Movement Prevented** |
| **High Distortion Clustered** | 3.6 px | 25.0% | 15.0% | **30.7 px** | 65.7 px | 42.0 px | **MOVED (False)** | **PRESENT (Correct)** | **False Movement Prevented** |

---

## 3. Regression Test Coverage

The new test suite [`tests/test_keyframe_registration.py`](tests/test_keyframe_registration.py) covers all 12 specified conditions:

| # | Test Case | Target Requirement | Status |
| :--- | :--- | :--- | :--- |
| 1 | `test_01_registration_direct_to_canonical_stable` | Direct canonical alignment under stable camera | **PASS** |
| 2 | `test_02_registration_fallback_to_secondary_keyframe` | Automatic fallback to secondary keyframe under zero overlap with canonical | **PASS** |
| 3 | `test_03_registration_fails_cleanly_all_keyframes` | Clean failure and gating when all keyframes fail | **PASS** |
| 4 | `test_04_scale_resolution_change_registers_accurately`| Accurate mapping across resolution changes (0.75x) | **PASS** |
| 5 | `test_05_large_perspective_shift_chained_keyframe` | Perspective shifts registered through chained keyframes | **PASS** |
| 6 | `test_06_camera_movement_alone_no_false_confirmed_moved`| Camera motion alone does not trigger false movement | **PASS** |
| 7 | `test_07_physical_object_movement_detected_under_camera_motion`| Physical object relocation detected under camera motion | **PASS** |
| 8 | `test_08_degenerate_or_non_convex_homography_rejected` | Folded/non-convex compound homographies rejected | **PASS** |
| 9 | `test_09_outlier_match_clusters_spatial_coverage` | Clustered matches yield low coverage, scaling uncertainty | **PASS** |
| 10 | `test_10_keyframe_promotion_criteria` | Meaningful framing shifts promoted; redundant views rejected | **PASS** |
| 11 | `test_11_canonical_coordinates_consistent_multi_hop` | Compound homography preserves canonical coordinate consistency | **PASS** |
| 12 | `test_12_gated_registration_prevents_false_confirmed_removed`| Gated registrations prevent false removal transitions | **PASS** |

---

## 4. Architectural Verification Summary

1. **Initial Canonical Invariance**:
   The initial reference frame established on Visit 1 remains the permanent root of the coordinate space. Secondary keyframes maintain compound mappings $H_{K \to C}$, never mutating canonical bounds or historical coordinates.
2. **Compound Homography Safeguards**:
   All compound matrices are validated with `is_convex_quadrilateral` and checked for finite entries. Degenerate warps are rejected with `IMPLAUSIBLE_WARP_GEOMETRY`.
3. **UI Diagnostics Transparency**:
   The frontend displays selected keyframe, direct vs. chained compound mode, spatial feature coverage, inlier ratio, mean reprojection error, decision status (`ACCEPTED`, `DEFERRED`, `REJECTED`), and explanatory diagnostics.
