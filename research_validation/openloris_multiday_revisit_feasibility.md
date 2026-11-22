# OpenLORIS-Scene Multi-Day Revisit Feasibility Study

**Document Version:** 1.0.0  
**Audit Date:** November 7, 2026  
**Auditor:** Antigravity Autonomous Research Suite  
**Project:** ObjectMemory (Long-Term Visual Memory for Detecting Persistent Environmental Changes)  
**Workspace:** `C:\Users\Rishh\.gemini\antigravity\scratch\objectmemory`  
**Output Target:** `research_validation/openloris_multiday_revisit_feasibility.md`

---

## 1. Executive Summary & Final Recommendation

### Final Recommendation: `DO NOT ACQUIRE YET`

| Strategic Dimension | Decision Rationale |
|---|---|
| **Pilot Completeness** | The extracted pilot (`office1-1`, `office1-3`, `office1-2`: 2,068 frames, 1.08 GB) **fully satisfies** all requirements for visual detector benchmarking, 6-DoF OptiTrack registration stress-testing, and registration-failure fallback evaluation. |
| **Absence of Change Ground Truth** | OpenLORIS-Scene was recorded as a SLAM dataset. Even across the later multi-day sequences (`office1-4` through `office1-7`), the dataset provides **zero 2D/3D object bounding boxes, zero persistent entity IDs, and zero change transition timestamps**. Acquiring 4,680 additional frames will not yield quantitative change detection metrics without an external manual annotation campaign. |
| **Acquisition Overhead** | Downloading the remaining 4 sequences requires re-downloading the entire 9.89 GB TAR archive (`package/office1-1_7-package.tar`) to extract ~3.25 GB of unannotated imagery, consuming bandwidth and storage without altering the theoretical evaluation baseline. |
| **Preferred Benchmark Allocation** | Quantitative metrics for ObjectMemory's primary claim (*Registration-Conditioned Evidence-Gated Absence Verification*) are already supported by the **Physical Workbench** (deterministic entity lifecycle ground truth) and **ChangeSim** (pixel-accurate change segmentation masks). |

---

## 2. All-Seven-Sequence Trajectory & Metadata Audit

Traversing the extracted ground-truth archive (`research_validation/data/raw/openloris_scene/groundtruth/extracted/per-sequence/office1-1` through `office1-7`) yields the complete physical trajectory timeline:

### 2.1 Complete Chronological Trajectory Table

| Sequence | Start Time (UTC) | End Time (UTC) | Duration | OptiTrack Poses | Sample Rate | Gaps >10ms | Max Gap | Path Length | Net Disp |
|---|---|---|---|---|---|---|---|---|---|
| **`office1-1`** | 2019-06-08 13:21:23.92 | 2019-06-08 13:21:50.92 | 27.00 s | 6,466 | 240.0 Hz | 3 | 16.7 ms | 8.42 m | 3.63 m |
| **`office1-3`** | 2019-06-08 13:21:50.92 | 2019-06-08 13:22:02.92 | 12.00 s | 2,878 | 240.0 Hz | 1 | 12.5 ms | 1.34 m | 0.20 m |
| **`office1-2`** | 2019-06-08 13:22:02.92 | 2019-06-08 13:22:32.92 | 30.00 s | 6,840 | 240.0 Hz | 26 | 270.8 ms | 9.79 m | 3.80 m |
| **`office1-4`** | 2019-06-09 03:44:45.00 | 2019-06-09 03:45:14.00 | 29.00 s | 6,904 | 240.0 Hz | 1 | 220.8 ms | 10.91 m | 3.84 m |
| **`office1-5`** | 2019-06-09 21:46:48.87 | 2019-06-09 21:47:41.87 | 53.00 s | 12,580 | 240.0 Hz | 7 | 283.3 ms | 17.96 m | 0.14 m |
| **`office1-6`** | 2019-06-11 09:11:26.27 | 2019-06-11 09:12:02.27 | 36.00 s | 8,567 | 240.0 Hz | 12 | 45.8 ms | 14.81 m | 3.66 m |
| **`office1-7`** | 2019-06-11 09:22:37.13 | 2019-06-11 09:23:15.13 | 38.00 s | 9,113 | 240.0 Hz | 0 | 8.3 ms | 13.35 m | 3.95 m |
| **Total** | **4 Calendar Days** | **June 8 – June 11** | **225.00 s** | **53,348** | **240.0 Hz** | **50** | **283.3 ms** | **76.58 m** | **—** |

### 2.2 Spatial Alignment Across Sessions
Across all seven sequences, the robot moves within the **exact same physical OptiTrack bounding box**:
- **$X$-axis:** $[-2.370\text{ m}, +1.682\text{ m}]$ (span: $4.05\text{ m}$, the length of the central office aisle).
- **$Y$-axis:** $[-0.980\text{ m}, +1.194\text{ m}]$ (span: $2.17\text{ m}$, the width between workstation desk banks).
- **$Z$-axis:** $[-1.096\text{ m}, -1.073\text{ m}]$ (span: $0.023\text{ m}$, planar floor level).

This confirms that the OptiTrack coordinate system remained rigidly fixed and calibrated throughout the entire 4-day experiment.

---

## 3. Recording Session Grouping & Temporal Relationships

```
+--------------------------------------------------------------------------------------------------------+
|                                  OPENLORIS-SCENE OFFICE TIMELINE                                       |
+--------------------------------------------------------------------------------------------------------+
|  Day 1: June 8, 2019                                                                                   |
|  13:21:23 UTC ─── [office1-1] ──(33ms)──> [office1-3] ──(34ms)──> [office1-2] (Session 1: 69s total)   |
|                                                                                                        |
|  Day 2: June 9, 2019                                                                                   |
|  03:44:45 UTC (+14.4h) ─── [office1-4] (Session 2: Morning, 29s)                                       |
|  21:46:48 UTC (+18.0h) ─── [office1-5] (Session 3: Night / Dark, 53s)                                  |
|                                                                                                        |
|  Day 4: June 11, 2019                                                                                  |
|  09:11:26 UTC (+35.4h) ─── [office1-6] (Session 4: Daytime, 36s)                                       |
|  09:22:37 UTC (+11.2m) ─── [office1-7] (Session 5: Daytime, 38s)                                       |
+--------------------------------------------------------------------------------------------------------+
```

### 3.1 Five Distinct Recording Sessions

1. **Session 1 (June 8, 13:21 UTC — Extracted Pilot):**
   - Comprises `office1-1` (forward pass), `office1-3` (turnaround), and `office1-2` (return pass).
   - Continuous 68.95-second traversal through a static environment. Zero object moves.
2. **Session 2 (June 9, 03:44 UTC — `office1-4`):**
   - Recorded 14.4 hours after Session 1 in early morning light.
   - Reverse traversal along the aisle ($x = +1.60\text{ m} \to -2.20\text{ m}$).
3. **Session 3 (June 9, 21:46 UTC — `office1-5`):**
   - Recorded 18.0 hours after Session 2 (nighttime).
   - Full circuit around the room ($17.96\text{ m}$ path length, returning to start point).
   - Extreme illumination shift: darkness outside windows, artificial overhead lighting.
4. **Session 4 (June 11, 09:11 UTC — `office1-6`):**
   - Recorded 35.4 hours (~1.5 days) after Session 3.
   - Forward traversal ($x = -1.95\text{ m} \to +1.68\text{ m}$), matching the exact viewpoint direction of `office1-1`.
5. **Session 5 (June 11, 09:22 UTC — `office1-7`):**
   - Recorded 11.2 minutes after Session 4.
   - Return traversal ($x = +1.66\text{ m} \to -2.10\text{ m}$).

---

## 4. Assessment of Remaining Archive Requirements

From `office_tar_inventory.json`, the specifications of the four unacquired archives are:

| Archive Member | Compressed Size | Est. RGB Frames | Est. Extracted RGB | Modalities Present | Ground Truth Pose Status |
|---|---|---|---|---|---|
| `office1-4.7z` | 1.079 GB (1,158,989,665 B) | ~870 frames | ~600 MB | Color, Depth, Fisheye, IMU, Odom | 6,904 OptiTrack poses (Verified) |
| `office1-5.7z` | 2.571 GB (2,760,991,340 B) | ~1,590 frames | ~1,100 MB | Color, Depth, Fisheye, IMU, Odom | 12,580 OptiTrack poses (Verified) |
| `office1-6.7z` | 1.626 GB (1,745,372,686 B) | ~1,080 frames | ~750 MB | Color, Depth, Fisheye, IMU, Odom | 8,567 OptiTrack poses (Verified) |
| `office1-7.7z` | 1.575 GB (1,691,063,390 B) | ~1,140 frames | ~800 MB | Color, Depth, Fisheye, IMU, Odom | 9,113 OptiTrack poses (Verified) |
| **Total Unacquired** | **6.851 GB compressed** | **~4,680 frames** | **~3.25 GB** | **Complete Sensor Suite** | **37,164 OptiTrack poses** |

### Container Access Constraint
Hugging Face does **not** provide direct URLs for individual `.7z` members. Acquiring `office1-4` through `office1-7` would require downloading `office1-1_7-package.tar` (9.89 GB) over the network, unpacking 6.85 GB of `.7z` files, and decompressing ~3.25 GB of image files.

---

## 5. Visual Comparison Feasibility

### 5.1 Workspace Overlap
Because the robot traverses the same 4-meter aisle across all sessions, the camera repeatedly observes the exact same physical workstations:
- **Primary Observable Entities:** Office desks, monitors, office chairs, keyboard trays, cups, backpacks, and wastebaskets.
- **Direct Path Counterparts:**
  - `office1-1` and `office1-6` travel in the exact same forward direction along the $+X$ axis ($x = -1.95 \to +1.68\text{ m}$), observing identical desks from identical viewing angles separated by 3 calendar days (June 8 vs June 11).
  - `office1-2`, `office1-4`, and `office1-7` travel in the reverse direction ($-X$), providing three separate temporal revisits of the opposing aisle view.

### 5.2 Detector Verification
Running ObjectMemory's integrated `OpenCVDNNDetector` (YOLOv8n) on the extracted pilot confirmed consistent detection of office entities:
- `chair` ($0.49 \to 0.79$)
- `tv` / monitor ($0.40 \to 0.76$)
- `laptop` ($0.33$)
- `cup` ($0.32 \to 0.39$)
- `backpack` ($0.61$)
- `suitcase` ($0.73$)

---

## 6. Research Suitability Assessment

| Research Evaluation Category | Evaluation Using Pilot (`1-1`, `1-3`, `1-2`) | Evaluation Using Full 7 Sequences (`1-1` to `1-7`) | Annotation Requirement |
|---|---|---|---|
| **A. Short-Term Object Tracking** | **Sufficient** (2,068 frames over 69s) | Redundant (4,680 more frames) | None for raw persistence; keyframes for F1. |
| **B. Cross-Session Identity Persistence** | **Insufficient** (Single continuous run) | **Feasible** (`1-1` vs `1-6` across 3 days) | **Requires manual keyframe entity association**. |
| **C. Environmental Change Detection** | **Insufficient** (Static environment) | **Incomplete** (Objects moved physically, but zero GT labels) | **Requires manual ground-truth annotation**. |
| **D. Object Movement & Removal Metrics** | **Insufficient** (No moves occurred) | **Insufficient** (No labels provided in OpenLORIS) | **Requires full state-transition manifest**. |
| **E. Registration Under Viewpoint & Light** | **Sufficient** (3.87m path + 180° turn) | **Enhanced** (Day vs Night lighting in `1-5`) | None (OptiTrack 6-DoF serves as GT). |

### The Essential Trade-Off
While sequences `office1-4` through `office1-7` capture genuine multi-day temporal revisits and day/night illumination shifts, **they do not provide object-level ground truth**. Evaluating object movement, removal, or absence metrics on these sequences would require an independent human annotation project to create bounding boxes and identity links.

In contrast:
1. **Physical Workbench** already contains complete ground truth for 42 evaluated entity-visit instances across controlled visits with explicit move/remove transitions.
2. **ChangeSim** provides synthetic pixel-level change segmentation masks with perfect 6-DoF pose and depth truth.

---

## 7. Strategic Action Plan

1. **Retain Current Pilot:** Maintain sequences `office1-1`, `office1-3`, and `office1-2` (2,068 frames, 1.08 GB) as the primary real-world SLAM, registration-failure, and detector robustness benchmark.
2. **Defer Remaining Sequences:** Do not download the remaining 9.89 GB TAR or extract `office1-4` through `office1-7` at this stage.
3. **Execute Comparative Validation:** Proceed with evaluating ObjectMemory's temporal memory engine against:
   - **Physical Workbench:** For end-to-end change detection metrics (Movement/Removal Precision, Recall, F1).
   - **OpenLORIS Pilot:** For geometric registration accuracy, tracking dropout tolerance, and `UNREGISTERED` detection fallback.
   - **ChangeSim:** For controlled simulation baselines with pixel-level ground truth.
