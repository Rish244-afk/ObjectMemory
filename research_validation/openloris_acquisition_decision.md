# OpenLORIS Office Archive Acquisition Decision & Critical Feasibility Audit

**Document Version:** 1.0.0  
**Audit Date:** November 7, 2026  
**Auditor:** Antigravity Autonomous Research Suite  
**Project:** ObjectMemory (Long-Term Visual Memory for Detecting Persistent Environmental Changes)  
**Workspace:** `C:\Users\Rishh\.gemini\antigravity\scratch\objectmemory`  
**Target File:** `research_validation/openloris_acquisition_decision.md`

---

## 1. Executive Summary & Final Acquisition Decision

### Final Acquisition Decision: `DO NOT ACQUIRE UNTIL ANNOTATION PROTOCOL IS FORMALIZED`
*(Targeted Fallback: If and when multi-day real-world benchmarking is scheduled, acquire **ONLY `office1-6.7z`** via HTTP Range).*

| Decision Dimension | Finding & Strategic Implication |
|---|---|
| **Contradiction Resolution** | The initial feasibility claim that the extracted pilot (`office1-1`, `office1-3`, `office1-2`) is "sufficient for all experimental goals" is **retracted**. The pilot is strictly sufficient for **short-term registration, tracking dropout gating, and single-session detector stability**, but is **completely insufficient for cross-session identity persistence and multi-day environmental change detection**. |
| **Cross-Session Spatial Overlap** | Mathematical analysis of 6-DoF camera poses reveals that **`office1-6` is an extraordinary 3-day forward revisit of `office1-1`** (median distance: $13.2\text{ cm}$, median viewing angle offset: $4.9^\circ$, with **100.0% moderate overlap** and **68.5% strict sub-20cm/sub-15° overlap**). In contrast, `office1-4` and `office1-7` duplicate the reverse aisle view, and `office1-5` introduces non-planar loop maneuvers. |
| **Selective HTTP Range Provenance** | We have verified experimentally that Hugging Face supports **HTTP 206 Partial Content**. Remote scanning of TAR headers proved that individual `.7z` members can be acquired independently without downloading the 9.89 GB TAR container. `office1-6.7z` can be downloaded alone as a single $1.626\text{ GB}$ stream (`bytes 7181891072–8927263757`). |
| **Annotation Bottleneck** | OpenLORIS-Scene provides zero object-level ground truth. Acquiring `office1-6` without an immediate commitment to manually annotate ~63 keyframes (~378 bounding boxes and cross-session entity IDs) would leave the images unutilized for quantitative change metrics. |
| **Immediate Priority** | Quantitative benchmarks for ObjectMemory's primary claim (*Registration-Conditioned Evidence-Gated Absence Verification*) must proceed on the **Physical Workbench** (deterministic move/remove truth) and **ChangeSim** (pixel-accurate change segmentation masks). |

---

## 2. Correction of Unsupported Feasibility Claims

The Phase 1D report contained an internal contradiction between its executive summary and its research suitability matrix. The table below documents the verified facts versus the retracted claims:

| Research Dimension | Initial Feasibility Claim (Section 1) | Suitability Matrix Assessment (Section 6) | Corrected Scientific Reality |
|---|---|---|---|
| **Short-Term Registration & VO** | Claimed sufficient. | Sufficient. | **VERIFIED SUFFICIENT.** 2,068 frames over 69s with 240 Hz OptiTrack ground truth provide rigorous validation of camera odometry and homography error bounds. |
| **Tracking Dropout Fallback** | Claimed sufficient. | Sufficient. | **VERIFIED SUFFICIENT.** 26 OptiTrack dropout intervals in `office1-2` (including the 270 ms marker occlusion at frame 97) validate ObjectMemory's `UNREGISTERED` fallback mechanism. |
| **Cross-Session Identity Persistence** | Claimed pilot satisfies requirements. | Evaluated as Insufficient. | **RETRACTED AS INSUFFICIENT ON PILOT.** Forensics prove `1-1`, `1-3`, and `1-2` are three consecutive chops of a single 69-second static run. Zero cross-session time elapsed. |
| **Environmental Change Detection** | Conflated with detector robustness. | Evaluated as Insufficient. | **RETRACTED AS INSUFFICIENT ON PILOT.** No physical objects moved or disappeared in the room during the 69-second traversal. Cannot benchmark movement or removal metrics. |
| **Multi-Day Generalization** | Claimed complete for evaluation goals. | Requires later sequences. | **CORRECTED.** Multi-day evaluation requires comparing Session 1 against Session 4 (`office1-6`, 3 days later) or the Physical Workbench. |

---

## 3. Trajectory & Viewpoint Overlap Forensics

Using the verified rigid transformation chain:
$$T_{W \leftarrow C} = T_{W \leftarrow M} \cdot \left(T_{B \leftarrow M}\right)^{-1} \cdot T_{B \leftarrow C}$$
Camera optical origins $\mathbf{p}_{\text{cam}, W}$ and forward viewing unit vectors $\mathbf{v}_{\text{opt}, W} = R_{W \leftarrow C} \cdot [0, 0, 1]^T$ were calculated across all 53,348 OptiTrack poses. Pairwise comparisons were evaluated at 48 Hz subsampling:

### 3.1 Empirical Cross-Session Overlap Metrics

| Sequence Pair | Semantic Traversal Relationship | Nearest Distance (Median / Mean) | Optical Angle Offset (Median / Mean) | Strict Overlap ($\le 20\text{cm}, \le 15^\circ$) | Moderate Overlap ($\le 50\text{cm}, \le 30^\circ$) |
|---|---|---|---|---|---|
| **`office1-1` vs `office1-6`** | **Forward Aisle Pass (Day 1 vs Day 4, +3 days)** | **$0.132\text{ m}$ / $0.124\text{ m}$** | **$4.9^\circ$ / $6.9^\circ$** | **68.5%** (887/1,294 poses) | **100.0%** (1,294/1,294 poses) |
| **`office1-2` vs `office1-4`** | **Reverse Aisle Pass (Day 1 vs Day 2, +14.4h)** | **$0.051\text{ m}$ / $0.059\text{ m}$** | **$6.2^\circ$ / $10.9^\circ$** | **79.9%** (1,093/1,368 poses) | **91.7%** (1,254/1,368 poses) |
| **`office1-2` vs `office1-7`** | **Reverse Aisle Pass (Day 1 vs Day 4, +3 days)** | **$0.145\text{ m}$ / $0.170\text{ m}$** | **$9.0^\circ$ / $10.6^\circ$** | **45.5%** (622/1,368 poses) | **96.8%** (1,324/1,368 poses) |
| **`office1-1` vs `office1-5`** | **Forward Pass vs Night Loop (+32.4h)** | **$0.075\text{ m}$ / $0.134\text{ m}$** | **$24.8^\circ$ / $77.5^\circ$** | **47.0%** (608/1,294 poses) | **51.2%** (662/1,294 poses) |
| **`office1-4` vs `office1-7`** | Reverse Pass (Day 2 vs Day 4) | $0.264\text{ m}$ / $0.269\text{ m}$ | $12.3^\circ$ / $35.1^\circ$ | 20.6% (284/1,381 poses) | 70.2% (969/1,381 poses) |

### 3.2 Key Scientific Takeaways
1. **The `office1-1` $\leftrightarrow$ `office1-6` Pairing is Exceptional:** Both sequences traverse from $x = -1.95\text{ m}$ to $+1.68\text{ m}$ along the central aisle looking forward. The median camera position difference across the entire room is only **$13.2\text{ cm}$**, and the median viewing angle deviation is only **$4.9^\circ$**. Every single camera pose in `office1-1` (100.0%) finds a matching camera viewpoint in `office1-6` within $50\text{ cm}$ and $30^\circ$.
2. **Reverse Passes Provide High Overlap but Redundant Views:** `office1-2`, `office1-4`, and `office1-7` all retrace the reverse aisle pass with 79.9% to 96.8% overlap. Acquiring all three provides no substantial new visual geometry beyond `office1-2`.
3. **`office1-5` Captures Illumination Shifts with High Angular Divergence:** While `office1-5` shares 47% forward aisle overlap with `office1-1` under night illumination, its loop path causes an overall mean angular divergence of $77.5^\circ$.

---

## 4. Detector Execution Audit & Forensic Verification

The Phase 1D report stated that YOLOv8n "confirmed consistent detection of office entities across the dataset." An audit of the actual execution trace reveals:

### 4.1 Execution Log Audit
1. **Failures 1–4:** Initial script runs failed due to workspace module import paths (`PYTHONPATH`), incorrect module naming (`detector.py` vs `yolo.py`), invalid configuration arguments (`model_path`), and non-existent attribute names (`class_name` vs `label`).
2. **Execution 5 (Successful):**
   ```python
   detector = OpenCVDNNDetector(weights_path='models/yolov8n.onnx')
   ```
   Completed successfully across 3 sample frames per sequence:
   - `office1-1` (3 frames): `chair` ($0.49 \to 0.78$), `tv` ($0.40$), `laptop` ($0.33$), `suitcase` ($0.27$).
   - `office1-3` (3 frames): `chair` ($0.55 \to 0.79$), `backpack` ($0.61$), `cup` ($0.39$), `mouse` ($0.30$).
   - `office1-2` (3 frames): `chair` ($0.29 \to 0.78$), `tv` ($0.31 \to 0.76$), `suitcase` ($0.73$), `cup` ($0.32$).

### 4.2 Qualification of Evidence
- **Total Frames Evaluated:** Exactly **9 frames** (out of 2,068 frames, or $0.44\%$ of the pilot).
- **Status:** The reported detections and confidence scores are authentic outputs from the local `models/yolov8n.onnx` model.
- **Correction:** Describing this 9-frame spot check as "confirming consistent detection across the dataset" was an overstatement. Full temporal detector persistence (e.g. tracking stability through motion blur) remains to be systematically benchmarked.

---

## 5. Storage & Remote Acquisition Feasibility

### 5.1 Remote TAR Member Scanning via HTTP Range
The assumption that individual `.7z` members cannot be acquired without downloading the full 9.89 GB TAR archive was **empirically disproven**.

Using a range-seeking utility (`research_validation/scripts/scan_remote_tar_offsets.py`), we queried the Hugging Face CDN using `Range: bytes=start-end`. Hugging Face returned `HTTP 206 Partial Content` with `Accept-Ranges: bytes`. By reading 512-byte POSIX TAR headers remotely, we established the exact byte boundaries for every member:

| Member Archive | Uncompressed Size | Remote Byte Range in TAR Container | Selective Download Feasibility |
|---|---|---|---|
| `office1-1.7z` | 1,365,585,538 B (~1.27 GB) | `bytes 512` to `1365586049` | Already Extracted |
| `office1-2.7z` | 1,347,899,302 B (~1.25 GB) | `bytes 1365586944` to `2713486245` | Already Extracted |
| `office1-3.7z` | 548,420,983 B (~0.51 GB) | `bytes 2713486848` to `3261907830` | Already Extracted |
| `office1-4.7z` | 1,158,989,665 B (~1.08 GB) | `bytes 3261908480` to `4420898144` | Verified Addressable |
| `office1-5.7z` | 2,760,991,340 B (~2.57 GB) | `bytes 4420898816` to `7181890155` | Verified Addressable |
| `office1-6.7z` | 1,745,372,686 B (~1.63 GB) | `bytes 7181891072` to `8927263757` | **Target Candidate ($1.626\text{ GB}$)** |
| `office1-7.7z` | 1,691,063,390 B (~1.58 GB) | `bytes 8927264768` to `10618328157` | Verified Addressable |

**Technical Finding:** If the multi-day revisit benchmark is authorized, we do **not** need to download 9.89 GB. We can selectively stream **only bytes 7,181,891,072 to 8,927,263,757 (1.626 GB)** to acquire `office1-6.7z` directly.

---

## 6. Manual Annotation Cost-Benefit Model

Because OpenLORIS-Scene contains no object-level ground truth, acquiring `office1-6` produces actionable metrics only if paired with an external annotation manifest:

### Annotation Protocol Model
- **Keyframe Subsampling:** $1.0\text{ Hz}$ (1 frame per second).
- **Keyframes in `office1-1` (27s):** 27 frames.
- **Keyframes in `office1-6` (36s):** 36 frames.
- **Total Keyframes:** **63 frames**.
- **Entities per Keyframe:** ~6 prominent objects (monitors, chairs, laptops, bags, cups).
- **Bounding Boxes to Label:** $\approx 378$ 2D bounding boxes.
- **Cross-Session Re-ID Links:** ~10 persistent physical entities mapped across Day 1 and Day 4.
- **Estimated Human Annotation Time:** 3 to 5 hours.

### Cost-Benefit Comparison

| Option | Network Transfer | Disk Storage | Annotation Effort | Scientific Yield | Recommendation |
|---|---|---|---|---|---|
| **Option A: Download Full TAR (9.89 GB)** | 9.89 GB | ~15 GB peak | 15+ hours (all 7 sequences) | Low efficiency; redundant reverse passes. | **REJECTED** |
| **Option B: Selective Range Download of `office1-6`** | **1.63 GB** | **~2.4 GB** (7z + RGB) | **3–5 hours** (63 keyframes) | **High.** Creates a definitive real-world multi-day benchmark with sub-millimeter OptiTrack truth. | **RECOMMENDED FOR PHASE 2** |
| **Option C: Retain Current Pilot (Do Not Acquire Yet)** | **0 GB** | **0 GB** (1.08 GB existing) | **0 hours** | **Optimal for Phase 1.** Preserves focus on Physical Workbench and ChangeSim. | **RECOMMENDED FOR PHASE 1** |

---

## 7. Strategic Action Plan & Verdict

### Final Action Plan
1. **Freeze Real-World Acquisition in Phase 1:** Do not download `office1-6` or any additional OpenLORIS archives during the current phase.
2. **Execute Primary Quantitative Validation:**
   - **Physical Workbench (42 evaluated instances):** Benchmark ObjectMemory's core state-transition metrics (Movement Precision/Recall, Removal F1, Absence Confirmation Delay).
   - **OpenLORIS Pilot (`1-1`, `1-3`, `1-2`):** Benchmark visual odometry drift, planar homography limits, and registration-failure gating under camera turns and OptiTrack occlusions.
   - **ChangeSim:** Benchmark pixel-level change segmentation against controlled synthetic ground truth.
3. **Phase 2 Trigger Condition:** If peer review or thesis requirements demand a real-world multi-day robotic evaluation, execute the verified **HTTP Range download of `office1-6.7z` (1.63 GB)** and label the 63-keyframe annotation manifest.
