# ETH ASL Change Detection (Lounge) — Honest ObjectMemory Evaluation Protocol

**Document ID:** `OM-ETH-ASL-PROTOCOL-01`  
**Date:** November 25, 2026  
**Dataset:** ETH ASL Change Detection — `lounge` (`10` multi-week observations, `observation_01`–`observation_10`)

---

## 1. Ground-Truth Availability & Separation

| Category | What Exists in ETH ASL `lounge` | Supported? |
| :--- | :--- | :---: |
| **Directly Provided Dataset GT** | 10 aligned 3D TSDF `.ply` meshes (`observation_0.ply`–`9.ply`), 6-DOF camera transforms (`T_G_C`, `T_G_D`), `tf` trees, and 3D point clouds (`point_cloud_D`, `point_cloud_G`). | **Yes** |
| **2D Object Bounding Boxes** | None provided natively. | **No** (Unavailable) |
| **Persistent 2D Object IDs** | None provided natively. | **No** (Unavailable) |
| **ObjectMemory Epistemic Lifecycle Labels** | None provided natively (`CONFIRMED_MOVED`, `PENDING_REMOVAL`, `CONFIRMED_REMOVED`). | **No** (Unavailable) |

**Strict Rule:** Never fabricate 2D bounding boxes or treat YOLOv8n detections as ground truth. All object-lifecycle metrics (`FRR`, `FDP`, `Removal Recall`, `Movement P/R`, `Identity Consistency`, `State Accuracy`) are reported strictly as **`UNAVAILABLE (No 2D Object Lifecycle GT)`** for ETH ASL Lounge.

---

## 2. Supported Pipeline Evaluation Scope

Because ETH ASL Lounge consists of 10 multi-week hand-held 3D room scans (`1280×720` RGB PNGs, `50–100%` overlap), we evaluate the **frozen ObjectMemory pipeline** (`B0` detector + `B3` PlanarRegistrar + TemporalMemoryEngine with unchanged thresholds) strictly on what the dataset legitimately supports:

1. **Cross-Observation Registration & Geometry Gate Evaluation:**
   - Sample representative RGB keyframes across `observation_01`–`observation_10` (e.g., `1` canonical keyframe per observation anchored to `observation_01`, plus intra-observation frame pairs `t -> t+1` within `observation_01`).
   - Measure: registration success rate, failure status distribution (`SUCCESS`, `SINGULAR_HOMOGRAPHY`, `INSUFFICIENT_INLIERS`, `INSUFFICIENT_OVERLAP`, `EXCESSIVE_REPROJECTION_ERROR`), mean inlier ratio, and mean reprojection error.
2. **Closed-Set Detector Behaviour (`OpenCVDNNDetector` YOLOv8n, `conf=0.35`):**
   - Measure per-observation detection counts and detected COCO class distribution (e.g., `chair`, `couch`, `dining table`, `laptop`, `cup`, `bottle`).
3. **Evidence-Gated Temporal Memory Behaviour:**
   - Verify that when planar registration fails across multi-week 3D viewpoint shifts, `TemporalMemoryEngine` **safely freezes** entity beliefs (`INSUFFICIENT_EVIDENCE` / absence freezing) rather than hallucinating false removals.
4. **Qualitative Visual Inspection:**
   - Save registration overlay / side-by-side keyframe comparisons and failure examples under `research_validation/results/eth_asl_lounge/` clearly marked as **qualitative observations**, separate from measured registration/detection counts.
