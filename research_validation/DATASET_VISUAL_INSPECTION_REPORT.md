# ObjectMemory — Visual Dataset Inspection Report

**Document ID:** `OM-VISUAL-INSPECT-01`  
**Date:** November 25, 2026  
**Gallery Location:** `research_validation/dataset_visual_inspection/index.html`  
**Purpose:** Summarize what is visually observable across all local and candidate external datasets to support a human dataset-selection decision (without automatically selecting or rejecting datasets).

---

## 1. Visual Summary by Dataset

### Section A — Synthetic Workbench v2 (`25` images, `5` sequences × `5` visits)
- **What is visible:** `1080×720` RGB PNG scenes showing a procedural oak wood-grain tabletop with 3 vertical plank seams and 4 `DICT_4X4_50` corner ArUco fiducials.
- **Objects & Assets:** 5 segmented photographic cutouts (`cup_01`, `bottle_01`, `book_01`, `scissors_01`, `plant_01`) composited with Gaussian cast shadows and contact shading. **Not genuine camera photography.**
- **Demonstrated Changes:**
  - `S01_stationary`: All 5 objects remain fixed across 5 visits.
  - `S02_relocation_return`: `cup_01` moves ~198 px (v2–v4) and returns to baseline (v5); `bottle_01` moves ~230 px (v4–v5).
  - `S03_removal_confirmation`: `bottle_01` disappears (v2–v5); `scissors_01` disappears (v4–v5).
  - `S04_occlusion_recovery`: A synthetic flat-colored manila rectangle partially occludes `book_01` (v2) and fully covers `scissors_01` (v3), then is removed (v4–v5).
  - `S05_lighting_viewpoint`: Object brightness scales (`0.65×` in v2, `1.25×` in v4) while tabletop background stays constant; whole-canvas projective warp (`+2.5°` in v3, `-1.8°` in v5).
- **Annotations:** Full 2D `[x_min, y_min, x_max, y_max]` bounding boxes, persistent `object_id`s, and epistemic states (`event_id="return_to_baseline"` omitted in `S02` v5).

### Section B — Desk Pilot 01 (`7` images, `data/real_workbench/`)
- **Provenance Label:** **`EXPLORATORY PILOT — PROVENANCE UNVERIFIED`** (7 cropped RGBA PNGs across two resolution groups: `780×636`/`784×636` for v1–v3 and `584×576`–`592×576` for v4–v7, `0` EXIF metadata).
- **What is visible:** Overhead desk surface with a potted plant (`plant_01`), white mug (`cup_01`), second white cup (`cup_02`, v1 only), clear water bottle (`bottle_01`), red-handled scissors (`scissors_01`), navy book (`book_01`), kraft notebook (`notebook_01`, v4 only), and an excluded partial laptop at the right edge.
- **Demonstrated Changes:** `cup_02` disappears after v1; `cup_01` relocates (v3) and returns (v4); `scissors_01` shifts/disappears in v4 and reappears in v5; `bottle_01` disappears in v5–v6 and reappears in v7; `notebook_01` appears only in v4.
- **Annotations & Missing Elements:** 2D bboxes, persistent IDs, and lifecycle states exist, but camera calibration/EXIF is absent, resolution/FOV jumps between v3 and v4, and no removal stays absent long enough for streak-2 confirmation (`0/3` events confirmed).

### Section C — OpenLORIS-Scene (`10` evaluated keyframes; `2,068` extracted frames)
- **What is visible:** `848×480` RGB PNG frames from an Intel RealSense D435i on an AGV traversing an indoor office (`office1-1`: `809` frames, `office1-2`: `899` frames, `office1-3`: `360` frames). The 10 evaluated keyframes (`frame_index` `0` to `808`) show large 3D camera translation/rotation across desks, chairs, monitors, and office partitions.
- **Annotations & Missing Elements:** 240 Hz OptiTrack 6-DOF pose trajectory (`groundtruth.txt`) is available; **no 2D object bounding boxes, no persistent object IDs, and no object-lifecycle ground truth exist**.

### Section D — ETH ASL Change Detection (`OFFICIAL EXTERNAL PREVIEW — NOT LOCALLY DOWNLOADED DATA`)
- **What is visible in official previews:** Three indoor RGB-D environments (`office`: 4 tripod scans; `living_room`: 9 hand-held scans; `lounge`: 10 hand-held scans over 2 weeks) with dense TSDF 3D reconstructions and 3D scene-change highlights.
- **Missing:** No local files downloaded; ROS `.bag` extraction is required for RGB frames, and 2D bounding boxes / persistent object IDs are absent.

### Section E — ChangeSim (`OPTIONAL CANDIDATE` — `OFFICIAL EXTERNAL PREVIEW — NOT LOCALLY DOWNLOADED DATA`)
- **What is visible in official preview:** Photo-realistic Unreal Engine 4 industrial indoor simulation showing lighting/turbidity changes and semantic/change segmentation masks. Full dataset audit has not been completed.

### Section F — Empty and Missing Data
- **`physical_pilot_v2` (`data/physical_pilot_v2/`):** `0` images (`TEMPLATE_AWAITING_PHYSICAL_DATA`).
- **`physical_workbench_v1` (`data/physical_workbench_v1/`):** `0` images (`AWAITING_PHYSICAL_DATA`).

---

## 2. Objective-to-Dataset Suitability Matrix (For Human Decision)

| ObjectMemory Objective | `synthetic_workbench_v2` | `desk_pilot_01` | `OpenLORIS-Scene` | `ETH ASL` *(if extracted/annotated)* | `ChangeSim` *(optional, unaudited)* | `physical_pilot_v2` *(once captured)* |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Long-term visual memory & event history** | High (controlled) | Partial (short 7-visit seq) | **Not suitable** (no object GT) | Only after 2D annotation | Pending full audit | High (target) |
| **Persistent object location changes** | High | Partial (`1/2` moves) | **Not suitable** | Only after 2D annotation | Pending full audit | High (target) |
| **Filtering lighting variations** | Partial (object-only scaling) | Uncontrolled | No GT | Uncontrolled indoor light | Yes (in sim) | High (target V03) |
| **Filtering occlusion** | Partial (flat synthetic box) | Partial (v4) | No GT | Dynamic clutter (3D) | Pending full audit | High (target V07–V08) |
| **Handling viewpoint / registration** | Planar warp (`±2.5°`) | Uncalibrated crop jump | **3D parallax failure boundary** | High-overlap (`office`/`living_room`) | Yes (6-DOF poses) | Planar tilt (`~5°`, V04) |

---

## 3. What Remains Uncertain
1. **Unverified Camera Provenance in `desk_pilot_01`:** Original uncropped photographs and EXIF metadata are unavailable.
2. **Real-World Multi-Visit Removal & Occlusion Behavior:** Neither `desk_pilot_01` (too short) nor `OpenLORIS` (no object GT) validates streak-2 real-world removal confirmation.
3. **ETH ASL & ChangeSim 2D Object-ID Effort:** Whether manual/semi-automatic 2D bounding-box annotation on ETH ASL `office`/`living_room` or a full audit of `ChangeSim` is worth the effort compared to capturing `physical_pilot_v2` and expanding `synthetic_workbench_v2`.
