# ObjectMemory — Read-Only Dataset Audit Report

**Document ID:** `OM-DATASET-AUDIT-01`  
**Date:** November 25, 2026  
**Project Title:** ObjectMemory: Long-Term Visual Memory for Detecting Persistent Environmental Changes  
**Official Description:** *"An AI-based visual memory system that compares scenes over time to detect and explain persistent changes in object locations while filtering out temporary variations such as lighting, occlusion, and viewpoint changes."*  
**Audit Mode:** Strictly read-only inspection against frozen `main` (`a612fab`).

---

## 1. Executive Summary

This audit establishes the inventory, image provenance, generator capabilities, and external dataset compatibility of all ObjectMemory datasets prior to designing an expanded ~90-observation benchmark:
- **`synthetic_workbench_v2`** (`25` RGB PNGs, `5` sequences) is the only complete multi-scenario object-lifecycle benchmark currently runnable end-to-end, but it uses procedurally composited 2D cutouts and has gaps in global lighting modulation, object addition, and `return_to_baseline` event tagging.
- **`desk_pilot_01` (`data/real_workbench/`)** (`7` RGBA PNGs, `1` sequence) contains cropped user-supplied images with **zero EXIF metadata** and two mismatched resolution groups (`780×636` vs `~584×576`). Because genuine camera provenance cannot be independently verified from image headers, it must be classified strictly as **`EXPLORATORY_PILOT` (unverified photographic provenance)** and not cited as a metrological physical benchmark.
- **`physical_pilot_v2`** (`0` images, `10`-visit template) and **`physical_workbench_v1`** (`0` images, `13`-visit protocol) await genuine photographic capture.
- **OpenLORIS-Scene** (`2,068` extracted color PNGs across `office1-1/2/3`, `10`-frame evaluated subset) and **ETH ASL Change Detection** (`office`, `living_room`, `lounge`) lack 2D bounding boxes and persistent object-lifecycle ground truth; they serve as external 3D scene-change / registration-boundary datasets rather than drop-in object-memory benchmarks.

---

## 2. Existing Dataset Inventory

| Dataset | Path | Image Files | Obs | Seqs | Dimensions & Format | Bounding Boxes | Persistent Object IDs | Lifecycle GT | Scenarios Covered | Missing Scenarios |
| :--- | :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :--- | :--- |
| **`synthetic_workbench_v2`** | `data/synthetic_workbench_v2/` | 25 | 25 | 5 | `1080×720` RGB PNG | Yes | Yes | Yes | Stationary (`S01`), relocation (`S02`), removal (`S03`), occlusion (`S04`), lighting/viewpoint (`S05`) | Object addition; explicit `return_to_baseline` event tag; identical-class distractor |
| **`desk_pilot_01`** | `data/real_workbench/` | 7 | 7 | 1 | `780×636`, `784×636`, `584×576`, `588×576`, `592×576` RGBA PNG | Yes | Yes | Yes | Relocation, return, addition, removal onset, partial occlusion | Controlled lighting; controlled camera shift; streak-2 confirmed removals (`0/3` confirmed) |
| **`physical_pilot_v2`** | `data/physical_pilot_v2/` | **0** | 10 (template) | 1 (template) | N/A (`AWAITING_PHYSICAL_DATA`) | Template only | Template only | Template only | Planned: all 8 core scenarios | **All 10 images & pixel bboxes missing** |
| **`physical_workbench_v1`** | `data/physical_workbench_v1/` | **0** | 0 | 0 | N/A (`AWAITING_PHYSICAL_DATA`) | No | No | No | Protocol doc only (`CAPTURE_PROTOCOL.md`) | **All images & manifest missing** |
| **`OpenLORIS-Scene` (`office1-1`)** | `research_validation/data/raw/openloris_scene/` | 2,068 total (`809` in `office1-1`; `10` evaluated) | 10 (eval) | 1 (eval) | `848×480` RGB PNG | **No** | **No** | **No** | Unconstrained 3D AGV traversal | 2D object bboxes, persistent IDs, lifecycle GT |
| **`controlled_temporal_v1`** *(legacy)* | `data/controlled_temporal_v1/` | 30 | 30 | 6 | PNG (legacy Phase 10) | Yes (legacy) | Yes (legacy) | Legacy schema | Superseded by `synthetic_workbench_v2` | Manifest v2.0 state alignment |
| **`coco_val2017_subset`** | `data/coco_val2017_subset/` | 20 | 20 | N/A | JPEG (natural single frames) | COCO static | No | **No** | Single-frame detector precision/recall | Temporal multi-visit sequences |

---

## 3. Image Provenance Audit

| Dataset | Observed Image Metadata | Documented Provenance | Provenance Classification | Audit Conclusion |
| :--- | :--- | :--- | :--- | :--- |
| **`synthetic_workbench_v2`** | `0` EXIF tags, `RGB` PNG, uniform `1080×720` | `SYNTHETIC_GENERATED` (`generate_synthetic_workbench_v2.py`) | **Procedurally composited photographic cutouts** | Verified: 5 segmented RGBA cutouts composited onto procedural sinusoidal oak wood grain with Gaussian shadows and ArUco markers. |
| **`desk_pilot_01` (`real_workbench`)** | `0` EXIF tags, `[]` PNG info keys, `RGBA` PNG, 5 distinct cropped dimensions (`780×636`–`592×576`) | `EXPLORATORY_PILOT` (`manifest_v2.json:8`: *"cropped and saved as RGBA PNG without EXIF metadata"*) | **UNKNOWN / Unverified Photographic Provenance** | **Cannot be verified as genuine unedited camera photographs** from file metadata; retain strictly as `EXPLORATORY_PILOT`, not as a physical metrology benchmark. |
| **`OpenLORIS-Scene`** | `848×480` `RGB` PNG (`1560000083.949196.png`, etc.) | `PUBLIC_BENCHMARK` (Intel RealSense D435i + OptiTrack) | **External research data** | Verified external RGB-D robotic SLAM benchmark. |
| **`physical_pilot_v2`** | No images present | `GENUINE_PHYSICAL` (template) | **Missing** | Must not be claimed complete until uncropped camera images with EXIF/rig provenance are captured. |

---

## 4. Synthetic Generator Capability Audit (`scripts/generate_synthetic_workbench_v2.py`)

Inspected read-only (`lines 1–534`). Actual capabilities verified from code:
- **Image Synthesis:** `generate_wood_tabletop(width=1080, height=720, seed=100*s_idx+v_idx)` generates procedural multi-octave sinusoidal wood grain, Gaussian micro-noise (`np.random.seed(seed)`), 3 vertical plank seams, and 4 `DICT_4X4_50` ArUco corner markers (`40×40` px).
- **Object Assets:** Loads 5 segmented RGBA photographic cutouts (`cup.png`, `bottle.png`, `scissors.png`, `book.png`, `plant.png` in `data/synthetic_workbench_v2/assets/`). Not 3D-rendered meshes or AI-generated graphics.
- **Position & Identity Control:** Deterministic `(bx, by)` pixel placements per `object_id`; persistent IDs (`cup_01`, `bottle_01`, `book_01`, `scissors_01`, `plant_01`) are explicitly maintained in code.
- **Automatic Bounding Boxes & Ground Truth:** `composite_asset()` computes tight axis-aligned `[x_min, y_min, x_max, y_max]` from `alpha > 0.1` automatically, and `warp_viewpoint()` transforms bounding boxes via `cv2.perspectiveTransform`. Ground-truth `manifest_v2.json` is emitted automatically.
- **Determinism & Reproducibility:** Deterministic given `seed` (`np.random.seed(seed)`), fixed asset paths, and fixed placement coordinates; exact environment can be reproduced bit-for-bit.
- **Supported Perturbations:**
  - *Lighting variation:* **Partial** — `brightness_factor` scales **only the pasted object's BGR channels** (`lines 142–143`), leaving the tabletop background and shadow opacity unchanged.
  - *Occlusion:* **Partial** — `apply_occluder()` draws a flat manila BGR rectangle `(120, 160, 205)` with a Gaussian shadow (`lines 174–195`) and sets `bbox = None` for `FULLY_OCCLUDED`.
  - *Camera viewpoint variation:* **Supported** — `warp_viewpoint()` applies 4-point projective homography (`cv2.getPerspectiveTransform` + `cv2.warpPerspective`).
  - *Movement, removal, return:* **Supported** in `S02`/`S03` (note: `S02` return-to-baseline omits `event_id="return_to_baseline"` in emitted JSON).
  - *Object addition:* **Not implemented** in `generate_synthetic_workbench_v2.py` (though `ManifestLoader` accepts `CONFIRMED_ADDED` / `ADDED`).

---

## 5. Proposed 90-Observation Synthetic Benchmark Feasibility Analysis

| Scenario | Target Obs | Support Classification | Dependencies & Implementation Requirements |
| :--- | :---: | :--- | :--- |
| **Stationary baseline** | 10 | **DIRECTLY SUPPORTED** | Already implemented in `S01_stationary`; can be scaled to two 5-visit or one 10-visit sequence. |
| **Lighting variation** | 10 | **PARTIALLY SUPPORTED** | Must extend generator to apply global illumination/gamma/tint or oblique shadow gradients to the **tabletop background + shadows**, not only `brightness_factor` on object pixels. |
| **Object movement** | 15 | **DIRECTLY SUPPORTED** | Already implemented in `S02`; vary displacement magnitudes (e.g., `50–250` px) and ensure resting visits vs active move visits match manifest state semantics. |
| **Object removal** | 15 | **DIRECTLY SUPPORTED** | Already implemented in `S03`; ensure ≥2 consecutive absent visits per removal event so `vacancy_visits_for_removal = 2` is tested. |
| **Object addition** | 10 | **PARTIALLY SUPPORTED** | Generator must omit an object in early visits and composite it in later visits with `CONFIRMED_ADDED` (or `CONFIRMED_PRESENT` + `NEWLY_APPEARED`); requires verifying `BenchmarkRunner` scoring of late-appearing entities. |
| **Temporary occlusion** | 10 | **PARTIALLY SUPPORTED** | `apply_occluder()` currently uses a synthetic flat rectangle; needs varied occlusion ratios (`30%–100%`) and multi-visit occlusion recovery sequences. |
| **Return-to-baseline** | 10 | **PARTIALLY SUPPORTED** | `S02` moves `cup_01` back, but generator must emit `event_id: "return_to_baseline"` so `HRA` is scorable (`runner.py:1148`), and requires `REQUIRES VALIDATION` with frozen `B3` (which currently kept `cup_01` as `CONFIRMED_MOVED` in `S02` v5). |
| **Camera viewpoint variation** | 10 | **DIRECTLY SUPPORTED** | `warp_viewpoint()` + `perspectiveTransform` already handle tilt/translation/scale; test both positive/negative angles within planar bounds (`±5°`). |
| **Total** | **90** | — | — |

### Sequence Structure & Data Leakage Prevention
- **Temporal Sequence Requirement:** 90 isolated single images cannot test temporal state transitions (`PENDING_REMOVAL`, `CONFIRMED_REMOVED`, `CONFIRMED_MOVED`, occlusion freeze). The 90 observations must be organized as **chronological multi-visit sequences** (e.g., 15 sequences × 6 visits or 18 sequences × 5 visits), where each sequence shares a coherent background seed and baseline state.
- **Data Leakage Prevention:**
  1. Keep `synthetic_workbench_v2` (25 obs) frozen and strictly separated from any new 90-observation suite (`synthetic_workbench_v3`).
  2. Never tune thresholds on the 90-observation evaluation split; evaluate frozen `B3` zero-shot (or split into separate calibration vs held-out test sequences by distinct background seeds and object layouts).
  3. Anchor each sequence's canonical frame strictly to its own `visit_01` inside isolated `:memory:` SQLite stores (already enforced by `BenchmarkRunner`).

---

## 6. ETH ASL Change Detection Dataset Compatibility Assessment

Evaluated via official documentation (`https://projects.asl.ethz.ch/datasets/change-detection/` and `https://github.com/ethz-asl/change_detection_ds`; Fehr et al., ICRA 2017):

- **Sequences:**
  - `office`: 4 static tripod RGB-D recordings (~100% overlap).
  - `living_room`: 9 hand-held RGB-D recordings (~100% overlap, dynamic object interactions).
  - `lounge`: 10 hand-held RGB-D recordings over 2 weeks (~50–100% overlap).
- **Available Data:** ROS `.bag` files (`sensor_msgs/Image` RGB, `sensor_msgs/PointCloud2` in camera & global frames, `geometry_msgs/TransformStamped` `T_G_C`/`T_G_D`, `tf/tfMessage`), aligned TSDF dense 3D meshes, and 3D TSDF/mesh-level static-vs-dynamic change ground truth.
- **Missing for ObjectMemory:**
  - No pre-extracted standalone RGB image directories (requires ROS bag extraction, e.g., via `rosbags`).
  - **No 2D object bounding boxes**, **no persistent per-object IDs**, and **no ObjectMemory epistemic lifecycle states** (`CONFIRMED_MOVED`, `PENDING_REMOVAL`, `CONFIRMED_REMOVED`).
- **Compatibility Classification:**
  - **For RGB scene-change / registration testing:** **`USABLE AFTER EXTRACTION`** (must extract RGB PNGs from ROS `.bag` files).
  - **For ObjectMemory object-lifecycle metrics (`FRR`, `RR`, `IC`, Movement P/R):** **`USABLE AFTER ANNOTATION`** (requires manual/semi-automated 2D bounding-box + persistent object-ID + lifecycle-state annotation into `manifest_v2.json` format). *Not directly usable out-of-the-box.*

---

## 7. OpenLORIS Role Assessment vs. ETH ASL

| Aspect | OpenLORIS-Scene (`office1-1`) | ETH ASL Change Detection (`office`, `living_room`, `lounge`) |
| :--- | :--- | :--- |
| **Primary Modality** | Continuous AGV RGB-D video stream + OptiTrack 6-DOF poses | Multi-visit RGB-D scans + TSDF meshes + `T_G_C`/`T_G_D` transforms |
| **Distinct Research Role** | **Camera/scene variation & planar registration failure boundary** (demonstrated `8/9` singular homography collapses under `>1.5` m 3D room translation). | **Multi-visit indoor scene changes & repeated RGB-D observations** (`office` tripod, `living_room` / `lounge` repeated visits with 3D change masks). |
| **Object-Lifecycle GT** | None (`0` bounding boxes, `0` object IDs) | 3D TSDF/mesh change GT only; **no 2D bounding boxes or persistent object IDs**. |
| **Valid Evaluation Scope** | Registration robustness & failure modes only. | Registration robustness + scene-change visualization; object-lifecycle metrics only if 2D annotated. |

---

## 8. Missing Data & Annotation Requirements

1. **Genuine Physical Photographs:** `data/physical_pilot_v2/visits/visit_01.png`–`visit_10.png` (single fixed resolution, with EXIF/camera metadata preserved) + completed `manifest_v2.json` bounding boxes.
2. **Synthetic 90-Obs Extensions (if pursued):** Global background lighting modulation, object addition sequences, `event_id="return_to_baseline"` tagging, and varied occlusion ratios.
3. **External 2D Annotations (if ETH ASL is used for lifecycle scoring):** ROS bag RGB extraction + 2D bounding boxes + persistent `object_id`s.

---

## 9. Recommended Final Dataset Architecture

Separate datasets strictly by provenance and evaluation role — **never combine accuracy scores across dataset types**:

1. **Tier A — Controlled Synthetic Temporal Benchmark (`synthetic_workbench_v2` / future `v3`):** Evaluates *mechanistic temporal reasoning* (long-term memory, object relocation, removal confirmation latency, occlusion freeze/recovery, ambiguity gating, and explainable event history).
2. **Tier B — External Real-World Research Datasets (`ETH ASL` [if extracted/annotated] & `OpenLORIS-Scene`):** Evaluates *indoor RGB-D multi-visit scene changes* (`ETH ASL`) and *3D viewpoint / parallax registration limits* (`OpenLORIS`).
3. **Tier C — Genuine Physical Demonstration Data (`physical_pilot_v2`, plus `desk_pilot_01` as exploratory pilot):** Evaluates *end-to-end physical camera robustness* (real lighting, shadows, handheld/tripod viewpoint shifts, occlusion, and removal on a real tabletop).
4. **Tier D — Registration-Specific Evaluation (`OpenLORIS` keyframes + `S05` / `ETH ASL office`):** Evaluates *planar homography inlier ratio, reprojection error, and failure-freezing behavior* independently of object detection.

---

## 10. Recommended Next Steps (Ordered by Priority)

1. **Priority 1 — Capture `physical_pilot_v2` (10 Genuine Photographs):** Follow `data/physical_pilot_v2/CAPTURE_CHECKLIST.md` at one fixed resolution with EXIF intact, annotate `manifest_v2.json`, and run `B0`/`B3`/`A2*`/`A5`.
2. **Priority 2 — Fix `generate_synthetic_workbench_v2.py` Gaps Before Expanding to 90 Observations:** Add global background lighting modulation, `object_addition` sequences, `return_to_baseline` `event_id` tagging, and varied occlusion ratios in a separate `synthetic_workbench_v3` generator (keeping `v2` frozen).
3. **Priority 3 — Optional External RGB-D Registration Study (ETH ASL `office` / `living_room`):** Extract RGB frames from `.bag` files to test planar registration stability under high-overlap (`office`/`living_room`) vs multi-week (`lounge`) indoor revisits, without claiming object-lifecycle GT unless annotated.

---

## Audit Readiness Summary
- **Ready Now:** Frozen `synthetic_workbench_v2` (25 obs), `desk_pilot_01` (7 obs, exploratory only), `OpenLORIS` registration boundary results (10 keyframes), `physical_pilot_v2` checklist & schema template, and `A5` ambiguous-distractor probe.
- **Missing:** Genuine physical images (`physical_pilot_v2`), global-background lighting & addition logic in the synthetic generator for 90-obs expansion, and 2D object-lifecycle annotations for ETH ASL / OpenLORIS.
- **Decisions Required:** (1) Whether to capture the 10 `physical_pilot_v2` photos now, (2) whether to authorize implementing the `synthetic_workbench_v3` (90-obs) generator extensions, and (3) whether ETH ASL ROS-bag extraction is needed for the final poster/report.
