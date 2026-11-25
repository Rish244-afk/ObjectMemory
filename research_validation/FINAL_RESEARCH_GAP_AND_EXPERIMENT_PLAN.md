# ObjectMemory — Final Research Gap Assessment & Experiment Plan

**Document ID:** `OM-RESEARCH-GAP-01`  
**Date:** November 25, 2026  
**Baseline:** Frozen GitHub `main` (`a612fab`)  
**Scope:** Read-only scientific audit of experimental coverage, threshold provenance, real-world evaluation gaps, and prioritized pre-poster/report experiments.

---

## 1. Existing Experimental Evidence

All serialized benchmark outputs reside in `research_validation/results/benchmarks/` (15 JSON files) executed via `BenchmarkRunner` (`src/objectmemory/benchmarks/runner.py`):

| Dataset | Provenance | Visits (Obs) | Executed Configs | Key Verified Findings (from JSON) |
| :--- | :--- | :---: | :--- | :--- |
| **`synthetic_workbench_v2`** | `SYNTHETIC_GENERATED` (2D photographic cutouts on procedural wood) | 25 (5 seqs × 5) | `B0`, `B3`, `A5`, `A2*`, `A3*` | `B3` State Acc `0.944` (`118/125`) vs `B0` `0.904` (`113/125`); `FRR = 0.00` (`0/121`), `FDP = 0.00` (`0/4`), `RR = 1.00` (`4/4` visits, `2/2` events), `CR = 1.00` (`4/4`), `IC = 0.9746` (`115/118`), `UMR = 0.0342` (`4/117`). `A2*` drops State Acc to `0.928` (`116/125`). |
| **`desk_pilot_01`** (`data/real_workbench/`) | `EXPLORATORY_PILOT` (7 smartphone crops, mixed resolutions `780×636` & `~584×576`) | 7 (1 seq) | `B0`, `B3`, `A5`, `A2*`, `A3*` | `B3` State Acc `0.7381` (`31/42`) vs `B0` `0.7143` (`30/42`); `FRR = 0.00` (`0/38`), `FDP = null` (`0/0`), `RR = 0.00` (`0/4` visits, `0/3` events), `CR = 0.20` (`1/5`), `IC = 0.8824` (`30/34`), `HRA = 1.00` (`5/5`). All temporal configs (`B3`, `A5`, `A2*`, `A3*`) produced identical scores. |
| **`OpenLORIS-Scene Office1-1`** | `PUBLIC_BENCHMARK` (AGV RealSense D435i) | 10 (1 seq) | `B0`, `B3`, `A5` | **No object bounding-box or lifecycle GT** (`total_entity_evaluations = 0`). All detection and temporal metrics are `null`. `8/9` planar transitions collapsed (`SINGULAR_HOMOGRAPHY` due to 3D parallax); demonstrates planar failure boundary only. |
| **`controlled_temporal_v1`** | `SYNTHETIC_GENERATED` (legacy Phase 10 manifest) | 30 (6 seqs × 5) | `B0`, `B3` | Legacy JSONs (`B0_synthetic_result.json`, `B3_synthetic_result.json`) show State Acc `0.0337` and `0.0449` due to legacy state-label schema mismatch; superseded by `synthetic_workbench_v2`. |

---

## 2. Existing Baseline and Ablation Coverage

Defined in `BenchmarkConfigFactory` (`src/objectmemory/benchmarks/config_factory.py`):

### 2.1 Baselines (`B0`–`B3`)
- **`B0` (`SUPPORTED`) — Memoryless Detector Reference:** Per-frame `OpenCVDNNDetector` (`YOLOv8n-ONNX`, `conf=0.35`, `nms=0.45`). No temporal state or tracking; temporal metrics reported as `N/A`.
- **`B1` (`BLOCKED`) — Conventional Hungarian Tracking (Immediate Removal):** Blocked by `VALID_STATE_TRANSITIONS` (`domain/states.py:77`) and `engine.py:1070`, which forbid `CONFIRMED_PRESENT -> CONFIRMED_REMOVED` on a single missed detection.
- **`B2` (`BLOCKED`) — Ungated Temporal Memory:** Blocked by `engine.py:983` (unconditionally freezes absence on registration failure or `overlap_mask is None`) and hardcoded occlusion IoU (`0.20` at `engine.py:1031`).
- **`B3` (`SUPPORTED`) — Complete ObjectMemory:** Production `TemporalEngineConfig()` defaults with all 10 safeguards enabled.

### 2.2 Ablations (`A1`–`A6`)
- **`A1` (`BLOCKED`) — No Registration Absence Freezing:** Blocked by `engine.py:83` (`registration.success == False` returns `False`) and `engine.py:985`.
- **`A2` (`PARTIALLY_SUPPORTED`, run as `A2*`) — No Appearance Vacancy Verification:** Sets `vacancy_similarity_max = 1.0`, forcing valid aligned crops (`patch_sim < 1.0`) to evaluate as vacant. However, missing aligned images or degenerate crops still fall back to `unobserved_visits_for_removal = 3`.
- **`A3` (`PARTIALLY_SUPPORTED`, run as `A3*`) — No Visibility & FOV Safeguards:** Sets `min_overlap_fraction = 0.0`, disabling canonical bbox FOV clipping (`engine.py:1013`). However, hardcoded inter-detection occlusion IoU (`0.20` at `engine.py:1031`) remains active.
- **`A4` (`BLOCKED`) — Single-Visit Removal Confirmation:** Blocked by `engine.py:1070`, which hardcodes `target_state = PENDING_REMOVAL` on visit 1 even when `vacancy_visits_for_removal = 1`.
- **`A5` (`SUPPORTED`) — No Ambiguity Handling:** Sets `ambiguity_margin = 0.0` in `TemporalEngineConfig`. (Currently yields identical scores to `B3` on both `synthetic_workbench_v2` and `desk_pilot_01` because neither dataset triggers a sub-`0.12` ambiguous candidate pair.)
- **`A6` (`BLOCKED`) — No Historical Identity Recovery:** Blocked by `engine.py:346, 614`, which unconditionally loads and re-identifies `CONFIRMED_REMOVED` entities.

---

## 3. Missing Metrics & Measurement Audit

Inspection of `BenchmarkRunner` (`src/objectmemory/benchmarks/runner.py`) and dataset manifests (`manifest_v2.json`) reveals:

1. **Object Movement Precision & Recall:**
   - **Current state:** `BenchmarkRunner` does **not** report standalone Movement Precision or Movement Recall. It reports `unsupported_movement_rate` (false-positive movement rate on stationary visits: `4/117 = 3.42%` on `synthetic_workbench_v2`, `0/30 = 0.0%` on `desk_pilot_01`) and pools `CONFIRMED_MOVED` with `CONFIRMED_REMOVED` inside `confirmation_rate`. (Note: `ground_truth_matcher.py` implements `movement_recall` and `false_movement_rate`, but `BenchmarkRunner` does not use it.)
   - **Computability from existing data:** **Yes.** Both `synthetic_workbench_v2` (`S02`: 2 `MOVED`->`CONFIRMED_MOVED` events) and `desk_pilot_01` (`visit_03`: 2 `CONFIRMED_MOVED` events) have movement GT, and `runner.py` already captures per-visit snapshots and `num_umr`.
2. **Removal Confirmation Latency:**
   - **Current state:** `runner.py` reports `mean_confirmation_delay_visits` (`1.0` visit on `synthetic_workbench_v2`). **Critical caveat:** `runner.py:1138-1143` sets `first_change_onset` at the first visit where `expected_state == CONFIRMED_REMOVED`, rather than the first visit where the object physically disappears (`PENDING_REMOVAL`). Thus `1.0` measures delay relative to the *expected confirmation visit*, not *physical disappearance latency* (which is 2 visits in `S03`).
   - **Computability from existing data:** **Yes** for `synthetic_workbench_v2` (both `PENDING_REMOVAL` and `CONFIRMED_REMOVED` are annotated); **not computable** on `desk_pilot_01` because 0 removals reached confirmation.
3. **Historical Retrieval Accuracy (`HRA`) on `synthetic_workbench_v2`:**
   - Reported as `null` (`0/0`) because `S02_relocation_return` `visit_05` lacks `event_id = "return_to_baseline"` or `"return"` in `evidence_notes` (`runner.py:1146-1150`).
4. **Registration Failure / Quality Metrics:**
   - `BenchmarkMetrics` does not aggregate per-visit registration success rate, mean inlier ratio, or mean reprojection error across sequences.

---

## 4. Threshold Provenance

All registration thresholds in `AlignmentConfig` (`src/objectmemory/registration/models.py:9-10, 51-52`) carry an explicit docstring disclaimer: *"engineering heuristics chosen for typical planar workbench scenarios. They are NOT claimed to be scientifically validated constants."*

| Threshold | Exact Source File & Parameter | Current Value | Origin Documented? | Classification | Existing Data Supports Sensitivity Analysis? |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **Registration inlier ratio** | `registration/models.py:60` (`AlignmentConfig.min_inlier_ratio`) & `temporal/models.py:45` (`TemporalEngineConfig.min_registration_inlier_ratio`) | `0.20` | Yes (disclaimer in `models.py`) | Engineering parameter | **Partial** — `synthetic_workbench_v2` & `desk_pilot_01` have high inlier ratios; `OpenLORIS` collapses on `cond > 1e7` before inlier check. |
| **Reprojection error** | `registration/models.py:57` (`ransac_reproj_threshold`), `registration/models.py:61` (`max_mean_reprojection_error`), `temporal/models.py:47` (`max_reprojection_error`) | `3.0` px (RANSAC) / `4.0` px (mean gate) | Partial (`3.0` px standard RANSAC; `4.0` px heuristic) | `3.0` px: Algorithmic convention; `4.0` px: Engineering parameter | **Partial** — computable on `S05` viewpoint shifts & `desk_pilot_01`, but no ground-truth homographies exist. |
| **Minimum inliers** | `registration/models.py:59` (`AlignmentConfig.min_inliers`; `min_tentative_matches=10` at line 58) | `12` | Yes (disclaimer in `models.py`) | Engineering parameter (above 4-pt DLT minimum) | **No** — current valid sequences have hundreds of inliers; blank/featureless frames have `<4`. |
| **Homography condition number** | `registration/evaluator.py:131-132` (hardcoded `cond > 1e7`; `abs(det) < 1e-8` at line 122) | `1e7` | No (undocumented magic constant) | Numerical stability engineering parameter | **No** — only tested cleanly in `OpenLORIS` where collapsed matrices hit `1e7–1e10`. |
| **Overlap threshold** | `registration/models.py:62` (`AlignmentConfig.min_overlap_fraction`) vs `temporal/models.py:46` (`TemporalEngineConfig.min_overlap_fraction`) | `0.15` (reg) / `0.70` (temporal bbox) | Yes (`models.py` heuristic; `A3` tests `0.0`) | Engineering parameter | **Yes (for `0.70`)** — `desk_pilot_01` has FOV shift between V1–V3 and V4–V7; `S05` has mild tilt. |
| **Occlusion threshold** | `temporal/engine.py:1031` (hardcoded `iou > 0.20`) | `0.20` | **No** (hardcoded, not in `TemporalEngineConfig`) | Engineering parameter | **Limited** — only `S04` (`scissors_01` under folder) and `desk_pilot_01` V4 test occlusion. |
| **Vacancy similarity threshold** | `temporal/models.py:51` (`TemporalEngineConfig.vacancy_similarity_max`) | `0.40` | Yes (`A2` tests `1.0`) | Engineering parameter | **Yes** — `S03`, `S04`, and `desk_pilot_01` provide vacant vs occluded/missed patches. |
| **Movement threshold** | `temporal/models.py:62` (`movement_displacement_threshold`) + `temporal/engine.py:98-159` ($\Delta U \in [0, 45]$ px) | `35.0` px base (up to `80.0` px effective) | Yes (`docs/phase6_registration_design.md`, `docs/phase7_evaluation_report.md` call $\Delta U$ an engineering heuristic) | Engineering parameter (resolution-dependent) | **Yes** — `S02` (`198` px), `S05` jitter, and `desk_pilot_01` (`cup_01` vs `scissors_01` micro-move) test stationary vs moved separation. |
| **Multi-visit vacancy confirmation** | `temporal/models.py:56-59` (`vacancy_visits_for_pending=1`, `vacancy_visits_for_removal=2`, `unobserved_visits_for_removal=3`) | `1` / `2` / `3` visits | Yes (`temporal/models.py:54-59`) | Design policy / engineering parameter | **Partial** — `S03` tests 2-visit confirmation; `desk_pilot_01` fails at `0/3` events due to short sequence length. |

---

## 5. Proposed Additional Physical Evaluation Protocol

The existing `desk_pilot_01` (`7` visits) mixes two resolutions (`780×636` vs `~584×576`), lacks EXIF/camera-rig control, and is too short to confirm removals (`0/3` events confirmed). Meanwhile, `data/physical_workbench_v1/CAPTURE_PROTOCOL.md` defines a 13-visit protocol that has `0` captured images (`AWAITING_PHYSICAL_DATA`).

### Smallest Useful Physical Protocol (`physical_pilot_v2`, 10 Visits, 1 Sequence)
- **Setup:** Single fixed resolution (e.g., `1280×720` or `1920×1080`), 5 COCO-supported tabletop objects (`cup`, `bottle`, `scissors`, `book`, `cell phone`) on a textured desk/mat, manual exposure/white-balance lock.
- **Required 10 Visits:**
  1. **V01 (Baseline):** All 5 objects placed (`CONFIRMED_PRESENT`).
  2. **V02 (Stationary Control):** No change (tests `UMR`, `FRR`).
  3. **V03 (Lighting Stress):** ~50% overhead dimming or oblique desk-lamp shadow, objects stationary (tests lighting invariance of 148-d HSV-Sobel & registration).
  4. **V04 (Camera Misalignment):** Tripod/handheld shift (~5° tilt + ~5 cm lateral shift), objects stationary (tests planar homography & $\Delta U$ uncertainty margin).
  5. **V05 (Object Movement):** `cup_01` relocated $>100$ px (`CONFIRMED_MOVED`).
  6. **V06 (Return to Baseline):** `cup_01` returned to V01 position (`CONFIRMED_PRESENT`, `event_id="return_to_baseline"`).
  7. **V07 (Temporary Occlusion):** Cardboard/folder placed over `scissors_01` (tests `PARTIALLY_OCCLUDED` freeze, no false removal).
  8. **V08 (Occlusion Recovery):** Folder removed; `scissors_01` re-detected (`CONFIRMED_PRESENT`).
  9. **V09 (Removal Onset):** `bottle_01` removed (`PENDING_REMOVAL`, streak 1).
  10. **V10 (Removal Confirmation):** `bottle_01` remains absent (`CONFIRMED_REMOVED`, streak 2).
- **Annotations (`manifest_v2.json`):** Per-object `bounding_box`, `object_state`, `visibility_status`, and explicit `event_id` (`UNCHANGED`, `RELOCATED`, `return_to_baseline`, `ABSENT`).
- **Metrics:** `FRR`, `FDP`, `RR` (visits & events), Physical Removal Latency (V09→V10), Movement Precision & Recall, `UMR`, `HRA`, `IC`, `Frag`, and per-visit registration inlier ratio / reprojection error.

---

## 6. Research Gap Assessment & Claim Corrections

| Area | What Project Descriptions Claim | What Code Actually Implements | What Current Experiments Demonstrate | Required Claim Correction |
| :--- | :--- | :--- | :--- | :--- |
| **Real-World Validation** | "Validates persistent object memory in real physical environments" | Full pipeline runs on real images | `desk_pilot_01` has `0%` (`0/4` visits, `0/3` events) removal recall and `20%` (`1/5`) CR; `physical_workbench_v1` has 0 images. | Frame `desk_pilot_01` strictly as an exploratory pipeline smoke pilot; complete the 10-visit physical protocol before claiming real-world removal/occlusion validation. |
| **Ablation Evidence** | "Ablations A1–A6 isolate each safeguard" | `B1, B2, A1, A4, A6` are `BLOCKED`; `A2, A3` are `PARTIAL`; `A5` is `SUPPORTED`. | Only `A2*` shows a measurable delta (`92.8%` vs `94.4%` on synthetic v2). `A3*` and `A5` equal `B3` across all tested datasets. | Explicitly label `A2*`/`A3*` as partial, state `B1/B2/A1/A4/A6` are unexecuted architectural invariants, and either add an ambiguous distractor test for `A5` or report `A5` as non-differentiated on current data. |
| **Movement & Latency Metrics** | "Tracks persistent displacements and multi-visit latency" | `runner.py` tracks `UMR` and `confirmation_rate` (combined) + `delay` from `CONFIRMED_REMOVED` onset. | Standalone movement P/R is unreported; `mean_confirmation_delay_visits = 1.0` understates true 2-visit physical removal latency. | Report movement P/R separately and define latency from physical disappearance (`PENDING_REMOVAL` onset) to confirmation. |
| **Scene Geometry & Detector** | "Robust to viewpoint changes and open environments" | 2D planar homography (`H`) + closed-set 80-class `YOLOv8n` + 148-d HSV-Sobel. | Planar homography fails (`8/9` singular) on 3D `OpenLORIS`; `YOLOv8n` precision on `desk_pilot_01` is `0.577`. | Restrict scope explicitly to near-planar tabletops/workbenches and COCO-80 object classes. |

---

## 7. Prioritized Execution Plan (For Report & Poster)

1. **Priority 1 — Capture & Evaluate 10-Visit Physical Protocol (`physical_pilot_v2`):** Execute the 10-visit capture and `manifest_v2.json` annotation using the frozen pipeline (`B0`, `B3`, `A2*`, `A5`) to obtain genuine physical removal recall, occlusion recovery, lighting invariance, and camera misalignment metrics.
2. **Priority 2 — Compute Missing Metrics on Existing Frozen Outputs (Zero Code/Threshold Changes):** Post-process existing `synthetic_workbench_v2` and `desk_pilot_01` predictions/manifests to report standalone **Movement Precision/Recall**, **Physical Removal Latency** (from `PENDING_REMOVAL` onset), and `S02` **Return-to-Baseline (`HRA`)**.
3. **Priority 3 — Run Read-Only Threshold Sensitivity Sweeps:** Using `TemporalEngineConfig` overrides in an isolated evaluation script (without modifying production defaults), sweep `vacancy_similarity_max` (`0.2–0.7`) and `movement_displacement_threshold` (`15–65` px) on `synthetic_workbench_v2` and `physical_pilot_v2` to give empirical justification for the heuristic defaults.
4. **Priority 4 — Add One Ambiguous-Object Test Case for `A5`:** Include two visually similar same-class objects (e.g., two white mugs) in one synthetic or physical sequence so `A5` (`ambiguity_margin=0.0`) vs `B3` (`0.12`) demonstrates an empirical identity-switch delta.
5. **Priority 5 — Align Poster & End-Semester Report Claims:** Present `synthetic_workbench_v2` as controlled mechanistic validation, `physical_pilot_v2` (once run) as physical validation, `OpenLORIS` strictly as the 3D planar failure boundary, and `B1/B2/A1/A4/A6` as architectural invariants enforced by design.
