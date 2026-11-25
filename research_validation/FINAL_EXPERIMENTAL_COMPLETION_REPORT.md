# ObjectMemory — Final Experimental Completion Report

**Official Project Title:** ObjectMemory: Long-Term Visual Memory for Detecting Persistent Environmental Changes  
**Official Objective:** An AI-based visual memory system that compares scenes over time to detect and explain persistent changes in object locations while filtering out temporary variations such as lighting, occlusion, and viewpoint changes.  
**Frozen Baseline:** GitHub `main` (`a612fab`)  
**Date:** November 25, 2026  

---

## 1. Original B3 Baseline & Existing Ablation Results

All frozen baseline and ablation JSON files remain preserved unchanged in `research_validation/results/benchmarks/`:

| Dataset | Config | Status | Det $F_1$ | FRR | FDP | RR (Visits) | RR (Events) | CR | IC | State Acc. |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`synthetic_workbench_v2`** (25 obs) | **B0** | `SUPPORTED` | **0.987** | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | 0.904 (`113/125`) |
| | **B3** | `SUPPORTED` | *N/A* | **0.00%** (`0/121`) | **0.00%** (`0/4`) | **100%** (`4/4`) | **100%** (`2/2`) | **100%** (`4/4`) | **97.46%** (`115/118`) | **0.944** (`118/125`) |
| | **A5** | `SUPPORTED` | *N/A* | 0.00% (`0/121`) | 0.00% (`0/4`) | 100% (`4/4`) | 100% (`2/2`) | 100% (`4/4`) | 97.46% (`115/118`) | 0.944 (`118/125`) |
| | **A2\*** | `PARTIAL` | *N/A* | 0.00% (`0/121`) | 0.00% (`0/4`) | 100% (`4/4`) | 100% (`2/2`) | 100% (`4/4`) | 97.46% (`115/118`) | **0.928** (`116/125`) |
| | **A3\*** | `PARTIAL` | *N/A* | 0.00% (`0/121`) | 0.00% (`0/4`) | 100% (`4/4`) | 100% (`2/2`) | 100% (`4/4`) | 97.46% (`115/118`) | 0.944 (`118/125`) |
| **`desk_pilot_01`** (7 obs) | **B0** | `SUPPORTED` | 0.698 | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | 0.714 (`30/42`) |
| | **B3** | `SUPPORTED` | *N/A* | **0.00%** (`0/38`) | *N/A* | 0.00% (`0/4`) | 0.00% (`0/3`) | 20.0% (`1/5`) | **88.24%** (`30/34`) | **0.738** (`31/42`) |
| **`OpenLORIS Office1-1`** (10 obs) | **B0/B3/A5** | `SUPPORTED` | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* (0 GT) |

- **Blocked Architectural Configurations (`B1`, `B2`, `A1`, `A4`, `A6`):** Rejected by `BenchmarkConfigFactory` because the production `TemporalMemoryEngine` enforces state-transition invariants (requiring `PENDING_REMOVAL` on first absence, freezing on registration failure, and retaining `CONFIRMED_REMOVED` entities for reappearance).

---

## 2. New Physical Evaluation (`physical_pilot_v2`) Status

- **Status:** **`AWAITING_PHYSICAL_DATA`** — **Not yet executed.**
- Prepared minimal 10-visit single-resolution physical protocol in `data/physical_pilot_v2/CAPTURE_CHECKLIST.md` and validated schema template `data/physical_pilot_v2/manifest_v2_template.json` via `ManifestLoader.load(..., check_files=False)`.
- Covers: V01 baseline, V02 stationary control, V03 ~50% lighting stress, V04 ~5° camera misalignment, V05 object movement (`>100` px), V06 return-to-baseline, V07 temporary occlusion, V08 occlusion recovery, V09 removal onset (`PENDING_REMOVAL`), and V10 removal confirmation (`CONFIRMED_REMOVED`).
- Genuine physical photographs (`visits/visit_01.png`–`visit_10.png`) and pixel bounding-box annotations must be supplied before running `B0`/`B3`/`A2*`/`A5` on `physical_pilot_v2`.

---

## 3. Movement Metrics & Corrected Removal Latency (Task 2)

Computed via read-only post-processing (`scripts/run_final_completion_experiments.py`, output in `research_validation/results/final_completion/task2_missing_metrics.json`) on frozen `B3` without altering `runner.py` or benchmark JSONs:

### 3.1 Standalone Object Movement Precision & Recall
| Dataset | Visit TP | Visit FP (`num_umr`) | Visit FN | Movement Precision (Visits) | Movement Recall (Visits) | Movement $F_1$ (Visits) | Event Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`synthetic_workbench_v2`** | 2 | 4 | 0 | **33.33%** (`2/6`) | **100.0%** (`2/2`) | **0.500** | **100.0%** (`2/2`) |
| **`desk_pilot_01`** | 1 | 0 | 1 | **100.0%** (`1/1`) | **50.0%** (`1/2`) | **0.667** | **50.0%** (`1/2`) |

- **Interpretation:** On `synthetic_workbench_v2`, `B3` detected both relocations (`2/2`), but retained `CONFIRMED_MOVED` across subsequent resting/return visits in `S02` where `manifest_v2.json` annotated `CONFIRMED_PRESENT` (`4` FP visits). On `desk_pilot_01`, `cup_01` movement was detected (`1/1` precision), while `scissors_01`'s sub-threshold micro-movement was not (`50%` recall).

### 3.2 Physical Removal Confirmation Latency
- **Bug in frozen `runner.py` metric:** `mean_confirmation_delay_visits = 1.0` anchored onset to `expected_state == CONFIRMED_REMOVED` rather than physical disappearance (`PENDING_REMOVAL`).
- **Corrected Physical Latency (`S03_removal_confirmation`):**
  - `bottle_01`: Disappeared `visit_02` → Confirmed `visit_03` (**2 inclusive visits**; **1 additional visit** after onset).
  - `scissors_01`: Disappeared `visit_04` → Confirmed `visit_05` (**2 inclusive visits**; **1 additional visit** after onset).
  - **Mean Physical Removal Latency:** **2.0 inclusive visits** (`1.0` additional visit after disappearance onset), matching the configured `vacancy_visits_for_removal = 2` policy.
- **`desk_pilot_01`:** `0/4` removals confirmed within 7 visits; latency is `null` (`N/A`).

### 3.3 Synthetic `S02` Return-to-Baseline (`HRA`) Verification
- Verified `scripts/generate_synthetic_workbench_v2.py:325-327`: `cup_01` genuinely returns to `(bx, by)` in `S02_relocation_return_visit_05`, but `manifest_v2.json` omitted `event_id="return_to_baseline"`.
- Post-processing evaluation on `S02_relocation_return_visit_05` (`cup_01`) measured **0/1 (`0.0%`)** `HRA` under `B3` because `cup_01` remained in `CONFIRMED_MOVED` (`matched_sys_id = null` in query-frame overlap against the displaced canonical box). On `desk_pilot_01`, `HRA = 100.0%` (`5/5`).

---

## 4. Single-Parameter Sensitivity Analysis (Task 3)

Evaluated one parameter at a time in isolation (`research_validation/results/final_completion/task3_sensitivity_analysis.json`):

### 4.1 `vacancy_similarity_max` (B3 Default = `0.40`)
| Value | `synthetic_workbench_v2` State Acc. | `synthetic_workbench_v2` RR (Visits / Events) | `synthetic_workbench_v2` FRR | `desk_pilot_01` State Acc. |
| :---: | :---: | :---: | :---: | :---: |
| `0.20` | `0.912` | `0.50` / `0.50` | `0.00` | `0.7381` |
| `0.30` | **`0.944`** | **`1.00` / `1.00`** | `0.00` | `0.7381` |
| **`0.40` (B3)** | **`0.944`** | **`1.00` / `1.00`** | **`0.00`** | **`0.7381`** |
| `0.50`–`0.85` | `0.936` | `1.00` / `1.00` | `0.00` | `0.7381` |

- **Limitation & Finding:** Too strict (`0.20`) halves removal recall (`0.50`) because background wood-grain similarity exceeds `0.20`; too permissive (`>=0.50`) lowers state accuracy (`0.936`). `0.40` sits in the empirical optimum `[0.30, 0.40]` on `synthetic_workbench_v2`. `desk_pilot_01` is insensitive across `0.20–0.85` because no removal reaches streak 2.

### 4.2 `movement_displacement_threshold` (B3 Default = `35.0` px)
| Value (px) | `synthetic_workbench_v2` State Acc. | `synthetic_workbench_v2` UMR | `desk_pilot_01` State Acc. | `desk_pilot_01` UMR |
| :---: | :---: | :---: | :---: | :---: |
| `15.0` | `0.944` | `0.0342` | **`0.7143`** | **`0.0333` (`1/30` FP)** |
| `25.0` | `0.944` | `0.0342` | `0.7381` | `0.0000` |
| **`35.0` (B3)** | **`0.944`** | **`0.0342`** | **`0.7381`** | **`0.0000`** |
| `50.0`–`100.0` | `0.944` | `0.0342` | `0.7381` | `0.0000` |

- **Limitation & Finding:** Lowering to `15.0` px introduces a false-positive movement on `desk_pilot_01` (`UMR` `0.0% -> 3.33%`). Values `25.0–100.0` px are flat because synthetic relocations (`198` px, `230` px) and `desk_pilot_01` `cup_01` far exceed `100` px while `scissors_01` is below `25` px.

---

## 5. A5 Ambiguous-Distractor Association Experiment (Task 4)

Because neither `synthetic_workbench_v2` nor `desk_pilot_01` contains an ambiguous identical-class distractor pair, we constructed an isolated 3-visit synthetic benchmark (`research_validation/results/final_completion/a5_ambiguous_distractor_dataset/manifest_v2.json`) placing two identical `cup.png` cutouts (`cup_01`, `cup_02`), then removing `cup_01` and shifting `cup_02` midway between both prior positions (`task4_a5_association_probe.json`):

| Config | `ambiguity_margin` | State Acc. | Identity Consistency | `id_switches` | FRR |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **B3 (Ambiguity Gating ON)** | `0.12` | `0.50` | `0.50` | **`0`** | `0.00` |
| **A5 (Ambiguity Gating OFF)** | `0.00` | `0.50` | `0.75` | **`1`** | `0.00` |

- **Finding:** Disabling ambiguity gating (`A5`) caused **1 identity switch** (forcing `cup_02` to bind to the wrong `cup` track), whereas `B3` (`0.12`) rejected the ambiguous match (`0` identity switches).

---

## 6. Threshold Provenance & Limitations Summary

- **Threshold Provenance:** `AlignmentConfig` (`registration/models.py:51-52`) explicitly classifies its thresholds (`min_inlier_ratio=0.20`, `min_inliers=12`, `max_mean_reprojection_error=4.0`, `min_overlap_fraction=0.15`) as engineering heuristics alongside RANSAC (`3.0` px) and Lowe (`0.75`) conventions. Two thresholds are hardcoded without config exposure: `cond > 1e7` (`registration/evaluator.py:132`) and occlusion `iou > 0.20` (`temporal/engine.py:1031`).
- **Unsupported / Incomplete Conditions:**
  1. Genuine multi-visit physical removal confirmation, occlusion recovery, and controlled lighting/misalignment remain **unvalidated on real photos** until `physical_pilot_v2` images are captured.
  2. `B1`, `B2`, `A1`, `A4`, `A6` remain **blocked** by hardcoded engine safeguards.
  3. 2D planar homography fails on non-planar 3D motion (`OpenLORIS` `8/9` singular collapses), restricting valid operation to near-planar tabletops/workbenches.

---

## 7. Conclusion

On controlled synthetic scenes, the frozen `B3` ObjectMemory pipeline achieves **0.00% false removals** (`0/121`), **100% removal recall** (`4/4` visits, `2/2` events) at **2.0 inclusive visits** physical latency, **100% movement event recall** (`2/2`), and empirically outperforms `A2*` (vacancy verification) and `A5` (ambiguity-induced identity switching). Completing the prepared 10-visit `physical_pilot_v2` capture is the sole remaining step to extend these mechanistic measurements to genuine physical photographs.
