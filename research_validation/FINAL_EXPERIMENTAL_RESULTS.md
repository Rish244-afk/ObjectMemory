# ObjectMemory — Final Experimental Results

**Official Project Title:** ObjectMemory: Long-Term Visual Memory for Detecting Persistent Environmental Changes  
**Official Description:** *"An AI-based visual memory system that compares scenes over time to detect and explain persistent changes in object locations while filtering out temporary variations such as lighting, occlusion, and viewpoint changes."*  
**Date:** November 25, 2026  
**Baseline:** Frozen `src/objectmemory/` (`main` `a612fab`, zero thresholds or model weights modified)

---

## 1. Summary of All Experimental Tiers (Kept Strictly Separate)

| Tier | Dataset | Provenance | Images / Obs | Primary Scientific Purpose |
| :--- | :--- | :--- | :---: | :--- |
| **A** | `synthetic_workbench_v2` | `SYNTHETIC_GENERATED` | 25 obs (5 seqs) | Controlled temporal lifecycle, ablation (`A2*`, `A3*`, `A5`), sensitivity & latency |
| **A2** | `a5_ambiguous_distractor_probe` | `SYNTHETIC_GENERATED` | 3 obs (1 seq) | Ambiguity-gating ablation (`B3` vs `A5`) on identical-class distractor |
| **B1** | **ETH ASL Change Detection (`lounge`)** | `PUBLIC_BENCHMARK` | 21,084 PNGs (10 obs) | Multi-week 3D room registration, closed-set detection & absence freezing |
| **B2** | `OpenLORIS-Scene Office1-1` | `PUBLIC_BENCHMARK` | 10 keyframes | 3D AGV depth-parallax planar registration failure boundary |
| **C** | `desk_pilot_01` (`data/real_workbench/`) | `EXPLORATORY_PILOT` *(AI-gen / unverified EXIF)* | 7 obs (1 seq) | Exploratory pipeline smoke sequence (not physical metrology) |
| **D** | `physical_pilot_v2` | `GENUINE_PHYSICAL` | **0** (`AWAITING_PHYSICAL_DATA`) | Prepared 6-scenario folder structure & template for user's physical photos |

---

## 2. Controlled Synthetic Benchmark & Ablations (`synthetic_workbench_v2`, 25 Obs)

Frozen results (`research_validation/results/benchmarks/*.json`) + post-processed metrics (`research_validation/results/final_completion/task2_missing_metrics.json`):
- **`B3` (Complete ObjectMemory):** State Acc **`0.944`** (`118/125`), `FRR` **`0.00%`** (`0/121`), `FDP` **`0.00%`** (`0/4`), Removal Recall **`100%`** (`4/4` visits, `2/2` events), Physical Removal Latency **`2.0` inclusive visits** (`1.0` visit after disappearance onset), Movement Event Recall **`100%`** (`2/2`; visit precision `33.33%` [`2/6`], visit recall `100%` [`2/2`]), `IC` **`97.46%`** (`115/118`), `HRA` on `S02` v5 **`0/1`** (`cup_01` stayed `CONFIRMED_MOVED`).
- **`B0` (Memoryless Detector):** Det $F_1$ **`0.987`**, State Acc **`0.904`** (`113/125`); temporal metrics `N/A`.
- **`A2*` (Partial — No Vacancy Verification):** State Acc drops to **`0.928`** (`116/125`), confirming patch vacancy prevents `2` state errors.
- **`A5` Ambiguity Probe (`task4_a5_association_probe.json`):** On two identical `cup.png` cutouts, **`B3`** (`ambiguity_margin=0.12`) yielded **`0` identity switches**, whereas **`A5`** (`0.0`) caused **`1` identity switch**.
- **Sensitivity Analysis (`task3_sensitivity_analysis.json`):** `vacancy_similarity_max = 0.40` sits in the optimal `[0.30, 0.40]` band (`0.20` drops Removal Recall to `0.50`; `>=0.50` drops State Acc to `0.936`).

---

## 3. ETH ASL Lounge Evaluation (`research_validation/results/eth_asl_lounge/eth_asl_lounge_evaluation_results.json`)

Evaluated strictly under `research_validation/ETH_ASL_EVALUATION_PROTOCOL.md` using frozen `YOLOv8n` (`conf=0.35`), `PlanarRegistrar`, and `TemporalMemoryEngine`:

| Experiment | Frames | Reg Success | Status Distribution | Mean Inlier Ratio / Reproj (Success) | Total Detections | Object-Lifecycle Metrics |
| :--- | :---: | :---: | :--- | :---: | :---: | :--- |
| **Smoke Test** (`obs_01` f0→f1) | 2 | **100%** (`1/1`) | `SUCCESS: 2` | `0.9329` / `0.303 px` | 10 (`couch:9, chair:1`) | `UNAVAILABLE` (No 2D GT) |
| **Intra-Observation** (`obs_01`, Δ150 frames) | 10 | **0.0%** (`0/9`) | `SUCCESS:1`, `SINGULAR:4`, `INSUFF_MATCH:3`, `HOMOG_FAIL:1`, `INSUFF_INL:1` | — | 25 (`couch:7, chair:8, bed:3, toilet:1, ...`) | `UNAVAILABLE` (No 2D GT) |
| **Cross-Observation (First Frames `obs_01`–`obs_10`)** | 10 | **22.2%** (`2/9`) | `SUCCESS:3`, `INSUFF_INL:5`, `SINGULAR:2` | `0.5175` / `0.974 px` | 63 (`chair:32, couch:13, book:9, ...`) | `UNAVAILABLE` (No 2D GT) |
| **Cross-Observation (Translation-Matched)** | 10 | **0.0%** (`0/9`) | `SUCCESS:1`, `INSUFF_INL:4`, `INSUFF_MATCH:4`, `SINGULAR:1` | — | 25 (`chair:11, couch:9, ...`) | `UNAVAILABLE` (No 2D GT) |

### Key Measured Takeaways on ETH ASL Lounge
1. **Honest 3D Viewpoint Boundary:** Across multi-week hand-held 3D room scans (`50–100%` overlap), 2D planar homography succeeded on `2/9` cross-observation first-frame pairs (`obs_02` `35` inliers `0.285`; `obs_06` `30` inliers `0.268`) and `0/9` when matching `T_G_C` translation without rotation. We do **not** claim ObjectMemory handles arbitrary 3D room viewpoints or parallax.
2. **Safe Absence Freezing:** Across all `7/9` failed cross-observation registrations, `TemporalMemoryEngine` safely froze beliefs and produced **zero false `CONFIRMED_REMOVED` hallucinations**.
3. **Closed-Set Detector & Ambiguity Behaviour:** `YOLOv8n` detected indoor furniture (`chair`, `couch`, `book`), with occasional clutter misclassifications (`toilet: 1`, `bed: 3`) during sweeping motion, and `AMBIGUOUS_IDENTITY` (`3–5` entities) triggered appropriately when multiple similar lounge chairs co-occurred.
4. **Qualitative Visual Comparisons:** Side-by-side reference/query detections and registration failure overlays are saved in `research_validation/results/eth_asl_lounge/visual_comparisons/`.

---

## 4. Physical Pilot Preparation (`data/physical_pilot_v2/`)

- Created the 6 scenario directories (`stationary/`, `movement/`, `removal/`, `addition/`, `occlusion/`, `return_to_baseline/`) and manifest templates (`manifest_6_scenarios_template.json`, `manifest_v2_template.json`).
- **Status:** **`AWAITING_PHYSICAL_DATA`** (`0` physical photographs captured yet; strictly separated from ETH ASL and AI-generated Desk Pilot).

---

## 5. Reproducibility Commands & Regression Status

```bash
# 1. Stream-download & extract ETH ASL Lounge (10 observations, 21,084 PNGs)
python -u scripts/prepare_eth_asl_lounge.py

# 2. Run ETH ASL Lounge evaluation (smoke, intra-obs, cross-obs, visual comparisons)
PYTHONPATH=src python -u scripts/evaluate_eth_asl_lounge.py

# 3. Run full regression test suite
PYTHONPATH=src python -m unittest discover -s tests -p "test_*.py"
```
- **Regression Suite Outcome:** `Ran 401 tests in 61.845s — OK (skipped=1)` (`400` passed, `1` skipped, `0` failures, `0` errors).
