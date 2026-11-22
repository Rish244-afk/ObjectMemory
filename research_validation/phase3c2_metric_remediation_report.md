# ObjectMemory — Phase 3C.2: Benchmark Metric Integrity Remediation Report

**Date:** 2026-11-14  
**Project:** ObjectMemory: Long-Term Visual Memory for Detecting Persistent Environmental Changes  
**Component:** Benchmark Harness (`src/objectmemory/benchmarks/`)  
**Status:** Remediated & Fully Verified (401/401 tests passing)  

---

## 1. Executive Summary

Phase 3C.1 identified critical defects in the evaluation runner (`src/objectmemory/benchmarks/runner.py`) that threatened scientific reproducibility and metric integrity:
1. **Historical State Leakage**: Future state transitions from visit $T$ leaked into visits $1 \dots T-1$ via `store.get_entity(sys_id).current_state` fallbacks, resulting in spurious false-removal penalties.
2. **Coordinate-Frame Distortions**: Non-affine homographies were misapplied to axis-aligned bounding boxes, and frame-level raw detections were inappropriately compared against canonical-space coordinates.
3. **Identity Dropout Under Persistent Relocation**: Hungarian bipartite matching evaluated Visit 1 ground-truth against final Visit $T$ positions, causing objects relocated by $>150$px to fail spatial matching and be discarded as "uninstantiated".
4. **Flawed Mathematical Accounting**: Division-by-zero errors were masked as arbitrary zero scores; False Discovery Proportion and False Removal Rate were conflated; Identity Consistency ignored false negatives / missed detections; and Confirmation Delay averaged zero across unconfirmed events.
5. **B0 Metric Conflation**: The stateless memoryless baseline (B0) was evaluated against temporal states rather than standard frame-level object detection metrics (Precision, Recall, F1).

**Remediation Principle**: All corrections were implemented **strictly within the benchmark evaluation layer** (`src/objectmemory/benchmarks/runner.py` and `tests/test_benchmark_runner.py`). No production temporal reasoning, registration, or detection algorithms were altered.

All 22 focused tests in `tests/test_benchmark_runner.py`, all 64 joint benchmark suite tests, and all 401 repository-wide unit and integration tests now pass with zero regressions.

---

## 2. Detailed Remediation Audit: Defect vs. Correction

| Defect ID | Component & Line Range | Root Cause in Phase 3C Implementation | Architectural Remediation in Phase 3C.2 |
| :--- | :--- | :--- | :--- |
| **D1: Historical State Leakage** | `runner.py`<br>`L879–1048` | Step C evaluated entity states post-hoc after all visits completed. For newly created entities in Visit 1 (`created_in_session`), `rep.entity_updates` was empty, triggering fallback to `store.get_entity(sys_id).current_state`. If an object was removed in Visit $T$, its state at Visit 1 was evaluated as `CONFIRMED_REMOVED`, penalizing Visit 1 as a false removal. | Introduced immutable `EntityVisitSnapshot` and `VisitPredictionSnapshot`. At the conclusion of every visit $v_i$, the exact state, streak counts, canonical bbox, and explanations of all entities are frozen into `visit_snapshots[v_i]`. Step C evaluates strictly against the snapshot corresponding to that visit. |
| **D2: Coordinate-Frame Matching** | `runner.py`<br>`L163–216`, `L938–974` | Perspective homographies were either approximated with min/max box arithmetic without finite checks or raw detections were matched against canonical coordinates. | Implemented `transform_query_bbox_to_reference()` applying exact 4-corner homogeneous coordinate perspective transformation $H \cdot [x, y, 1]^T$, normalized by $w$, validated for finiteness, clamped to canvas bounds, and rejected on degenerate/non-invertible matrices. Frame-level matching between ground truth and detections is performed in query pixels on the identical image canvas. |
| **D3: Relocation Identity Retention** | `runner.py`<br>`L1049–1059`, `L1262–1325` | Hungarian bipartite matching compared Visit 1 GT annotations against final Visit $T$ store bounding boxes. Objects that moved $>150$px exceeded the cost ceiling (1000.0) and were flagged as `uninstantiated_gt`. | Implemented anchored 1-to-1 bipartite matching (`_compute_stable_bipartite_mapping`). GT entities are matched against system entities using their **initial canonical bounding boxes** recorded in Visit 1 (`initial_canonical_bbox` / initial snapshot). Spatial identity is preserved across arbitrary physical displacements and subsequent removals. |
| **D4.1: FRR vs. FDP Conflation** | `runner.py`<br>`L1108–1118`, `L654–661` | $FRR$ was computed as $FP / (TP + FP)$ in some modules and $FP / \text{Total Non-Removed}$ in others, leading to conflicting reporting. | Disentangled into two distinct, mathematically rigorous metrics: **False Removal Rate ($FRR$)** as False Positive Rate $FP / (FP + TN)$ over non-removed eligible visits, and **False Discovery Proportion ($FDP$)** as $FP / (TP + FP)$ over removal predictions. Returns `None` (N/A) when denominator is zero. |
| **D4.2: Confirmation Rate & Delay** | `runner.py`<br>`L1138–1144`, `L1203–1220` | Mean confirmation delay included zero-delay visits for unconfirmed events, artificially diluting latency scores. | Split into **Confirmation Rate ($CR$)** ($N_{\text{confirmed}} / N_{\text{total\_events}}$), accompanied by explicit `unconfirmed_change_events_count`, and **Conditional Confirmation Delay ($CD$)** computed strictly over successfully confirmed change events. Returns `None` when zero change events are confirmed. |
| **D4.3: Identity Consistency ($IC$)** | `runner.py`<br>`L1156–1193` | Denominator summed only visits where the entity was detected ($|V_{\text{detected}}|$), giving $IC = 1.0$ (100%) even if an object was missed in 9 out of 10 visits. | Denominator is defined strictly as total ground-truth present visits ($\sum N_{\text{present}}$), penalizing missed observations / false negatives. An object present in 4 visits but detected in 2 receives $IC = 2/4 = 0.50$ (50%). |
| **D4.4: Track Fragmentation** | `runner.py`<br>`L1194–1201` | Duplicate identity count conflated spurious detections with true track fragmentations without explicit accounting. | Explicitly formalized **Track Fragmentation** as excess system tracks mapped to the same ground-truth identity: $\sum_g \max(0, |S(g)| - 1)$, alongside unmapped **Spurious System Tracks**. |
| **D5: B0 Detector Evaluation** | `runner.py`<br>`L770–850`, `L626–652` | Evaluated B0 (stateless detector) against temporal epistemic states, producing meaningless zero scores for change detection metrics. | Implemented frame-level greedy 1-to-1 bipartite matching (IoU $\ge 0.50$, class match) computing standard computer vision metrics: Detection Precision, Detection Recall, and Detection F1 score ($TP, FP, FN$). Temporal metrics are reported as `None` (N/A) and documented in `unscorable_cases`. |
| **D6: Persistence Integrity** | `tests/test_benchmark_runner.py`<br>`L716–773` | Benchmarks tested only in-memory SQLite (`:memory:`), leaving file-backed connection closure and reconnection unverified. | Added `test_18_restart_persistence_with_file_backed_database` using a real SQLite file, verifying cold connection closure, fresh engine re-opening, entity recovery, and subsequent visit continuation. |
| **D7: Run Fingerprinting** | `runner.py`<br>`L736–743` | Parameter fingerprints omitted detector name, manifest version, and dataset identifiers. | Comprehensive fingerprint captures detector class name, manifest version, dataset name, matching IoU threshold ($0.50$), and all effective temporal parameters. |

---

## 3. Mathematical Formulas and Zero-Denominator Rules

| Metric Name | Mathematical Definition | Denominator | Zero-Denominator Behavior | Logged Unscorable Case |
| :--- | :--- | :--- | :--- | :--- |
| **False Removal Rate ($FRR$)** | $\frac{FP_{\text{removal}}}{FP_{\text{removal}} + TN_{\text{removal}}}$ | Total eligible non-removed GT visits | Returns `None` (N/A) | `"No eligible non-removed ground-truth entity visits evaluated."` |
| **False Discovery Proportion ($FDP$)** | $\frac{FP_{\text{removal}}}{TP_{\text{removal}} + FP_{\text{removal}}}$ | Total removal predictions made by system | Returns `None` (N/A) | `"Zero confirmed removal predictions made by system."` |
| **Removal Recall ($RR$)** | $\frac{TP_{\text{removal}}}{TP_{\text{removal}} + FN_{\text{removal}}}$ | Total GT confirmed removal visits | Returns `None` (N/A) | `"No confirmed removal ground-truth visits present in sequence."` |
| **Event Removal Recall** | $\frac{TP_{\text{event}}}{Total\text{ }Removal\text{ }Events}$ | Total ground-truth removal events | Returns `None` (N/A) | `"No confirmed removal ground-truth events present in sequence."` |
| **Confirmation Rate ($CR$)** | $\frac{|\text{Events}_{\text{confirmed}}|}{|\text{Events}_{\text{total}}|}$ | Total eligible change events (removals + moves) | Returns `None` (N/A) | `"No eligible change events (removals or relocations) in ground truth."` |
| **Conditional Confirmation Delay ($CD$)** | $\frac{1}{|\text{Events}_{\text{confirmed}}|} \sum (v_{\text{conf}} - v_{\text{onset}})$ | Total confirmed change events | Returns `None` (N/A) | `"Zero persistent change events confirmed by system."` |
| **Unsupported Movement Rate ($UMR$)** | $\frac{FP_{\text{movement}}}{FP_{\text{movement}} + TN_{\text{movement}}}$ | Total stationary ground-truth visits | Returns `None` (N/A) | `"No stationary ground-truth entity visits evaluated."` |
| **Identity Consistency ($IC$)** | $\frac{\sum_{g} \max_{s} N(g, s)}{\sum_{g} N_{\text{present}}(g)}$ | Total eligible present GT visits | Returns `None` (N/A) | `"No observable present entity visits in ground truth."` |
| **Track Fragmentation** | $\sum_{g \in G} \max(0, |S(g)| - 1)$ | Integer count | Integer count ($0$) | N/A |
| **Historical Retrieval Accuracy ($HRA$)** | $\frac{|\text{Successful Re-identifications}|}{|\text{Eligible Reappearance Events}|}$ | Total return-to-baseline / reappearance events | Returns `None` (N/A) | `"No return-to-baseline / reappearance events present in ground truth."` |
| **B0 Detection Precision** | $\frac{TP_{\text{det}}}{TP_{\text{det}} + FP_{\text{det}}}$ | Total detector bounding boxes | Returns `None` (N/A) | `"temporal_change_metrics": "Stateless memoryless detection baseline..."` |
| **B0 Detection Recall** | $\frac{TP_{\text{det}}}{TP_{\text{det}} + FN_{\text{det}}}$ | Total active ground-truth annotations | Returns `None` (N/A) | `"temporal_change_metrics": "Stateless memoryless detection baseline..."` |
| **B0 Detection F1** | $\frac{2 \cdot P \cdot R}{P + R}$ | $P + R$ | Returns `None` (N/A) | `"temporal_change_metrics": "Stateless memoryless detection baseline..."` |

---

## 4. Before vs. After Behavior Verification

| Evaluation Scenario | Pre-Remediation Behavior (Phase 3C) | Post-Remediation Behavior (Phase 3C.2) |
| :--- | :--- | :--- |
| **Object removed in Visit 3 of 3** | Visits 1 & 2 evaluated as `CONFIRMED_REMOVED` via future state leakage. $FRR = 2/2 = 1.0$ (100% false removal penalty on correct behavior). | Visits 1 & 2 frozen as `CONFIRMED_PRESENT` from their visit snapshots. False removal count = 0, $FRR = 0.0$. |
| **Object relocates by 424px in Visit 2** | Hungarian match against Visit 2 entity failed spatial threshold. Entity marked `uninstantiated_gt`. Metrics evaluated on 0 entities. | Anchored bipartite matching against Visit 1 canonical bbox links ground-truth identity to system entity. Uninstantiated list is empty; relocation evaluated cleanly. |
| **Detector misses object in 2 of 4 visits** | Evaluated only over 2 detected visits: $IC = 2/2 = 1.0$ (100% identity consistency, hiding missed detections). | Evaluated over all 4 present visits: $IC = 2/4 = 0.50$ (50% identity consistency, properly penalizing detector dropouts). |
| **System predicts zero removals** | $FDP = 0 / 0 \implies$ runtime error or misleading $FDP = 0.0$. | $FDP = \text{None}$ (N/A), logged under `unscorable_cases["false_discovery_proportion"]`. |
| **1 of 2 change events confirmed** | Confirmation Delay averaged 0 visits for unconfirmed event, skewing mean latency downwards. | Confirmation Rate = $1/2 = 50\%$, `unconfirmed_change_events_count = 1`, and Conditional Delay averaged strictly over the 1 confirmed event. |
| **Stateless Detector (B0) evaluated** | Evaluated against temporal states, outputting arbitrary 0 scores for FRR, FDP, CR, Delay. | Evaluates frame-level 1-to-1 matches, outputting Precision, Recall, and F1. Temporal metrics set to `None` (N/A). |

---

## 5. Verification Test Suite Execution Results

### 5.1 Focused Remediation Suite (`tests.test_benchmark_runner`)
```
Ran 22 tests in 12.721s
OK
```
Covering:
- `test_01_valid_and_invalid_manifest_handling`
- `test_02_all_configurations_loadable_and_executable`
- `test_03_rejection_of_blocked_configuration`
- `test_04_execution_isolation_across_multiple_runs`
- `test_05_execution_isolation_across_sequences`
- `test_06_synthetic_and_exploratory_provenance_gating`
- `test_07_end_to_end_metric_reproducibility`
- `test_08_per_frame_detection_matching_b0`
- `test_09_state_accuracy_temporal_baseline_b3`
- `test_10_partial_ablation_reporting_a5`
- `test_11_uninstantiated_gt_entity_reporting`
- `test_12_non_destructive_execution`
- `test_13_exploratory_pilot_disclaimer_in_report`
- `test_14_result_serialization_and_schema`
- `test_15_regression_historical_state_leakage_eliminated`
- `test_16_coordinate_frame_transformations`
- `test_17_object_relocation_retains_identity`
- `test_18_restart_persistence_with_file_backed_database`
- `test_19_identity_consistency_penalizes_missed_observations`
- `test_20_false_discovery_proportion_with_zero_removals`
- `test_21_confirmation_rate_and_unconfirmed_events`
- `test_22_fragmented_identities_per_entity`

### 5.2 Joint Benchmark Suite
```
python -m unittest tests.test_benchmark_manifest tests.test_benchmark_config_factory tests.test_benchmark_runner
Ran 64 tests in 12.098s
OK (skipped=1)
```

### 5.3 Full Repository Test Suite (Regression Verification)
```
python -m unittest discover -s tests -p "test_*.py"
Ran 401 tests in 148.261s
OK (skipped=1)
```
**Regressions: Zero.**

---

## 6. Scope Boundary and Pilot Safety Assessment

1. **Production Code Untouched**:
   All changes were strictly confined to `src/objectmemory/benchmarks/runner.py` and test harnesses. `temporal/engine.py`, `temporal/associator.py`, `registration/`, `persistence/`, and `detection/` were unmodified.
2. **Evaluation Layer Defensibility**:
   The benchmark harness now possesses mathematical rigor, elimination of future-to-past data leakage, perspective coordinate consistency, anchored identity stability under relocation, and honest zero-denominator reporting.
3. **Pilot Evaluation Readiness**:
   The evaluation harness is now safe to execute controlled benchmark runs (B0–B3, A1–A6) on exploratory pilot and physical workbench datasets without generating artifactually biased or invalid metrics.
