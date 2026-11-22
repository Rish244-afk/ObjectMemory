# ObjectMemory Studio — Phase 14C.3 Scientific Audit
## Metric Integrity, Denominator Accounting, and Benchmark Forensic Audit

**Date:** 2026-10-30  
**Phase:** 14C.3  
**Target:** Metric Integrity and Forensic Audit of Physical Workbench Evaluation  
**Reference Results:** `scripts/phase14c3_metric_integrity_results.json`  
**Test Suite:** `tests/test_phase14c3_metric_integrity.py` (6 tests), `tests/test_phase14c2_evaluator.py` (7 tests), 329 full suite tests  

---

### 1. Executive Summary & Purpose

Phase 14C.3 provides an exhaustive mathematical and algorithmic verification of the evaluation metrics introduced in Phase 14C.2. Specifically, this audit addresses:
1. **The 42 Entity-Visit Denominator**: Explicit verification of why each entity has its specific denominator and how appearances/removals are bounded.
2. **Bipartite Kuhn-Munkres Matching**: Verification of one-to-one mapping, coordinate space consistency, class constraints, and audit of uninstantiated ground-truth entities.
3. **Confusion Matrix Reconciliation**: Recomputation of all row sums, column sums, and diagonal entries, resolving an audit finding where uninstantiated detector entities were previously credited with a spurious match under `INSUFFICIENT_EVIDENCE`.
4. **Track B versus Track C Fairness**: Verifying parity of registration, coordinate projection, and temporal reasoning pipelines while explaining the epistemic cause of disparate performance.
5. **Removal and Reappearance Semantics**: Tracing verified vs. unverified vacancy, registration failure gates, and the difference between physical ground truth and epistemic belief.

---

### 2. Denominator Definition and Per-Entity Accounting

The primary metric evaluates **42 entity-visit transitions** across 7 sequential physical visits. The denominator is derived strictly from human annotations in `data/real_workbench/manifest.json`.

#### Denominator Inclusion/Exclusion Rules
- **Visits Before Introduction:** Strictly **excluded**. `notebook_01` was physically introduced to the scene in Visit 4; it does not exist in Visits 1–3 and is not evaluated in those visits.
- **Visits After Confirmed Removal:** `cup_02` was removed in Visit 2 and confirmed removed in Visit 3. In Visits 4–7, the field-of-view was cropped (Group B resolution 584 $\times$ 576), and `cup_02` was dropped from manifest annotations once terminal removal was achieved.
- **Evaluator Consistency:** Both Track B and Track C iterate over the identical parsed annotations from `PhysicalWorkbenchDataset.load()`. Neither track injects phantom annotations or alters the denominator.

#### Per-Entity Lifecycle Stage Breakdown

| Entity ID | Class Label | Evaluated Visits | Visits Excluded | Lifecycle Stage Transitions | Denominator |
| :--- | :--- | :---: | :---: | :--- | :---: |
| `book_01` | book | V01–V07 | None | Stationary reference control (`CONFIRMED_PRESENT` across all visits). | 7 |
| `plant_01` | potted plant | V01–V07 | None | Edge reference control (`CONFIRMED_PRESENT` across all visits; activates 12px border clipping safeguard). | 7 |
| `bottle_01` | bottle | V01–V07 | None | Presence V01–V04 $\rightarrow$ Physical removal V05 (registration failed) $\rightarrow$ Unverified vacancy V06 (streak=1) $\rightarrow$ Physical return V07 (`CONFIRMED_PRESENT`). | 7 |
| `scissors_01`| scissors | V01–V07 | None | Presence V01–V02 $\rightarrow$ Small move V03 $\rightarrow$ Occlusion/absence under notebook V04 $\rightarrow$ Reappearance V05 $\rightarrow$ Presence V06–V07. | 7 |
| `cup_01` | cup | V01–V07 | None | Presence V01–V02 $\rightarrow$ Relocation V03 (`CONFIRMED_MOVED`) $\rightarrow$ Return to baseline V04 $\rightarrow$ Presence V05–V07. | 7 |
| `notebook_01`| book | V04–V07 | V01–V03 (Prior to appearance) | Introduced V04 (`INSUFFICIENT_EVIDENCE`) $\rightarrow$ Removed V05 (`PENDING_REMOVAL`) $\rightarrow$ Persistent absence V06–V07 (`CONFIRMED_REMOVED`). | 4 |
| `cup_02` | cup | V01–V03 | V04–V07 (Terminal post-removal) | Initial observation V01 (`INSUFFICIENT_EVIDENCE`) $\rightarrow$ Removed V02 (`PENDING_REMOVAL`) $\rightarrow$ Absence V03 (`CONFIRMED_REMOVED`). | 3 |
| **TOTAL** | — | — | — | **7 + 7 + 7 + 7 + 7 + 4 + 3** | **42** |

---

### 3. Bipartite Matching Implementation Audit

The entity association between ground-truth annotations and system memory entities was audited in [`scripts/evaluate_physical_workbench.py`](scripts/evaluate_physical_workbench.py) (`compute_ground_truth_entity_mapping`):

1. **Algorithm & One-to-One Guarantee:**
   - Uses `solve_optimal_assignment` from [`src/objectmemory/baseline/comparator.py`](src/objectmemory/baseline/comparator.py), which implements the $O(N^3)$ Kuhn-Munkres (Hungarian) algorithm on rectangular cost matrices.
   - Strictly guarantees that **one ground-truth entity cannot match multiple system entities**, and **one system prediction cannot match multiple ground-truth objects**.
2. **Class Constraints:**
   - Enforced by initializing non-matching class pairs to cost $1000.0$.
   - A threshold gate `cost_matrix[i, j] < 999.0` prevents cross-class assignments even if spatial IoU is high.
3. **Coordinate Space Consistency:**
   - Visit 1 ground-truth annotations and initial system entities are matched in canonical space (in Visit 1, the reference image defines the canonical frame, so raw coordinates equal canonical coordinates).
   - In subsequent visits (e.g. `notebook_01` in Visit 4), newly created system entities possess canonical bounding boxes projected via the visit's planar homography $H$.
4. **Audit Finding — Uninstantiated Entity False Match Correction:**
   - *Forensic Discovery:* In Phase 14C.2, when a ground-truth entity was never detected by the model (`sys_id is None`), the evaluator set `observed = EpistemicState.INSUFFICIENT_EVIDENCE`. Because `cup_02` in V01 and `notebook_01` in V04 had ground-truth annotations expecting `INSUFFICIENT_EVIDENCE`, the comparison `(observed == expected)` evaluated to `True`!
   - *Impact:* The vision detector (YOLOv8n) completely missed `cup_02` in V01 and `notebook_01` in V04, yet the benchmark silently credited the detector with 2 successful predictions!
   - *The Fix:* When `sys_id is None`, the observed state is formally recorded as `"UNKNOWN"`, and `is_match = False`. Uninstantiated entities are placed in the `UNKNOWN` column of the confusion matrix and can never receive a false positive match.
5. **Unmatched Predictions Accounting:**
   - Spurious model detections are audited via `duplicate_tracks = max(0, total_system_ents - unique_gt_ids)`.
   - In Track C, YOLOv8n created 8 persistent entities for 7 ground-truth objects, correctly reporting `duplicate_tracks = 1`.
6. **Identity Persistence & Switch Counting:**
   - Handled across visits by tracking `prior_mappings` against `current_mapping` using IoU $\ge 0.30$.
   - Both Track B and Track C recorded `total_identity_switches = 0`.

---

### 4. Confusion Matrix Reconciliation

All rows, columns, and diagonal elements were recomputed directly from the underlying evaluation records in [`scripts/phase14c3_metric_integrity_results.json`](scripts/phase14c3_metric_integrity_results.json).

#### Track B (Oracle Ground-Truth Inputs) Reconciled Matrix

- **Total Evaluated Transitions:** 42
- **Diagonal Sum (Correct Matches):** 17 + 1 + 0 + 0 + 2 + 0 = **20**
- **Recomputed State Accuracy:** $20 / 42 = \mathbf{0.4762}$ (47.6%)

| Expected State \ Observed State | PRESENT | MOVED | PENDING_REM | CONF_REM | INSUFFICIENT_EVID | AMBIGUOUS_ID | UNKNOWN | Row Sum (GT) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CONFIRMED_PRESENT** | **17** | 3 | 0 | 0 | 5 | 5 | 0 | **30** |
| **CONFIRMED_MOVED** | 0 | **1** | 0 | 0 | 0 | 1 | 0 | **2** |
| **PENDING_REMOVAL** | 3 | 1 | **0** | 0 | 0 | 0 | 0 | **4** |
| **CONFIRMED_REMOVED** | 3 | 0 | 1 | **0** | 0 | 0 | 0 | **4** |
| **INSUFFICIENT_EVIDENCE**| 0 | 0 | 0 | 0 | **2** | 0 | 0 | **2** |
| **AMBIGUOUS_IDENTITY** | 0 | 0 | 0 | 0 | 0 | **0** | 0 | **0** |
| **Column Sum (Observed)**| **23** | **5** | **1** | **0** | **7** | **6** | **0** | **42** |

#### Track C (End-to-End YOLOv8n Vision) Reconciled Matrix

- **Total Evaluated Transitions:** 42
- **Diagonal Sum (Correct Matches):** 24 + 1 + 1 + 0 + 0 + 0 = **26**
- **Recomputed State Accuracy:** $26 / 42 = \mathbf{0.6190}$ (61.9%)
*(Note: Corrected from 66.7% in Phase 14C.2 by removing the 2 false matches for uninstantiated `cup_02` and `notebook_01`)*.

| Expected State \ Observed State | PRESENT | MOVED | PENDING_REM | CONF_REM | INSUFFICIENT_EVID | AMBIGUOUS_ID | UNKNOWN | Row Sum (GT) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CONFIRMED_PRESENT** | **24** | 0 | 1 | 0 | 5 | 0 | 0 | **30** |
| **CONFIRMED_MOVED** | 1 | **1** | 0 | 0 | 0 | 0 | 0 | **2** |
| **PENDING_REMOVAL** | 1 | 0 | **1** | 0 | 0 | 0 | 2 | **4** |
| **CONFIRMED_REMOVED** | 1 | 0 | 0 | **0** | 0 | 0 | 3 | **4** |
| **INSUFFICIENT_EVIDENCE**| 0 | 0 | 0 | 0 | **0** | 0 | 2 | **2** |
| **AMBIGUOUS_IDENTITY** | 0 | 0 | 0 | 0 | 0 | **0** | 0 | **0** |
| **Column Sum (Observed)**| **27** | **1** | **2** | **0** | **5** | **0** | **7** | **42** |

#### Reconciliation Verification
1. **Total Matrix Entries:** Row sums sum to 42; column sums sum to 42.
2. **Ground Truth Consistency:** Row totals (`CONFIRMED_PRESENT`: 30, `CONFIRMED_MOVED`: 2, `PENDING_REMOVAL`: 4, `CONFIRMED_REMOVED`: 4, `INSUFFICIENT_EVIDENCE`: 2) are **100% identical** across Track B and Track C.
3. **Per-Entity and Per-Class Agreement:**
   - Sum of per-entity totals: $\sum \text{total\_visits} = 42$; $\sum \text{correct\_visits} = 26$ (Track C), $20$ (Track B).
   - Sum of per-class totals: $\sum \text{total\_evals} = 42$; $\sum \text{correct\_evals} = 26$ (Track C), $20$ (Track B).
   - Zero discrepancy between aggregate metrics, class metrics, entity records, and confusion matrix cells.

---

### 5. Track B versus Track C Fairness Analysis

Both Track B and Track C execute under strictly equivalent architectural configurations:
- **Registration Engine:** Both run [`PlanarRegistrar`](src/objectmemory/registration/registrar.py) with ORB feature matching and RANSAC homography estimation against the Visit 1 keyframe image.
- **Coordinate Projection:** Both supply raw bounding boxes and rely on `transform_bbox_to_canonical(det.raw_bbox, reg.homography)` inside `engine.process_visit`.
- **Temporal Configuration:** Both use identical parameters (`movement_threshold=35.0px`, `return_to_baseline_threshold=35.0px`, `min_overlap_fraction=0.70`, `ambiguity_margin=0.12`, `vacancy_similarity_max=0.40`, `unobserved_visits_for_removal=3`).
- **State Transition Machine:** Both use identical rules in [`src/objectmemory/temporal/engine.py`](src/objectmemory/temporal/engine.py).

#### The Cause of the Performance Disparity

| Evaluation Dimension | Track B (Oracle Inputs) | Track C (End-to-End Vision) | Scientific Finding |
| :--- | :---: | :---: | :--- |
| **Input Detections** | Perfect human ground-truth boxes | YOLOv8n detector inference ($\text{conf} \ge 0.20$) | Ground truth contains all fine-grained objects. |
| **Initial Entities in V01** | 6 instantiated entities (both `cup_01` and `cup_02`) | 5 instantiated entities (`cup_02` missed by YOLOv8n) | Detector false negative in Visit 1. |
| **Subsequent Spatial Association** | In V02, `cup_02` disappears. The remaining single cup detection falls within the ambiguity margin of both cups ($< 0.12$). | In V02, only one cup exists in memory. No spatial ambiguity competition occurs. | **Detector omission artificially simplified the temporal association problem.** |
| **Epistemic Result for `cup_01`** | **0/7 correct** (Locked in `AMBIGUOUS_IDENTITY`) | **6/7 correct** (Tracks cleanly through relocation and return) | Track B correctly executed the ambiguity safeguard; Track C bypassed ambiguity due to detector blindness. |
| **Primary State Accuracy** | **47.6% (20/42)** | **61.9% (26/42)** | **Crucial Scientific Rule:** Higher Track C accuracy is NOT evidence of better temporal reasoning; it is a direct consequence of detector false negatives eliminating multi-entity ambiguity. |

---

### 6. Removal and Reappearance Lifecycle Analysis

#### State Transition Logic under Unobserved Visits
When an entity is not detected in a visit, the temporal engine enforces a multi-stage verification pipeline:
1. **Registration Check:**
   If registration fails (`AlignmentStatus.INSUFFICIENT_INLIERS` or inlier ratio $< 0.15$), the visit is unverified. Absence streaks are frozen at their prior values, and state updates are suppressed.
2. **Overlap Mask Gate:**
   If the entity's canonical position falls outside registered field-of-view overlap ($< 70\%$), status is set to `OUT_OF_FRAME`, preventing spurious removal.
3. **Occlusion Gate:**
   If another detected object overlaps the entity's canonical box by $\text{IoU} > 0.20$, status is set to `PARTIALLY_OCCLUDED`, preventing removal progression.
4. **Verified Vacancy Branch (Image Patch Similarity $< 0.40$):**
   - 1st visit: `CONFIRMED_PRESENT` $\rightarrow$ `PENDING_REMOVAL` (vacant streak = 1).
   - 2nd visit: `PENDING_REMOVAL` $\rightarrow$ `CONFIRMED_REMOVED` (vacant streak = 2).
5. **Unverified Vacancy Branch (Image Patch Similarity $\ge 0.40$ or Patch Unavailable):**
   - 1st visit: Preserves `CONFIRMED_PRESENT` (unobserved streak = 1). Suppresses removal on single-frame detector false negatives.
   - 2nd visit: `CONFIRMED_PRESENT` $\rightarrow$ `PENDING_REMOVAL` (unobserved streak = 2).
   - 3rd visit: `PENDING_REMOVAL` $\rightarrow$ `CONFIRMED_REMOVED` (unobserved streak = 3).
6. **Reappearance Branch:**
   If an entity currently in `CONFIRMED_REMOVED` or `PENDING_REMOVAL` is re-detected:
   - Immediately transitions to `CONFIRMED_PRESENT` (or `CONFIRMED_MOVED` if relocated $> 35$px).
   - Resets unobserved and vacant streaks to 0.

#### Physical Removal vs. Epistemic Confirmation in the Bottle Sequence
In the real desk pilot:
- **Physical Reality:** Bottle removed in Visit 5; absent in Visit 6; returned to desk in Visit 7.
- **Epistemic System Reality:**
  - Visit 5: Camera shake/blur caused registration to fail. Epistemic streak remained 0.
  - Visit 6: Registration succeeded, but the vacant transparent bottle patch had texture similarity $0.47 \ge 0.40$ (unverified vacancy). Unobserved streak advanced to 1. Epistemic state remained `CONFIRMED_PRESENT`.
  - Visit 7: Bottle returned and was re-detected. Streak reset to 0.

#### Extra Visits Required to Formally Confirm Removal
Because unverified vacancy requires 3 consecutive unobserved visits:
- Visit 5: unverified registration (streak 0, `CONFIRMED_PRESENT`).
- Visit 6: valid unobserved visit 1 (streak 1, `CONFIRMED_PRESENT`).
- **Additional Visit A:** valid unobserved visit 2 (streak 2, transitions to `PENDING_REMOVAL`).
- **Additional Visit B:** valid unobserved visit 3 (streak 3, transitions to `CONFIRMED_REMOVED`).
- **Additional Visit C:** bottle re-placed on desk (re-detected, transitions to `CONFIRMED_PRESENT`).
Therefore, **at least 2 additional unobserved visits (or 3 including terminal reappearance)** are required to validate the complete removal-to-reappearance lifecycle.

---

### 7. Regression Test Suite and Verification

A dedicated regression test suite was implemented in [`tests/test_phase14c3_metric_integrity.py`](tests/test_phase14c3_metric_integrity.py):

| Test Case | Capability Tested | Result |
| :--- | :--- | :---: |
| `test_entity_level_denominator_accounting` | Confirms 42 total entity-visits, 7 each for 5 objects, 4 for notebook, 3 for cup_02. | **PASSED** |
| `test_uninstantiated_ground_truth_accounting` | Confirms uninstantiated entities receive `UNKNOWN` and cannot match `INSUFFICIENT_EVIDENCE`. | **PASSED** |
| `test_unmatched_prediction_accounting_duplicates` | Confirms spurious model detections correctly increment `duplicate_tracks`. | **PASSED** |
| `test_one_to_one_matching_and_class_constraints` | Confirms Kuhn-Munkres enforces strict 1:1 mapping and rejects cross-class assignment. | **PASSED** |
| `test_confusion_matrix_reconciliation_track_b` | Confirms Track B matrix sums equal 42 and diagonal matches reported accuracy. | **PASSED** |
| `test_reappearance_after_confirmed_removal` | Confirms `CONFIRMED_REMOVED` entity re-detected on desk transitions to `CONFIRMED_PRESENT`. | **PASSED** |

#### Execution Command and Output
```powershell
python -m unittest tests/test_phase14c3_metric_integrity.py
# Ran 6 tests in 1.526s — OK (Exit Code: 0)

python -m unittest discover -s tests -p "test_*.py"
# Ran 329 tests in 64.591s — OK (Exit Code: 0)
```

---

### 8. Remaining Scientific Limitations

1. **Short Pilot Sequence Length (7 visits):** The pilot sequence is too short to observe unverified removal confirmation (which requires 3 unobserved visits). Sequences of 10–12 visits are needed for thorough removal testing.
2. **Cropping & Field of View Discontinuity:** The crop from Group A (780 $\times$ 636) to Group B (584 $\times$ 576) between Visits 3 and 4 removes scene margins, contributing to the Visit 5 registration failure.
3. **Small Object Detector Sensitivity:** Standard small model YOLOv8n missed `cup_02` (occluded/small in background) and `notebook_01` (flat kraft paper texture). Multi-scale feature pyramids or specialized fine-tuning are needed to eliminate uninstantiated objects.

---

### 9. Final Trustworthiness Recommendation

**Verdict: HIGH INTEGRITY — TRUSTWORTHY FOR NEXT PHYSICAL VALIDATION STAGE.**

The evaluation harness, bipartite matching algorithms, coordinate projection pipelines, and confusion matrix accounting have been forensically audited and proven mathematically sound:
1. The 42-entity-visit denominator is strictly justified and identically enforced across tracks.
2. The uninstantiated entity false match bug was identified and resolved, eliminating inflated benchmark scores.
3. Confusion matrices reconcile with zero error across row sums, column sums, per-entity records, and class totals.
4. All temporal engine safeguards (registration gating, ambiguity locking, unverified vacancy delay, reappearance restoration) are validated by passing tests.

The benchmark harness is ready to receive full-scale multi-visit laboratory sequences (Phase 15).
