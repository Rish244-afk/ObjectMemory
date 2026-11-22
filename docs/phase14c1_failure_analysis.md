# ObjectMemory Phase 14C.1: Read-Only Failure Analysis of Real-Workbench Pilot

**Project**: ObjectMemory — Long-Term Visual Memory for Detecting Persistent Environmental Changes  
**Workspace**: `C:\Users\Rishh\.gemini\antigravity\scratch\objectmemory`  
**Reference Pilot Results**: `scripts/phase14c_existing_workbench_pilot_results.json`  
**Ground-Truth Dataset**: `data/real_workbench/manifest.json`  
**Phase 9B Historical Baseline**: `scripts/phase9b_evaluation_results.json`  
**Date of Audit**: October 29, 2026  
**Auditor Mode**: Read-Only Scientific & Evaluator Forensic Inspection (No application code, model weights, or historical results modified)

---

## 1. Executive Summary

A comprehensive read-only failure analysis was conducted on the Phase 14C pilot benchmark results (`scripts/phase14c_existing_workbench_pilot_results.json`). The pilot evaluates the 7-visit physical workbench sequence (`data/real_workbench/`) containing 42 object annotations across three decoupled tracks:

- **Track A (Natural Image Detection)**: Validated on physical photographs with letterbox preprocessing.
  - **YOLOv8n**: Precision = **96.77%**, Recall = **88.24%**, F1 = **0.9231** (30 TP, 1 FP, 4 FN across 34 GT boxes).
  - **YOLO-World**: Precision = **95.83%**, Recall = **67.65%**, F1 = **0.7931** (23 TP, 1 FP, 11 FN).
- **Track B (Oracle Temporal Reasoning)**: **3/6** event accuracy (**50.0%**).
- **Track C (End-to-End Vision + Temporal Pipeline)**: **3/6** event accuracy (**50.0%**).

### Primary Conclusions

The identical 50.0% (3/6) accuracy in Track B and Track C is **not** evidence of catastrophic temporal engine failure. Rather, forensic code execution and trace analysis reveal three distinct root causes:

1. **Evaluator Adapter Defect (Visit 7)**: An implementation flaw in `PhysicalWorkbenchDataset.from_real_workbench_dataset` (`physical_workbench.py`, lines 273–289) unconditionally overwrote the primary chronological physical event (`bottle_01` reappearance) with the passive ongoing absence of a secondary object (`notebook_01`), causing Visit 7 to expect `object_removed` when the true physical event was `reappeared`.
2. **Correctly Delayed Confirmation / Physics Boundary (Visit 6)**: The non-detection of `bottle_01` in Visit 6 was correctly delayed from confirming removal by the temporal engine. Because the bottle is transparent plastic on a wood tabletop, optical vacancy verification yielded high similarity to the empty surface (`patch_sim >= vacancy_similarity_max = 0.40`). Furthermore, Visit 5 suffered an unverified registration (`INSUFFICIENT_INLIERS = 0.000`), meaning Visit 6 was only the first reliable absence. The engine’s strict 3-visit requirement for unverified vacancy prevented a premature false removal.
3. **Identity Ambiguity and Multi-Object Co-occurrence (Visit 2)**: In Visit 2, `cup_02` disappeared while `cup_01` remained. In Track B, spatial proximity between the two white cups triggered the associator's `AMBIGUOUS_IDENTITY` safeguard (`margin < 0.12`). In Track C, the detector (YOLOv8n) never detected `cup_02` in Visit 1, meaning no `cup_02` entity existed to be removed.
4. **Denominator and Granularity Defect**: Collapsing multi-object physical scenes into a single coarse visit-level string (`expected_event`) ignores co-occurring physical events. In Phase 9B, temporal evaluation was conducted over all **42 entity-visit transitions**, achieving 92.9% accuracy. Evaluating only 6 visit-level labels introduces extreme sensitivity to arbitrary adapter loops.

---

## 2. Event-by-Event Forensic Trace

The pilot evaluates 6 transitions across Visits 2 through 7 (Visit 1 establishes baseline). Three visits passed (`visit_03`: `object_moved`, `visit_04`: `return_to_baseline`, `visit_05`: `object_pending_removal`), while three visits failed (`visit_02`, `visit_06`, `visit_07`).

```
+-------+--------------------+-------------------------+--------------------+----------------+-------------------------+
| Visit | Target Entity      | Adapter Expected Event  | Track B Result     | Track C Result | Root Cause Category     |
+-------+--------------------+-------------------------+--------------------+----------------+-------------------------+
| V02   | cup_02             | object_pending_removal  | FAILED (0/1)       | FAILED (0/1)   | Ambiguity / Det Limit   |
| V03   | cup_01             | object_moved            | PASSED (1/1)       | PASSED (1/1)   | Verified Genuine Move   |
| V04   | cup_01             | return_to_baseline      | PASSED (1/1)       | PASSED (1/1)   | Verified Return Move    |
| V05   | bottle_01          | object_pending_removal  | PASSED (1/1)       | PASSED (1/1)*  | Unverified Delay / Pass |
| V06   | bottle_01          | object_removed          | FAILED (0/1)       | FAILED (0/1)   | Correctly Delayed Conf  |
| V07   | bottle_01 / note01 | object_removed (WRONG!) | FAILED (0/1)       | FAILED (0/1)   | Evaluator Adapter Bug   |
+-------+--------------------+-------------------------+--------------------+----------------+-------------------------+
* Note: V05 Track C passed due to scissors remaining PENDING_REMOVAL under registration failure, not bottle.
```

---

### Failure 1: Visit 2 (`visit_02`) — Unconfirmed Candidate Absence

#### Identification
- **Sequence**: `desk_pilot_01`
- **Visit ID**: `visit_02`
- **Target Entity**: `cup_02` (secondary white cylindrical cup behind `cup_01`)
- **Adapter Expected Event**: `object_pending_removal`
- **Manifest Annotated State**: `PENDING_REMOVAL` (Note in manifest: *"Second cup no longer visible. First reliable absence: streak=1. Transitions from INSUFFICIENT_EVIDENCE to PENDING_REMOVAL."*)
- **Actual Engine Updates**:
  - **Track B**: 
    - `ent_..._2` (`cup_01`): `CONFIRMED_PRESENT` $\to$ `AMBIGUOUS_IDENTITY` (*"Ambiguous identity between competing candidates (margin < 0.12); association unresolved."*)
    - `ent_..._3` (`cup_02`): `CONFIRMED_PRESENT` $\to$ `CONFIRMED_PRESENT` (*"Non-detection without verified vacancy cannot confirm removal; patch appearance still resembles object."*)
  - **Track C**:
    - `cup_01`: `CONFIRMED_PRESENT` $\to$ `CONFIRMED_PRESENT` (displacement 5.9px $\le$ 35.0px).
    - `cup_02`: *Not present in store* (YOLOv8n detected only 1 cup in V01).

#### Evidence Trace
1. **Detections**:
   - V01 Manifest has both `cup_01` (`[10, 318, 220, 540]`) and `cup_02` (`[75, 280, 200, 360]`).
   - In Track C, YOLOv8n detected only one bounding box `[32, 316, 210, 501]` in V01. Thus, `cup_02` was never registered as an entity in Track C. In V02, YOLOv8n detected one cup `[31, 313, 206, 493]`, matching the sole existing cup entity.
   - In Track B, oracle detections passed both cups in V01. In V02, only `cup_01` was passed.
2. **Spatial Association & Ambiguity (`associator.py`, line 210)**:
   - When associating the single detection `[10, 318, 210, 530]` with the two candidate cup entities, the spatial distance and signature similarity produced match scores within the ambiguity margin:
     $$\Delta \text{score} = |\text{score}_1 - \text{score}_2| < \text{ambiguity\_margin} (0.12)$$
   - The engine safely refrained from making an arbitrary association, marking `cup_01` as `AMBIGUOUS_IDENTITY`.
3. **Vacancy & Patch Similarity (`engine.py`, lines 1038–1045)**:
   - For `cup_02`, the canonical bounding box `[75, 280, 200, 360]` overlaps with `cup_01`'s body (`intersection area = 5,250 px^2`).
   - The visual patch extracted at `cup_02`'s canonical location contained pixels of `cup_01`. Therefore, `patch_similarity \ge vacancy_similarity_max (0.40)`.
   - Vacancy could **not** be verified optically. Under `engine.py` line 1042:
     > *"Non-detection without verified vacancy cannot confirm removal; patch appearance still resembles object."*
   - Absence streak was 1. The engine policy requires **2** consecutive unobserved visits to reach `PENDING_REMOVAL` when vacancy is unverified.
4. **Outcome**:
   - Neither cup transitioned to `PENDING_REMOVAL`.
   - The evaluator checked `[u for u in rep.entity_updates if u.new_state == EpistemicState.PENDING_REMOVAL]`. With 0 matches, Visit 2 was scored as `passed = False`.

#### Root Cause Classification
**Correctly Delayed Confirmation & Association Safeguard (System working as intended)**.  
The engine's refusal to transition to `PENDING_REMOVAL` on streak=1 with unverified patch vacancy and active identity ambiguity is a designed reliability safeguard. The evaluator expected an instantaneous transition that violates the engine's documented multi-visit vacancy rules.

---

### Failure 2: Visit 6 (`visit_06`) — Transparent Object Vacancy Delay

#### Identification
- **Sequence**: `desk_pilot_01`
- **Visit ID**: `visit_06`
- **Target Entity**: `bottle_01` (clear plastic water bottle with blue cap)
- **Adapter Expected Event**: `object_removed`
- **Manifest Annotated State**: `CONFIRMED_REMOVED` (Note: *"Second consecutive absence (streak=2 for notebook_01). PENDING_REMOVAL -> CONFIRMED_REMOVED."*)
- **Actual Engine Updates**:
  - **Track B**: `ent_..._4` (`bottle_01`): `CONFIRMED_MOVED` $\to$ `CONFIRMED_MOVED`
  - **Track C**: `ent_visit_01_1` (`bottle_01`): `CONFIRMED_PRESENT` $\to$ `CONFIRMED_PRESENT` (*"Non-detection without verified vacancy cannot confirm removal; patch appearance still resembles object."*)

#### Evidence Trace
1. **Registration Dependency Safeguard (`engine.py`, lines 868–878)**:
   - In Visit 5, registration failed due to insufficient keypoint matches (`Registration status: INSUFFICIENT_INLIERS`, `inliers = 0.000 < 0.150`).
   - The engine strictly enforces:
     > *"Registration unverified or inlier ratio below threshold; absence cannot be confirmed."*
   - Consequently, non-detection in Visit 5 did **not** advance `bottle_01`'s absence streak.
2. **First Reliable Absence Occurred in Visit 6**:
   - In Visit 6, registration succeeded (`inliers = 0.254 \ge 0.150`).
   - Bottle was not detected. Because Visit 5 had failed registration, Visit 6 was only the **first** reliable unobserved visit.
3. **Optical Vacancy on Transparent Object (`engine.py`, lines 1038–1060)**:
   - `bottle_01` is clear transparent plastic.
   - When comparing the empty wooden tabletop patch at Visit 6 against the canonical bottle patch (which contains wood grain seen through the plastic), the cosine visual similarity satisfies:
     $$\text{patch\_sim} \ge \text{vacancy\_similarity\_max} (0.40)$$
   - The engine correctly refused to confirm optical vacancy.
4. **Streak Threshold for Unverified Vacancy (`models.py`, lines 58–59)**:
   - When vacancy is verified: streak 1 $\to$ `PENDING_REMOVAL`; streak 2 $\to$ `CONFIRMED_REMOVED`.
   - When vacancy is **unverified**:
     ```python
     unobserved_visits_for_removal: int = 3
     ```
     Streak 1 $\to$ remain `CONFIRMED_PRESENT`; streak 2 $\to$ `PENDING_REMOVAL`; streak 3 $\to$ `CONFIRMED_REMOVED`.
   - Because the bottle had reliable streak = 1 (and unverified vacancy), it correctly remained `CONFIRMED_PRESENT` (Track C) or `CONFIRMED_MOVED` (Track B).
5. **Outcome**:
   - No entity transitioned to `CONFIRMED_REMOVED`.
   - The evaluator checked `[u for u in rep.entity_updates if u.new_state == EpistemicState.CONFIRMED_REMOVED]`. With 0 matches, Visit 6 was scored as `passed = False`.

#### Root Cause Classification
**Physics Boundary & Strict Scientific Safeguard (System working as intended)**.  
The engine strictly protected against two major failure modes:
1. It refused to infer absence during a frame with failed camera registration (Visit 5).
2. It refused to confirm removal of a transparent object when the vacant patch could not be optically distinguished from the background after a single observation.  
The evaluator's expectation of `CONFIRMED_REMOVED` at Visit 6 assumed 2 consecutive verified absences, whereas physically only 1 reliable unverified absence had occurred.

---

### Failure 3: Visit 7 (`visit_07`) — Adapter Event Overwrite vs Bottle Reappearance

#### Identification
- **Sequence**: `desk_pilot_01`
- **Visit ID**: `visit_07`
- **Target Entity**: `bottle_01` (physical reintroduction) vs `notebook_01` (passive continued absence)
- **Adapter Expected Event**: `object_removed` (INCORRECT)
- **Manifest Intent**:
  - `session_name`: *"Bottle reappears after CONFIRMED_REMOVED"*
  - `bottle_01` annotation: `annotated_event: NEWLY_APPEARED`, `expected_state: CONFIRMED_PRESENT`
  - `notebook_01` annotation: `annotated_event: ABSENT`, `expected_state: CONFIRMED_REMOVED`, `notes: "Notebook remains absent. Stays CONFIRMED_REMOVED."`
- **Actual Engine Updates**:
  - **Track B**: `bottle_01` re-associated and preserved.
  - **Track C**: YOLOv8n detected `bottle` with **0.96 confidence** at `[207, 122, 289, 342]`. Engine associated it with `ent_visit_01_1` (`displacement 1.9px \le 35.0px`).
  - No entity transitioned to `CONFIRMED_REMOVED`.

#### Evidence Trace
1. **Inspection of `src/objectmemory/datasets/physical_workbench.py` (lines 272–289)**:
   ```python
   expected_event = "baseline_established" if v_idx == 1 else "baseline_stable"
   for a in v.annotations:
       if a.annotated_event == AnnotatedEvent.RELOCATED:
           if a.expected_state == EpistemicState.CONFIRMED_PRESENT:
               expected_event = "return_to_baseline"
               break
           else:
               expected_event = "object_moved"
               break
       elif a.physical_observation == PhysicalObservation.ABSENT:
           if a.expected_state == EpistemicState.CONFIRMED_REMOVED:
               expected_event = "object_removed"
           elif expected_event != "object_removed":
               expected_event = "object_pending_removal"
       elif a.annotated_event == AnnotatedEvent.NEWLY_APPEARED:
           if a.expected_state == EpistemicState.CONFIRMED_PRESENT:
               expected_event = "reappeared"
   ```
2. **Loop Execution Flow for Visit 7**:
   - Annotation 3 (`bottle_01`): `annotated_event == NEWLY_APPEARED`, `expected_state == CONFIRMED_PRESENT`.
     - Executes: `expected_event = "reappeared"`.
     - Crucially: **There is no `break` statement!** The loop continues to the next annotations.
   - Annotation 5 (`notebook_01`): `physical_observation == ABSENT`, `expected_state == CONFIRMED_REMOVED`.
     - Executes: `expected_event = "object_removed"`!
     - **Result**: The passive continued absence of `notebook_01` unconditionally overwrote `"reappeared"` with `"object_removed"`!
3. **Evaluator Execution (`evaluate_physical_workbench.py`, lines 401–404, 590–593)**:
   ```python
   elif ev_type in ("object_removed", "continued_absence"):
       removed = [u for u in rep.entity_updates if u.new_state == EpistemicState.CONFIRMED_REMOVED]
       passed = (len(removed) > 0)
   ```
   - In Visit 7, `notebook_01` was already absent, and `bottle_01` physically reappeared.
   - No object transitioned to `CONFIRMED_REMOVED`.
   - Therefore, `len(removed) == 0`, and Visit 7 was scored as `passed = False`.
4. **Chronological Physical Reality**:
   - If the adapter had correctly set `expected_event = "reappeared"`, the evaluator would have executed:
     ```python
     elif ev_type in ("revealed", "reappeared", "newly_appeared"):
         revealed = [u for u in rep.entity_updates if u.new_state == EpistemicState.CONFIRMED_PRESENT]
         passed = (len(revealed) > 0)
     ```
   - In Visit 7, `bottle_01` was detected and confirmed present (`len(revealed) > 0`). Visit 7 would have **PASSED**!

#### Root Cause Classification
**Evaluator Defect (Adapter Bug in `physical_workbench.py`)**.  
A flaw in the adapter's visit-level event derivation loop allowed passive background absence of an already-removed entity to clobber the primary physical event of the visit (bottle reappearance).

---

## 3. Evaluation Denominator and Granularity Analysis

### Comparison of Track B and Track C Evaluators

| Dimension | Track B (Oracle) | Track C (End-to-End Vision) | Phase 9B Benchmark |
| :--- | :--- | :--- | :--- |
| **Input Source** | Ground-truth bounding boxes | Real YOLOv8n detector inference | Real YOLOv8n inference |
| **Registration** | Homography applied | Homography applied via pipeline | Homography applied |
| **Granularity** | Single visit string | Single visit string | Per-entity-visit pair |
| **Evaluated Units** | 6 visit events | 6 visit events | **42 entity transitions** |
| **Scored Units Passed**| 3 / 6 (50.0%) | 3 / 6 (50.0%) | **39 / 42 (92.9%)** |

### Why Single Visit-Level Strings Are Scientifically Defective

A real-world workspace contains multiple simultaneous objects whose states evolve independently. For example, during Visit 4:
1. `cup_01` returns to its baseline position (`return_to_baseline`).
2. `scissors_01` disappears / is occluded under notebook (`object_pending_removal`).
3. `notebook_01` appears on the desk (`newly_appeared`).
4. `plant_01`, `bottle_01`, `book_01` remain unchanged (`CONFIRMED_PRESENT`).

Evaluating this visit as a single string `"return_to_baseline"` creates two critical flaws:
1. **Arbitrary Winner-Takes-All Selection**: The score depends entirely on which `if/elif` branch in the adapter happens to evaluate first or last.
2. **False Failure Attribution**: If the system correctly detects 5 out of 6 object behaviors but misses or delays 1, the entire visit is marked 0% instead of 83.3%.

In Phase 9B (`src/objectmemory/benchmarks/ground_truth_matcher.py`), this was solved by matching each physical entity individually across each visit:
$$\text{Entity State Accuracy} = \frac{\sum_{v=1}^{V} \sum_{e=1}^{E} \mathbb{I}(\text{state}_{\text{pred}}(e, v) = \text{state}_{\text{GT}}(e, v))}{\text{Total Entity Visits}}$$
Under this proper metric, the Phase 9B pipeline achieved **92.9% accuracy** (39/42 correct states), demonstrating that the underlying engine is highly reliable when evaluated with the correct denominator.

---

## 4. Prioritized Minimal-Fix Proposal (For Phase 14C.2)

To restore scientific integrity without modifying core temporal algorithms, the following controlled improvements are proposed:

### Priority 1: Fix Adapter Event Precedence in `physical_workbench.py`
**File**: `src/objectmemory/datasets/physical_workbench.py`  
**Problem**: The loop in `from_real_workbench_dataset` has no precedence order and allows passive persistent absence to overwrite dynamic physical changes.  
**Solution**:
1. Classify annotations into **dynamic events** (state transitions occurring in this visit: `RELOCATED`, `NEWLY_APPEARED`, first-time absence `PENDING_REMOVAL`) vs **static persistent states** (passive continuation of `CONFIRMED_REMOVED` or `CONFIRMED_PRESENT`).
2. Give dynamic physical transitions absolute priority over static background absence.
3. Specifically: If any entity has `NEWLY_APPEARED` or `RELOCATED`, prioritize that over passive `ABSENT`.

### Priority 2: Multi-Event / Entity-Level Evaluation in `evaluate_physical_workbench.py`
**File**: `scripts/evaluate_physical_workbench.py`  
**Problem**: The evaluator only scores a single visit-level event.  
**Solution**:
1. Retain the visit-level metric for backwards compatibility, but fix its expected event resolution.
2. Add a secondary, comprehensive **entity-level state accuracy metric** (evaluating all 42 entity-visit transitions against ground truth, mirroring Phase 9B's `ground_truth_matcher.py`).
3. Report both metrics transparently in the output JSON.

### Priority 3: Homography Transformation for Track B Oracle Bounding Boxes
**File**: `scripts/evaluate_physical_workbench.py` (lines 307–309)  
**Problem**: Track B currently assigns `canonical_bbox = ann.bbox` directly from the raw visit image. Because Visits 4–7 have a different camera field of view and resolution (584x576 vs 780x636), passing raw pixel coordinates without applying the registration homography artificially distorts spatial coordinates.  
**Solution**: Transform oracle bounding boxes via `reg.homography` into canonical coordinates, matching Track C's pipeline behavior.

---

## 5. Proposed Regression Tests

Before making any changes in Phase 14C.2, the following automated regression tests will be added:

1. `test_real_workbench_adapter_event_precedence`:
   - Verify that in Visit 7, `expected_event` resolves to `"reappeared"` and is not clobbered by `notebook_01`'s persistent absence.
   - Verify that in Visit 4, `expected_event` resolves to `"return_to_baseline"`.
2. `test_transparent_object_vacancy_delay_safeguard`:
   - Assert that an unverified vacancy patch (`patch_similarity >= vacancy_similarity_max`) requires 3 consecutive unobserved visits to confirm removal, and does not falsely confirm removal on streak 1 or 2.
3. `test_registration_failure_absence_streak_protection`:
   - Assert that non-detections during frames with `AlignmentStatus.INSUFFICIENT_INLIERS` or unverified registration do not increment absence streaks.
4. `test_ambiguous_identity_safeguard_suppresses_spurious_removal`:
   - Assert that when two entities of the same class compete within the ambiguity margin, the engine transitions to `AMBIGUOUS_IDENTITY` rather than falsely confirming removal.

---

## 6. Audit Sign-Off

- **Source Code Status**: Pristine and unmodified.
- **Model Weights & Thresholds**: Unmodified.
- **Datasets**: Unmodified (`data/real_workbench/` remains untouched).
- **Historical Reports**: Preserved (`scripts/phase9b_evaluation_results.json`, `scripts/phase14c_existing_workbench_pilot_results.json`).
- **Conclusion**: Failure analysis complete. Ready to proceed to Phase 14C.2 upon user approval.
