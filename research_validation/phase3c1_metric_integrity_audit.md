# ObjectMemory — Phase 3C.1: Independent Benchmark Metric Integrity Audit
**Project:** ObjectMemory: Long-Term Visual Memory for Detecting Persistent Environmental Changes  
**Workspace:** `C:\Users\Rishh\.gemini\antigravity\scratch\objectmemory`  
**Phase:** 3C.1 (Inspection-Only Metric & Runner Integrity Audit)  
**Status:** AUDIT COMPLETE — VERDICT: **NO-GO FOR FINAL EXPERIMENTS** (Conditional Go for Pilot Harness Testing Only)  

---

## Executive Summary

An independent, line-by-line inspection audit was conducted on the Phase 3C benchmark evaluation runner and supporting modules:
- `src/objectmemory/benchmarks/runner.py`
- `src/objectmemory/benchmarks/config_factory.py`
- `src/objectmemory/benchmarks/ground_truth_matcher.py`
- `src/objectmemory/baseline/comparator.py`
- `src/objectmemory/benchmarks/manifest.py`
- `tests/test_benchmark_runner.py`

While the architectural structure (decoupled configuration factory, manifest path safety, temporary database isolation, and error handling) is cleanly engineered, the mathematical audit revealed **four CRITICAL architectural defects** in ground-truth matching and metric calculation that invalidate scientific conclusions if used in final benchmark evaluation.

### Summary of Audit Findings

| Category | Item / Metric | Severity | Primary Defect / Scientific Risk |
|---|---|---|---|
| **Matching** | Future State Leakage in Historical State Resolution | **CRITICAL** | Line 830: Historical visit evaluation falls back to `store.get_entity(sys_id).current_state`, leaking the final visit state ($T$) into past visits ($t < T$). Newly created entities in Visit 1 have zero updates and always read the end-of-sequence state. |
| **Matching** | Initial Box vs. Final Box Bipartite Matching Mismatch | **CRITICAL** | Lines 974–1005: GT bounding box is locked to Visit 1, while system entity box is taken from Visit $T$. Relocated objects ($>150$ px displacement) fail matching and are erroneously marked `uninstantiated_gt`. |
| **Matching** | Coordinate Frame Mismatch in Frame Matches | **CRITICAL** | Line 779: Computes IoU between raw query-image bbox (`ann.bounding_box`) and canonical keyframe bbox (`ent.canonical_bbox`) without homography projection. Fails under any camera displacement or viewpoint shift. |
| **Metrics** | Optimistic Censoring in Confirmation Delay | **CRITICAL** | Lines 925–929: Unconfirmed (missed) change events are completely excluded from the denominator. A system confirming 1 of 10 removals in 1 visit reports a "perfect" 1.0-visit delay, hiding a 90% failure rate. |
| **Metrics** | Unmatched-Observation Blindness in Identity Consistency | **CRITICAL** | Lines 894–906: Denominator sums only *detected* visits ($\sum \max / \sum \text{matched}$). An object present in 10 visits but detected in only 2 reports 100% identity consistency, ignoring 80% track loss. |
| **Metrics** | Duplicate Track Cancellation via Aggregate Subtraction | **HIGH** | Lines 914–916: Uses $\max(0, |\text{Sys}| - |\text{GT}|)$. If one object is undetected and another is fragmented into 2 tracks, $2 - 2 = 0$. Fails to detect track duplication and fragmentation. |
| **Metrics** | False Removal Rate vs. False Discovery Proportion | **HIGH** | Lines 848–852: FRR is calculated as False Positive Rate ($FP / (TN + FP)$). In long stationary sequences, the large present denominator dilutes false removals to $<1\%$, while False Discovery Proportion ($FP / (TP + FP)$) is unmeasured. |
| **B0 Baseline** | Missing Standard Object Detection Metrics | **HIGH** | Lines 573–679: B0 fails to calculate standard detector metrics (precision, recall, F1, mAP). Instead, it outputs misleading zero scores ($0.0\%$) for temporal removal and movement metrics it cannot compute. |
| **Persistence** | SQLite `:memory:` Inability to Verify Restart Persistence | **MEDIUM** | Line 694: Exclusively uses `:memory:`. In-memory databases are destroyed on process termination and cannot test cross-restart serialization or state recovery. |
| **Provenance** | Omission of Manifest Version and Detector Settings in Fingerprint | **LOW** | Lines 540–554: Result omits `manifest_version` and detector confidence threshold from the configuration fingerprint. |

---

## 1. B0 Detection Evaluation Audit

### 1.1 Structural Compliance
- **Per-Frame Raw Evaluation**: Lines 589–591 correctly invoke `detector.detect(img)` per observation.
- **Statelessness**: B0 does not initialize a `SQLiteMemoryStore`, does not access historical entities, and does not maintain persistent track identifiers across frames.
- **One-to-One Matching**: Lines 601–614 implement greedy bipartite matching based on class label and $\text{IoU} \ge 0.50$, descending by IoU.

### 1.2 Identified Deficiencies in B0

#### Defect 1.1: Omission of Standard Object Detection Metrics
- **Severity**: **HIGH**
- **Location**: `runner.py`, lines 573–679 (`_execute_b0_sequence`)
- **Nature**: B0 is the memoryless detector baseline. Its scientific purpose is to answer: *"What is the raw detection performance before temporal memory is applied?"*
- **Source Code Finding**:
  `_execute_b0_sequence` only computes pseudo-epistemic state accuracy (`CONFIRMED_PRESENT` vs. `INSUFFICIENT_EVIDENCE`), counting `correct_state_evals`. It does **not** compute:
  - Detection Precision: $TP / (TP + FP)$
  - Detection Recall: $TP / (TP + FN)$
  - Detection F1 Score: $2 \cdot \frac{P \cdot R}{P + R}$
  - Mean Average Precision (mAP)
  Without detection precision and recall, it is impossible to determine whether downstream temporal reasoning errors stem from detector failure (missed bounding box / spurious false positive) or temporal association failure.
- **Recommended Correction**:
  Accumulate frame-level $TP, FP, FN$ across observations and report `detection_precision`, `detection_recall`, and `detection_f1` explicitly in `BenchmarkMetrics`.

#### Defect 1.2: Misleading Zero Scores for Temporal Metrics B0 Cannot Compute
- **Severity**: **HIGH**
- **Location**: `runner.py`, lines 635–646, 476–487
- **Source Code Finding**:
  ```python
  # runner.py:635
  if expected_state != EpistemicState.CONFIRMED_REMOVED.value:
      den_frr += 1  # num_frr remains 0
  if expected_state == EpistemicState.CONFIRMED_REMOVED.value:
      den_rr += 1   # num_rr remains 0
  if expected_state in (EpistemicState.CONFIRMED_PRESENT.value, "STATIONARY"):
      den_umr += 1  # num_umr remains 0
  ```
  Because B0 never predicts `CONFIRMED_REMOVED` or `CONFIRMED_MOVED`, the runner computes:
  - `false_removal_rate = 0.0%`
  - `removal_recall = 0.0%`
  - `unsupported_movement_rate = 0.0%`
- **Scientific Impact**:
  Reporting `false_removal_rate = 0.0%` for B0 is scientifically misleading. A reader or reviewer sees a table where B0 achieves "0.0% false removals", making a memoryless detector appear superior to temporal systems in preventing false alarms. In truth, B0 cannot predict removal at all.
- **Recommended Correction**:
  Set `false_removal_rate = None`, `removal_recall = None`, and `unsupported_movement_rate = None` for B0, logging them in `unscorable_cases`:
  `"Temporal change metrics (FRR, RR, UMR) are not defined for stateless memoryless detection."`

---

## 2. Metric Mathematical Audit

### 2.1 Detailed Specification per Metric

#### Metric 1: False Removal Rate ($FRR$)
- **File & Lines**: `runner.py:848–853`, `476–479`
- **Current Formulation**:
  $$\text{FRR} = \frac{\sum_{i} \mathbb{I}(\text{GT}_i \neq \text{REMOVED} \land \text{Pred}_i = \text{CONFIRMED\_REMOVED})}{\sum_{i} \mathbb{I}(\text{GT}_i \neq \text{REMOVED})}$$
- **Unit of Evaluation**: Entity-visit.
- **Severity**: **HIGH** (Confused with False Positive Rate; omits False Discovery Proportion).
- **Mathematical Counterexample**:
  Suppose a sequence has 1,000 entity-visits of present objects and 10 removal visits.
  The system falsely declares `CONFIRMED_REMOVED` on 5 present entity-visits.
  - Calculated FRR (False Positive Rate): $\frac{5}{1000} = 0.5\%$.
  - True False Discovery Proportion ($FDP$): $\frac{FP}{TP + FP} = \frac{5}{5} = 100\%$.
  The report claims "False Removal Rate: 0.5%", creating the illusion of high precision, when in reality 100% of removal alerts were false alarms.
- **Recommended Correction**:
  Retain False Positive Rate as `false_removal_rate_on_present`, but add `removal_false_discovery_proportion`:
  $$\text{FDP}_{\text{removal}} = \frac{FP_{\text{removal}}}{TP_{\text{removal}} + FP_{\text{removal}}}$$

---

#### Metric 2: Removal Recall ($RR$)
- **File & Lines**: `runner.py:854–860`, `480–483`
- **Current Formulation**:
  $$\text{RR} = \frac{\sum_{i} \mathbb{I}(\text{GT}_i = \text{REMOVED} \land \text{Pred}_i = \text{CONFIRMED\_REMOVED})}{\sum_{i} \mathbb{I}(\text{GT}_i = \text{REMOVED})}$$
- **Unit of Evaluation**: Entity-visit.
- **Severity**: **MEDIUM** (Visit-level vs. Event-level discrepancy).
- **Audit Analysis**:
  Because B3 enforces a mandatory streak requirement ($k=2$ consecutive vacant visits), an object removed at Visit 2 is in `PENDING_REMOVAL` at Visit 2, and becomes `CONFIRMED_REMOVED` only at Visit 3.
  If the sequence ends at Visit 3 (2 removal visits total):
  - Visit 2: GT = `CONFIRMED_REMOVED`, Pred = `PENDING_REMOVAL` (Missed).
  - Visit 3: GT = `CONFIRMED_REMOVED`, Pred = `CONFIRMED_REMOVED` (Hit).
  - Visit-level Removal Recall: $\frac{1}{2} = 50.0\%$.
  The system functioned exactly as designed, yet visit-level recall penalizes the necessary confirmation streak.
- **Recommended Correction**:
  Report both:
  1. `visit_removal_recall`: $\frac{\text{Confirmed Removal Visits}}{\text{Total Ground-Truth Removal Visits}}$
  2. `event_removal_recall`: $\frac{\text{Unique Removal Events Confirmed}}{\text{Total Unique Removal Events}}$

---

#### Metric 3: Unsupported Movement Rate ($UMR$)
- **File & Lines**: `runner.py:861–867`, `484–487`
- **Current Formulation**:
  $$\text{UMR} = \frac{\sum_{i} \mathbb{I}(\text{GT}_i = \text{STATIONARY} \land \text{Pred}_i = \text{CONFIRMED\_MOVED})}{\sum_{i} \mathbb{I}(\text{GT}_i = \text{STATIONARY})}$$
- **Unit of Evaluation**: Entity-visit.
- **Severity**: **PASS** (Correctly conditioned on stationary ground-truth visits).

---

#### Metric 4: Identity Consistency ($IC$)
- **File & Lines**: `runner.py:888–907`, `488–495`
- **Current Formulation**:
  $$IC = \frac{\sum_{g} \max_{s} N(g, s)}{\sum_{g} \sum_{s} N(g, s)}$$
- **Unit of Evaluation**: Matched detection observations.
- **Severity**: **CRITICAL** (Ignores unmatched / missed observations).
- **Mathematical Counterexample**:
  Consider entity $G_1$ present in ground truth across 10 consecutive visits.
  - Visit 1: Detected and assigned track $S_1$.
  - Visits 2–9: Detector misses the object, or association fails (8 missed visits).
  - Visit 10: Detected and assigned track $S_1$.
  In `runner.py`:
  `matched_seq = ['S1', 'S1']`
  $\max_s N(g, s) = 2$
  $\sum_s N(g, s) = 2$
  $$\text{Calculated } IC = \frac{2}{2} = 100.0\%$$
  The report announces: *"Identity Continuity: 100.0% (perfect track consistency!)"*, completely ignoring that the entity had zero continuity for 80% of the experiment.
- **Recommended Correction**:
  The denominator must be the total number of visits where the ground-truth entity was physically present and observable:
  $$IC_{\text{ground\_truth}} = \frac{\sum_{g} \max_{s} N(g, s)}{\sum_{g} N_{\text{present\_visits}}(g)}$$

---

#### Metric 5: ID Switches ($IDS$)
- **File & Lines**: `runner.py:908–912`, `496`
- **Current Formulation**:
  Counts transitions in `matched_seq` where $s_{k} \neq s_{k-1}$.
- **Unit of Evaluation**: Consecutive detected transitions.
- **Severity**: **MEDIUM** (Collapses non-adjacent visits across detection gaps).
- **Audit Analysis**:
  If an entity is detected at Visit 1 ($S_1$), absent/missed in Visits 2–5, and redetected at Visit 6 ($S_2$), `matched_seq` is `['S1', 'S2']`. This registers as 1 switch across 1 transition. While an ID switch did occur, it treats a 5-visit gap identically to an immediate frame-to-frame identity flip.
- **Recommended Correction**:
  Distinguish immediate frame-to-frame ID switches from reappearance track re-allocations by checking visit index adjacency ($\Delta v = 1$ vs. $\Delta v > 1$).

---

#### Metric 6: Duplicate Identities ($DI$)
- **File & Lines**: `runner.py:914–920`
- **Current Formulation**:
  $$DI = \max\Big(0, \; |\text{System Entities Instantiated}| - |\text{Unique GT Entities Present}|\Big)$$
- **Unit of Evaluation**: Sequence aggregate count.
- **Severity**: **CRITICAL** (Blind to offsetting false negatives and track fragmentation).
- **Mathematical Counterexample**:
  Suppose a sequence contains 2 ground-truth objects: $G_1$ (cup) and $G_2$ (book).
  - System never detects $G_2$ ($G_2$ is uninstantiated, 0 tracks).
  - System detects $G_1$ at Visit 1 as $S_1$, then drops $S_1$ and creates track $S_2$ for $G_1$ at Visit 2 (track fragmentation: 2 tracks for 1 object).
  What does `runner.py` compute?
  $|\text{System Entities}| = |\{S_1, S_2\}| = 2$
  $|\text{Unique GT Entities}| = |\{G_1, G_2\}| = 2$
  $$DI = \max(0, 2 - 2) = 0$$
  The report announces: **0 duplicate identities!** Track fragmentation is completely hidden.
- **Recommended Correction**:
  Compute track fragmentation per ground-truth entity:
  $$DI_{\text{true}} = \sum_{g} \max\Big(0, \; |\{s : \text{assigned}(g, s)\}| - 1\Big)$$
  In the counterexample above: $G_1$ has tracks $\{S_1, S_2\} \implies 2 - 1 = 1$ duplicate track. $G_2$ has $0 \implies 0$. Total $DI = 1$.

---

#### Metric 7: Confirmation Delay ($CD$)
- **File & Lines**: `runner.py:869–875`, `922–930`, `498–501`
- **Current Formulation**:
  $$\text{Mean Delay} = \frac{\sum_{e \in E_{\text{confirmed}}} (t_{\text{confirmed}} - t_{\text{onset}} + 1)}{|E_{\text{confirmed}}|}$$
- **Unit of Evaluation**: Confirmed change events.
- **Severity**: **CRITICAL** (Severe survival bias / optimistic censoring).
- **Mathematical Counterexample**:
  Suppose 10 objects are permanently removed at Visit 2:
  - For Object 1, the system promptly confirms removal at Visit 2 (delay = 1 visit).
  - For Objects 2–10 (9 objects!), the system fails to verify vacancy or registration fails, leaving them in `PENDING_REMOVAL` or `INSUFFICIENT_EVIDENCE` until the sequence terminates at Visit 10.
  What does `runner.py` compute?
  Line 925: `if ev_key in first_change_confirmed:`
  Objects 2–10 are never confirmed, so they are **completely ignored**!
  $$|E_{\text{confirmed}}| = 1$$
  $$\text{Mean Delay} = \frac{1}{1} = 1.0 \text{ visits}$$
  The report proudly states: *"Confirmation Delay: 1.0 visits (rapid confirmation!)"*, while hiding that 90% of removals were never confirmed.
- **Recommended Correction**:
  Must report `confirmation_rate` ($|E_{\text{confirmed}}| / |E_{\text{total}}|$) alongside delay, or compute penalized delay where unconfirmed events are assigned the maximum observation horizon ($T - t_{\text{onset}} + 1$).

---

#### Metric 8: Historical Retrieval Accuracy ($HRA$)
- **File & Lines**: `runner.py:877–886`, `932–934`, `502–505`
- **Unit of Evaluation**: Reappearance / return events.
- **Severity**: **MEDIUM** (Valid formula, but corrupted by coordinate bug in `obs_to_frame_matches`).

---

## 3. Ground-Truth Matching Deep Audit

### Defect 3.1: Future State Leakage in Historical State Resolution
- **Severity**: **CRITICAL**
- **Location**: `runner.py`, lines 813–834
- **Code Path**:
  ```python
  # Step A: Chronological execution processes all visits 1..T
  for v_idx, obs in enumerate(observations, start=1):
      ...
      rep = engine.process_visit(v_in)
      visit_reports[v_idx] = rep

  # Step C: Historical evaluation AFTER all visits have finished
  for v_idx, obs in enumerate(observations, start=1):
      rep = visit_reports.get(v_idx)
      rep_updates = {u.entity_id: u for u in (rep.entity_updates if rep else [])}

      for ann in obs.objects:
          ...
          if sys_id in rep_updates:
              observed_state = rep_updates[sys_id].new_state
          else:
              ent = store.get_entity(sys_id)
              observed_state = ent.current_state if ent else EpistemicState.INSUFFICIENT_EVIDENCE
  ```
- **The Defect**:
  1. In `engine.py:379–403`, when a new entity is created in Visit 1, it is added directly to `store` (`store.add_entity(new_ent)`), but **is never added to `rep.entity_updates`** (which only tracks existing active entities).
  2. Therefore, in Visit 1, `rep.entity_updates` is **always empty** for newly instantiated entities.
  3. Consequently, for Visit 1, `sys_id in rep_updates` evaluates to `False`.
  4. The code falls into `else: ent = store.get_entity(sys_id)`.
  5. Because Step C runs after Step A has executed all $T$ visits, `store.get_entity(sys_id)` retrieves the entity's state at **Visit $T$**!
- **Concrete Failure Case**:
  - Visit 1: Object is present. System creates entity with state `CONFIRMED_PRESENT`.
  - Visit 2: Object is removed. System transitions entity to `CONFIRMED_REMOVED`.
  - In Step C for Visit 1: `rep_updates` is empty. Line 830 queries `store.get_entity(sys_id)`. The store returns `CONFIRMED_REMOVED` (from Visit 2)!
  - The runner compares `CONFIRMED_REMOVED` against GT `CONFIRMED_PRESENT`, flags a **False Removal in Visit 1**, and scores Visit 1 as incorrect!
- **Contrast with `ground_truth_matcher.py`**:
  `ground_truth_matcher.py:178–179` explicitly documented and prevented this bug:
  > *"Missing data: Returns EpistemicState.INSUFFICIENT_EVIDENCE with an explicit missing-data explanation. Never substitutes sys_ent.current_state."*
- **Recommended Correction**:
  Replace line 830 with backward inspection of historical reports, identical to `ground_truth_matcher.py:212–220`:
  ```python
  # Check if entity was created in this visit
  if v_idx == 1 or sys_id in [e.entity_id for e in rep.new_entities]:
      observed_state = EpistemicState.CONFIRMED_PRESENT
  else:
      # Walk backwards through visit_reports[v_idx-1 ... 1]
      for k in range(v_idx - 1, 0, -1):
          prev_rep = visit_reports.get(k)
          prev_updates = {u.entity_id: u for u in (prev_rep.entity_updates if prev_rep else [])}
          if sys_id in prev_updates:
              observed_state = prev_updates[sys_id].new_state
              break
  ```

---

### Defect 3.2: Bipartite Matching Uses Visit 1 GT Box vs. Final Visit Entity Box
- **Severity**: **CRITICAL**
- **Location**: `runner.py`, lines 974–1005 (`_compute_bipartite_mapping`)
- **Code Path**:
  ```python
  # Collect GT box (breaks on first visit found)
  for gid in all_gt_ids:
      for obs in seq.observations:
          for ann in obs.objects:
              if ann.object_id == gid and ann.bounding_box is not None:
                  gt_info[gid] = (ann.class_name.lower(), ann.bounding_box)
                  break
          if gid in gt_info:
              break

  # Compare with system_entities
  for g_idx, gid in enumerate(gt_ids):
      g_class, g_box = gt_info[gid]
      for s_idx, s_ent in enumerate(sys_entities):
          ...
          iou = calculate_bbox_iou(g_box, s_ent.canonical_bbox)
          dist = calculate_centroid_distance(g_box, s_ent.canonical_bbox) or 999.0
  ```
- **The Defect**:
  - `g_box` is locked to the **first visit** where the GT object had a bounding box (Visit 1).
  - `s_ent.canonical_bbox` is taken from `store.list_entities()` at the **end of the sequence** (Visit $T$).
  - In any scenario where an object is persistently relocated (e.g. Phase 14C scenario `S03_persistent_relocation`), the object moved from $(100, 100)$ in Visit 1 to $(400, 400)$ in Visit 3.
  - At Visit $T$, `s_ent.canonical_bbox` is at $(400, 400)$.
  - Distance between `g_box` $(100, 100)$ and `s_ent` $(400, 400)$ is $424.2 \text{ px} > 150 \text{ px}$.
  - The cost is set to $1000.0$.
  - Hungarian matching rejects the pair!
  - The correctly relocated system entity is **unmatched**, and the ground-truth object is classified as **`uninstantiated_gt`**!
- **Recommended Correction**:
  Ground-truth bipartite matching must be solved on the reference/initial visit (Visit 1) using `v1_system_entities` or initial canonical positions (`ent.metadata['initial_canonical_bbox']`), exactly as performed in `ground_truth_matcher.py:122` and `scripts/evaluate_physical_workbench.py:300–314`.

---

### Defect 3.3: Raw Pixel vs. Canonical Coordinate Mismatch in Frame Matches
- **Severity**: **CRITICAL**
- **Location**: `runner.py`, lines 770–786
- **Code Path**:
  ```python
  active_ents = store.list_entities()
  for ann in obs.objects:
      for ent in active_ents:
          if ent.label.lower() == ann.class_name.lower() and ent.canonical_bbox is not None:
              iou = calculate_bbox_iou(ann.bounding_box, ent.canonical_bbox)
  ```
- **The Defect**:
  - `ann.bounding_box` is in **raw query image coordinates**.
  - `ent.canonical_bbox` is in **canonical reference coordinates** (projected via homography $H$).
  - In Visit 1, reference equals query, so $H = I$.
  - In Visit 2+, if the camera moved, tilted, or experienced jitter (e.g. scenario `S07_viewpoint_perturbation`), raw query pixels do not align with canonical coordinates.
  - Computing IoU between unaligned coordinate spaces causes valid tracks to drop to $0.0$ IoU.
  - This corrupts `obs_to_frame_matches`, which in turn corrupts `Identity Consistency`, `ID Switches`, and `Historical Retrieval Accuracy`.
- **Recommended Correction**:
  Transform `ann.bounding_box` to canonical coordinates using `transform_bbox_to_canonical(ann.bounding_box, reg.homography)` before computing IoU against `ent.canonical_bbox`.

---

## 4. Persistence Verification & SQLite Storage Limitations

### 4.1 In-Memory Database Limitation
- **Location**: `runner.py:694`
- **Implementation**:
  ```python
  store = SQLiteMemoryStore(":memory:")
  ```
- **Severity**: **MEDIUM** (Architectural limitation for restart testing).
- **Audit Findings**:
  SQLite `:memory:` databases are purely ephemeral. They reside within process RAM. When `runner.py` finishes a sequence, the connection closes and the database is completely destroyed.
  Consequently, **the current runner cannot test backend restart persistence**. It cannot prove that:
  - If the server restarts or crashes between Visit 2 and Visit 3, entity belief states, keyframes, and observations are recovered intact from disk.
  - SQLite foreign key constraints, index integrity, and migration schemas function properly across cold boots.

### 4.2 Proposed Restart-Persistence Test Design
To test restart persistence safely without touching production databases:
1. Create a temporary file on disk using `tempfile.NamedTemporaryFile(suffix=".db", delete=False)`.
2. Instantiate `store1 = SQLiteMemoryStore(temp_db_path)` and execute Visits 1 and 2.
3. Explicitly close connections: `store1.close()`, `del engine1`, `del store1`.
4. Instantiate a brand new `store2 = SQLiteMemoryStore(temp_db_path)` and `engine2 = TemporalMemoryEngine(store2)`.
5. Verify that `store2.list_entities()` contains all entities from Visits 1 and 2 with identical epistemic states and canonical bounding boxes.
6. Execute Visit 3 using `engine2`.
7. Verify that state progression (e.g. `PENDING_REMOVAL -> CONFIRMED_REMOVED`) transitions seamlessly across the restart boundary.
8. Unlink the temporary database file.

---

## 5. Reproducibility & Fingerprinting Audit

### 5.1 Configuration Fingerprint Audit
- **Files**: `config_factory.py:82–85`, `runner.py:549`
- **Finding**:
  `deterministic_hash` is computed from `BenchmarkConfig.to_json(indent=0)`.
  `BenchmarkConfig.effective_parameters` records 13 engine parameters (`ambiguity_margin`, `vacancy_similarity_max`, `min_registration_inlier_ratio`, etc.).
- **Missing Parameters**:
  The configuration fingerprint does **not** capture:
  - Detector identifier (e.g. `yolov8n` vs. `mock`)
  - Detector confidence threshold (e.g. `0.25` vs. `0.50`)
  - Registration method and feature parameters (ORB n_features=2000, SIFT, etc.)
  - Manifest schema version
- **Severity**: **LOW** (Does not invalidate metrics, but weakens cross-environment hash comparison).

### 5.2 Error Containment & Provenance Separation
- **Finding**:
  - Failed sequences are safely caught and recorded in `result.execution_errors` without aborting the batch (PASS).
  - Exploratory pilot datasets (`manifest.is_pilot_dataset`) are prominently flagged with mandatory warnings preventing equivalence claims (PASS).
  - Partial ablations (`A2`, `A3`) are explicitly flagged with `is_partial_ablation=True` and non-empty limitation strings (PASS).

---

## 6. Audit Verdict and Recommendations

### Scientific Verdict: **NO-GO FOR FINAL BENCHMARK EXPERIMENTS**
*(Conditional Go for internal pilot harness development only)*

The runner in its current form cannot be used to generate published research figures or final benchmark tables because:
1. **Future State Leakage** (Defect 3.1) corrupts historical entity state evaluations whenever state changes occur.
2. **Initial vs. Final Box Bipartite Matching** (Defect 3.2) marks relocated objects as uninstantiated.
3. **Coordinate Frame Mismatch** (Defect 3.3) breaks tracking metrics under camera motion.
4. **Optimistic Censoring in Confirmation Delay** (Defect 2.4) hides unconfirmed removals.
5. **Denominator Blindness in Identity Consistency** (Defect 2.3) ignores missed detections.

### Corrective Action Roadmap (Phase 3C.2)

1. **Fix Historical State Resolution (`runner.py:824–834`)**:
   Implement backwards historical report traversal rather than querying `store.get_entity()` at time $T$.
2. **Fix Bipartite Ground-Truth Matching (`runner.py:974–1021`)**:
   Anchor matching to Visit 1 using initial canonical bounding boxes (`ent.metadata['initial_canonical_bbox']`).
3. **Fix Coordinate Transformation in Frame Matches (`runner.py:770–786`)**:
   Project query bounding boxes into canonical reference coordinates before computing IoU against entity bounding boxes.
4. **Implement Dual Metrics for Confirmation Delay & Removal Rate**:
   Add `confirmation_rate` alongside delay, and report `removal_false_discovery_proportion` alongside False Removal Rate.
5. **Add Detection Precision, Recall, and F1 to B0 Baseline**:
   Report genuine computer vision detector metrics and mark temporal-only metrics as `N/A`.
6. **Implement Restart-Persistence Test**:
   Validate file-backed SQLite state recovery across process lifecycles.
